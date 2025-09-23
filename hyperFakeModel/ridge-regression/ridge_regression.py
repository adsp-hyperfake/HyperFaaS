import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import argparse

# Feature and target column definitions
INPUT_COLS = [
    "request_size_bytes",
    "function_instances_count",
    "active_function_calls_count",
    "worker_cpu_usage",
    "worker_ram_usage",
]

OUTPUT_COLS = [
    "function_processing_time_ns",
    "function_cpu_usage",
    "function_ram_usage",
]


def load_data_from_db(db_path, table_name, func_tag):
    """Load feature and target arrays from the SQLite training_data table for a given function tag."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT {', '.join(INPUT_COLS + OUTPUT_COLS)} FROM {table_name} WHERE image_tag = ?"
    df = pd.read_sql_query(query, conn, params=(func_tag,))
    conn.close()

    if df.empty:
        raise ValueError(f"No rows found for tag '{func_tag}'")
    if df.isnull().any().any():
        df = df.dropna()
        
    X = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values
    print(f"Loaded {len(X)} rows for tag '{func_tag}'")
    return X, y


def get_splits(X, y, seed=42, test_size=0.2, val_size=0.25):
    """Split data into train, validation, and test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_ridge_model(X_train, y_train, X_val, y_val, ridge_params=None):
    """Train Ridge regression model with efficient alpha tuning."""
    if ridge_params is None:
        ridge_params = {}
    
    # Set default parameters if not provided
    alpha_min = ridge_params.get('alpha_min', -3)
    alpha_max = ridge_params.get('alpha_max', 3) 
    alpha_num = ridge_params.get('alpha_num', 25)
    cv_folds = ridge_params.get('cv_folds', 5)
    
    # Create alpha range - focus on what matters
    alphas = np.logspace(alpha_min, alpha_max, alpha_num)
    print(f"Using alpha range: {alpha_min} to {alpha_max} with {alpha_num} values")
    
    # Create pipeline with scaling and RidgeCV for hyperparameter selection
    print("Selecting optimal alpha with cross-validation...")
    model_cv = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=alphas, cv=cv_folds)
    )
    
    # Train on the training set to select best alpha
    model_cv.fit(X_train, y_train)
    
    # Get the best alpha
    best_alpha = model_cv.named_steps['ridgecv'].alpha_
    print(f"Best alpha: {best_alpha:.6f}")
    
    # Make predictions on validation set (for reporting)
    y_val_pred = model_cv.predict(X_val)
    
    # Now refit on combined train+val data with the selected alpha
    print("Refitting on combined train+validation data...")
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.vstack([y_train, y_val])
    
    # Create final model with the selected alpha
    from sklearn.linear_model import Ridge
    final_model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Ridge(alpha=best_alpha)
    )
    
    # Fit on all available non-test data
    final_model.fit(X_combined, y_combined)
    
    return final_model, y_val_pred


def evaluate_model(y_true, y_pred):
    """Evaluate model performance with overall metrics only (for backward compatibility)."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def evaluate_multi(y_true, y_pred, col_names):
    """Evaluate multi-output model with both overall and per-column metrics."""
    overall = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2_uniform": r2_score(y_true, y_pred, multioutput="uniform_average"),
        "R2_var_weighted": r2_score(y_true, y_pred, multioutput="variance_weighted"),
    }
    per_col = {}
    for i, c in enumerate(col_names):
        per_col[c] = {
            "MSE": mean_squared_error(y_true[:, i], y_pred[:, i]),
            "RMSE": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "R2": r2_score(y_true[:, i], y_pred[:, i]),
        }
    return overall, per_col


def export_pipeline_to_onnx(pipeline, input_dim, target_path):
    """Export the fitted pipeline directly to ONNX format."""
    initial_type = [('input', FloatTensorType([None, input_dim]))]
    final_type = [('output', FloatTensorType([None, 3]))]  # 3 outputs: runtime, cpu, ram
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, final_types=final_type)
    onnx.save_model(onnx_model, target_path)
    print(f"Exported model to '{target_path}'")


def main(table_name, func_tag, target_path, db_path, model_params=None):
    """Main training function."""
    if model_params is None:
        model_params = {}
        
    # Load data
    X, y = load_data_from_db(db_path, table_name, func_tag)
    
    # Split data with configurable sizes
    test_size = model_params.get('test_size', 0.2)
    val_size = model_params.get('val_size', 0.25) 
    random_state = model_params.get('random_state', 42)
    
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        X, y, seed=random_state, test_size=test_size, val_size=val_size
    )
    
    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model with configurable ridge parameters
    ridge_params = {
        'alpha_min': model_params.get('alpha_min', -3),
        'alpha_max': model_params.get('alpha_max', 3),
        'alpha_num': model_params.get('alpha_num', 25),
        'cv_folds': model_params.get('cv_folds', 5)
    }
    
    model, y_val_pred = train_ridge_model(X_train, y_train, X_val, y_val, ridge_params)
    
    # Evaluate on validation set with detailed metrics
    overall_val, per_col_val = evaluate_multi(y_val, y_val_pred, OUTPUT_COLS)
    
    # Evaluate on test set with detailed metrics
    y_test_pred = model.predict(X_test)
    overall_test, per_col_test = evaluate_multi(y_test, y_test_pred, OUTPUT_COLS)
    
    # Print results
    print("=" * 60)
    print(f"Results for {func_tag}")
    print("=" * 60)
    
    print("\nVALIDATION METRICS:")
    print("Overall:", overall_val)
    print("Per-column:")
    for col, metrics in per_col_val.items():
        print(f"  {col}: {metrics}")
    
    print("\nTEST METRICS:")
    print("Overall:", overall_test)
    print("Per-column:")
    for col, metrics in per_col_test.items():
        print(f"  {col}: {metrics}")
    
    print("=" * 60)

    # Export to ONNX
    input_dim = X.shape[1]
    export_pipeline_to_onnx(model, input_dim, target_path)


def get_function_tags_from_db(db_path, table_name):
    """Get all unique function tags from the database."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT DISTINCT image_tag FROM {table_name}"
    cursor = conn.execute(query)
    func_tags = [row[0] for row in cursor.fetchall()]
    conn.close()
    return func_tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ridge regression models and export them to ONNX.")
    parser.add_argument("--db-path", help="Path to the SQLite metrics database.")
    parser.add_argument("--table", default="training_data", help="Name of the table with training data.")
    
    # Ridge regression model parameters
    parser.add_argument("--alpha-min", type=float, default=-3, help="Minimum alpha exponent for logspace (default: -3)")
    parser.add_argument("--alpha-max", type=float, default=3, help="Maximum alpha exponent for logspace (default: 3)")
    parser.add_argument("--alpha-num", type=int, default=25, help="Number of alpha values to try (default: 25)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    
    # Data splitting parameters
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.25, help="Fraction of remaining data to use for validation (default: 0.25)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility (default: 42)")
    
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(curr_dir, "..", "..", "benchmarks", "metrics.db")
    db_path = os.path.abspath(args.db_path) if args.db_path else default_db_path
    table_name = args.table

    # Collect model parameters from args
    model_params = {
        'alpha_min': args.alpha_min,
        'alpha_max': args.alpha_max,
        'alpha_num': args.alpha_num,
        'cv_folds': args.cv_folds,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'random_state': args.random_state
    }

    # Get all function tags from the database
    func_tags = get_function_tags_from_db(db_path, table_name)
    print(f"Found {len(func_tags)} function tags in database: {func_tags}")
    print(f"Using model parameters: {model_params}")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(curr_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for func_tag in func_tags:
        func_name = func_tag.split(":")[0]
        target_path = os.path.join(models_dir, f"{func_name}.onnx")
        main(table_name, func_tag, target_path, db_path=db_path, model_params=model_params)
