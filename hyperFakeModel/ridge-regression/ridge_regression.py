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


def get_splits(X, y, seed=42):
    """Split data into train, validation, and test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_ridge_model(X_train, y_train, X_val, y_val):
    """Train Ridge regression model with efficient alpha tuning."""
    # Create alpha range - focus on what matters
    alphas = np.logspace(-3, 3, 25)  # 25 alphas from 0.001 to 1000
    
    # Create pipeline with scaling and RidgeCV
    print("Training Ridge model with cross-validation...")
    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=alphas, cv=5)
    )
    
    # Train on the training set
    model.fit(X_train, y_train)
    
    # Get the best alpha
    best_alpha = model.named_steps['ridgecv'].alpha_
    print(f"Best alpha: {best_alpha:.6f}")
    
    # Make predictions on validation set
    y_val_pred = model.predict(X_val)
    
    return model, y_val_pred


def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
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


def export_model_to_onnx(model, input_dim, target_path):
    """Export the trained Ridge model pipeline to ONNX format."""
    # Define the input type
    initial_type = [('float_input', FloatTensorType([None, input_dim]))]
    
    # Convert to ONNX (model is already a pipeline with scaler + RidgeCV)
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save the model
    onnx.save_model(onnx_model, target_path)
    print(f"Exported model to '{target_path}'")


def main(table_name, func_tag, target_path, db_path):
    """Main training function."""
    # Load data
    X, y = load_data_from_db(db_path, table_name, func_tag)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(X, y)
    
    # Train model
    model, y_val_pred = train_ridge_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    metrics_val = evaluate_model(y_val, y_val_pred)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    metrics_test = evaluate_model(y_test, y_test_pred)
    
    # Print results
    print("=" * 40)
    print(f"Validation results for {func_tag}: {metrics_val}")
    print(f"Test results for {func_tag}: {metrics_test}")
    print("=" * 40)

    # Export to ONNX
    input_dim = X.shape[1]
    export_model_to_onnx(model, input_dim, target_path)


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
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(curr_dir, "..", "..", "benchmarks", "metrics.db")
    db_path = os.path.abspath(args.db_path) if args.db_path else default_db_path
    table_name = args.table

    # Get all function tags from the database
    func_tags = get_function_tags_from_db(db_path, table_name)
    print(f"Found {len(func_tags)} function tags in database: {func_tags}")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(curr_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for func_tag in func_tags:
        func_name = func_tag.split(":")[0]
        target_path = os.path.join(models_dir, f"{func_name}.onnx")
        main(table_name, func_tag, target_path, db_path=db_path)
