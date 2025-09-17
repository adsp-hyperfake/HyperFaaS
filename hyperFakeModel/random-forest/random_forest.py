import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from datetime import datetime
import argparse
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

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
    # 35% test, 50% of remaining for validation
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.35, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(model, X, y):
    """Compute regression metrics for a model and dataset."""
    y_pred = model.predict(X)
    return {
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred),
        "R2": r2_score(y, y_pred),
    }


def export_model_to_onnx(model, input_dim, target_path):
    """Export the trained scikit-learn model to ONNX format."""
    # Ensure the output directory exists
    dirpath = os.path.dirname(target_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    initial_types = [("input", FloatTensorType([None, input_dim]))]

    final_types = [("variable", FloatTensorType([None, len(OUTPUT_COLS)]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        final_types=final_types,
        target_opset=15,
    )
    metadata = {
        "features": ",".join(INPUT_COLS),
        "targets": ",".join(OUTPUT_COLS),
        "training_date": datetime.now().isoformat(),
        "output_shape": f"[None, {len(OUTPUT_COLS)}]"
    }
    for k, v in metadata.items():
        onnx_model.metadata_props.add(key=k, value=v)
    with open(target_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported model to '{target_path}'")


def main(table_name, func_tag, target_path, db_path, model_params=None):
    """Main pipeline for training and exporting a random forest model."""
    X, y = load_data_from_db(db_path, table_name, func_tag)

    # Split into train/validation/test
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(X, y, seed=42)

    # Train model with configurable parameters
    if model_params is None:
        model_params = {}
    
    # Set default parameters if not provided
    rf_params = {
        'n_estimators': model_params.get('n_estimators', 100),
        'max_depth': model_params.get('max_depth', None),
        'min_samples_split': model_params.get('min_samples_split', 2),
        'min_samples_leaf': model_params.get('min_samples_leaf', 1),
        'max_features': model_params.get('max_features', 'sqrt'),
        'random_state': model_params.get('random_state', 42),
        'n_jobs': model_params.get('n_jobs', -1)
    }
    
    input_dim = X_train.shape[1]
    model = RandomForestRegressor(**rf_params)
    print(f"Training Random Forest with parameters: {rf_params}")
    
    with tqdm_joblib(tqdm(total=model.n_estimators, desc="Training RF")):
        model.fit(X_train, y_train)

    # Evaluate on validation and test sets
    metrics_val = compute_metrics(model, X_val, y_val)
    metrics_test = compute_metrics(model, X_test, y_test)
    
    # Print results
    print("=" * 40)
    print(f"Validation results for {func_tag}: {metrics_val}")
    print(f"Test results for {func_tag}: {metrics_test}")
    print("=" * 40)

    # Export to ONNX
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
    parser = argparse.ArgumentParser(description="Train RandomForest models and export them to ONNX.")
    parser.add_argument("--db-path", help="Path to the SQLite metrics database.")
    parser.add_argument("--table", default="training_data", help="Name of the table with training data.")
    
    # Random Forest model parameters
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the forest (default: 100)")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth of trees (default: None)")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split an internal node (default: 2)")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples required to be at a leaf node (default: 1)")
    parser.add_argument("--max-features", type=str, default="sqrt", help="Number of features to consider when looking for the best split (default: sqrt)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of jobs to run in parallel (default: -1)")
    
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(curr_dir, "..", "..", "benchmarks", "metrics.db")
    db_path = os.path.abspath(args.db_path) if args.db_path else default_db_path
    table_name = args.table

    # Collect model parameters from args
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'max_features': args.max_features,
        'random_state': args.random_state,
        'n_jobs': args.n_jobs
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