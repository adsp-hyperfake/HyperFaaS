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


def main(table_name, func_tag, target_path, db_path):
    """Main pipeline for training and exporting a random forest model."""
    X, y = load_data_from_db(db_path, table_name, func_tag)

    # Split into train/validation/test
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(X, y, seed=42)

    # Train model
    input_dim = X_train.shape[1]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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