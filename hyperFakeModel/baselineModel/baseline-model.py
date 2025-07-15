import os
import sqlite3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from sklearn.preprocessing import StandardScaler

# --------------------------------------
# Configurable global constants
# --------------------------------------

INPUT_COLS = [
    "request_body_size",
    "function_instances_count",
    "active_function_calls_count",
    "worker_cpu_usage",
    "worker_ram_usage",
]
OUTPUT_COLS = [
    "function_runtime",
    "function_cpu_usage",
    "function_ram_usage",
]

# --------------------------------------
# Load data
# --------------------------------------

def load_data_from_dbs(dbs_dir, table_name):
    """Load data from all SQLite databases in a directory and return a merged DataFrame."""
    all_dfs = []
    query = f"SELECT {', '.join(INPUT_COLS + OUTPUT_COLS + ['function_image_tag'])} FROM {table_name}"
    for filename in os.listdir(dbs_dir):
        if filename.endswith(".db") or filename.endswith(".sqlite"):
            db_path = os.path.join(dbs_dir, filename)
            df = None
            try:
                with sqlite3.connect(db_path) as conn:
                    df = pd.read_sql_query(query, conn)
            except Exception as e:
                print(
                    f"{'-' * 60}\n"
                    f"Error loading data from {db_path}: {e}\n"
                    f"Continuing with remaining databases...\n"
                    f"{'-' * 60}"
                )
            if df is not None and not df.empty:
                all_dfs.append(df)
    if not all_dfs:
        raise EmptyDataError(
            f"No data loaded from any database in directory: {dbs_dir}"
        )
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Clean rows containing zero values (except for "safe" columns)
    safe_columns = ["active_function_calls_count", "function_instances_count"]
    columns_to_clean = [
        column for column in INPUT_COLS + OUTPUT_COLS if column not in safe_columns
    ]
    mask = (combined_df[columns_to_clean] <= 0).any(axis=1)
    cleaned_df = combined_df[~mask]

    return cleaned_df

def filter_data_by_image_tag(df, image_tag, samples=0):
    """Filter the DataFrame by function_image_tag and optionally sample."""
    image_tag_data_only = df[df["function_image_tag"] == image_tag]
    if samples > 0:
        image_tag_data_only = image_tag_data_only.sample(n=samples, random_state=42)
    X = image_tag_data_only[INPUT_COLS].values
    y = image_tag_data_only[OUTPUT_COLS].values
    print(f"Loaded {len(X)} data points for image tag '{image_tag}'.")
    return X, y

def compute_output_means(y):
  """Compute mean values for the output columns."""
  return torch.tensor(np.mean(y, axis=0), dtype=torch.float32)
  
# --------------------------------------
# Model definition
# --------------------------------------

class MeanPredictor(nn.Module):
  def __init__(self, mean_outputs: torch.Tensor, input_dim=5, output_dim=3):
    super().__init__()
    
    self.input_dim = input_dim
    self.output_dim = output_dim
    
    self.register_buffer("mean_outputs", mean_outputs)
    
  def forward(self, x):
    batch_size = x.shape[0]
    return self.mean_outputs.expand(batch_size, -1)

# --------------------------------------
# Validation loss computation
# --------------------------------------  

def compute_validation_loss(model, X_val, y_val):
    """Compute MSE loss of model predictions against actual values."""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32)
        targets = torch.tensor(y_val, dtype=torch.float32)
        predictions = model(inputs)

        mse_loss_fn = nn.MSELoss()
        loss = mse_loss_fn(predictions, targets)

    return loss.item()

# --------------------------------------
# ONNX export
# --------------------------------------

def export(mean_tensor, onnx_path, input_size):
  model = MeanPredictor(mean_tensor)
  model.eval()
  
  dummy_input = torch.zeros(1, model.input_dim)
  
  torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
      "input": {0: "batch_size"},
      "output": {0: "batch_size"}
    },
    opset_version=11
  )
  
  print(f"ONNX model exported to {onnx_path}")
  
# --------------------------------------
# Main execution
# --------------------------------------
  
def build_and_export_mean_model(dbs_dir, table_name, image_tag, onnx_path, sample_size=0):
    df = load_data_from_dbs(dbs_dir, table_name)
    X, y = filter_data_by_image_tag(df, image_tag, sample_size)
    mean_tensor = compute_output_means(y)
    export(mean_tensor, onnx_path, X.shape[1])

if __name__ == "__main__":
    function_tags = [
        "hyperfaas-echo:latest",
        "hyperfaas-thumbnailer-json:latest",
        "hyperfaas-bfs-json:latest"
        ]
    
    for tag in function_tags:
        try:
            build_and_export_mean_model(
                dbs_dir="./training-dbs",
                table_name="training_data",
                image_tag=tag,
                onnx_path=f"{tag.replace(':', '_')}_mean_predictor.onnx",
                sample_size=0 # Set to 0 to use all data, or specify a sample size
            )
        except EmptyDataError as e:
            print(f"Skipping model export for {tag}: {e}")