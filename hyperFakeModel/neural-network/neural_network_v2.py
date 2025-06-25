import os
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

###########
### WIP ###
###########

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

INPUT_COLS = [
    "request_body_size",
    "function_instances_count",
    "active_function_calls_count",
    "worker_cpu_usage",
    "worker_ram_usage",
]

OUTPUT_COLS = ["function_runtime", "function_cpu_usage", "function_ram_usage"]


class CustomDataset(Dataset):
    def __init__(self, X, y, scaler_X=None, scaler_y=None, fit_scalers=True):
        if scaler_X is None:
            self.scaler_X = StandardScaler()
        else:
            self.scaler_X = scaler_X

        if scaler_y is None:
            self.scaler_y = StandardScaler()
        else:
            self.scaler_y = scaler_y

        if fit_scalers:
            self.X = self.scaler_X.fit_transform(X)
            self.y = self.scaler_y.fit_transform(y)
        else:
            self.X = self.scaler_X.transform(X)
            self.y = self.scaler_y.transform(y)

        # convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class MultiOutputNetwork(nn.Module):
    """Multi-output neural network"""

    def __init__(
        self,
        input_dim=5,
        output_dim=3,
        hidden_dims=[32, 16, 8],
        dropouts=[0.3, 0.23, 0.15],
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim, dropout in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        # might need further tuning
        # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


def create_sample_data(n_samples=10000):
    """Create sample data for demonstration purposes"""
    # Just for testing. This will obviously lead to a terrible model evaluation since it's all random!
    data = {
        "request_body_size": np.random.randint(1, 10, n_samples),
        "function_instances_count": np.random.randint(1, 10, n_samples),
        "active_function_calls_count": np.random.randint(1, 10, n_samples),
        "worker_cpu_usage": np.random.randint(1, 10, n_samples),
        "worker_ram_usage": np.random.randint(1, 10, n_samples),
        "function_runtime": np.random.randint(1, 10, n_samples),
        "function_cpu_usage": np.random.randint(1, 10, n_samples),
        "function_ram_usage": np.random.randint(1, 10, n_samples)
    }
    df = pd.DataFrame(data)
    X = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values
    return X, y


def load_data_from_db(db_path, table_nam, func_tag):
    """Load data from SQLite database and return features and targets as numpy arrays."""
    # Read data from database
    query = f"SELECT {', '.join(INPUT_COLS + OUTPUT_COLS)} FROM {table_name} WHERE function_image_tag = '{func_tag}'"
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error loading data: {e}")
    # Split features and targets
    X = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values
    print(f"Loaded {len(X)} rows from database")
    return X, y


def train_epoch(model, train_data_loader, criterion, optimizer):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for data, targets in train_data_loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        # forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_data_loader)


def evaluate_model(model, data_loader, criterion):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            # .cpu() copies from gpu to cpu mem if needed
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return total_loss / num_batches, all_predictions, all_targets


def train_model(
    model,
    train_data_loader,
    val_data_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=100,
    patience=20,
):
    """Train the model with early stopping."""

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("Starting training...")
    print(
        f"Training for up to {num_epochs} epochs with early stopping (patience={patience})"
    )
    print("-" * 60)

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", ncols=100):
        train_loss = train_epoch(model, train_data_loader, criterion, optimizer)

        # Evaluate on validation set
        val_loss, _, _ = evaluate_model(model, val_data_loader, criterion)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            tqdm.write(
                f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stoppage after {epoch + 1} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return train_losses, val_losses


def calculate_metrics(y_true, y_pred, target_names):
    """Calculate regression metrics for each target."""
    metrics = {}

    for i, target_name in enumerate(target_names):
        true_values = y_true[:, i]
        pred_values = y_pred[:, i]

        metrics[target_name] = {
            "MSE": mean_squared_error(true_values, pred_values),
            "RMSE": np.sqrt(mean_squared_error(true_values, pred_values)),
            "MAE": mean_absolute_error(true_values, pred_values),
            "R2": r2_score(true_values, pred_values),
        }

    # Overall metrics (average across all targets)
    overall_mse = mean_squared_error(y_true, y_pred)
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_r2 = r2_score(y_true, y_pred)

    metrics["Overall"] = {
        "MSE": overall_mse,
        "RMSE": np.sqrt(overall_mse),
        "MAE": overall_mae,
        "R2": overall_r2,
    }

    return metrics

def export_model_to_onnx(model, target_path):
    # Save the model
    model.eval()
    dummy_input = torch.zeros(1, 5)
    torch.onnx.export(
        model.to(DEVICE),
        dummy_input.to(DEVICE),
        target_path,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True, # Optimize the model
    )
    print(f"Exported model to {target_path}.")

def predict(model, scaler_X, scaler_y, input_data):
    """Make predictions using the trained model."""
    # ensure input_data is a 2D array
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)

    # standardize inputs
    X_scaled = scaler_X.transform(input_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    X_tensor.to(DEVICE)

    # make prediction
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    # reverse standardization
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return y_pred

def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    
    plt.pause(0.01)

def main(table_name, func_tag, target_path, db_path=None):
    """Main training pipeline."""
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Starting training pipeline for {func_tag} using device {DEVICE}.")

    # Load and preprocess data to tensors
    if db_path is None:
        X, y = create_sample_data()
    else:
        X, y = load_data_from_db(db_path, table_name, func_tag)
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Create datasets and data loaders
    train_dataset = CustomDataset(X_train, y_train, fit_scalers=True)
    val_dataset = CustomDataset(
        X_val,
        y_val,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        fit_scalers=False,
    )
    test_dataset = CustomDataset(
        X_test,
        y_test,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        fit_scalers=False,
    )

    # Data loaders
    batch_size = 32
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model configuration
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    hidden_dims = [32, 16, 8]

    # Create model, loss function, and optimizer
    model = MultiOutputNetwork(input_dim, output_dim, hidden_dims).to(DEVICE)
    criterion = nn.MSELoss()
    # we can also try 1e-5 (simpler) or even 1e-3 (complex) for the weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )
    # Train the model
    train_losses, val_losses = train_model(
        model,
        train_data_loader,
        val_data_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=100,
        patience=20,
    )
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)

    # Evaluate on test set
    test_loss, test_predictions, test_targets = evaluate_model(
        model, test_data_loader, criterion
    )
    test_predictions_inverted = train_dataset.scaler_y.inverse_transform(test_predictions)
    test_targets_inverted = train_dataset.scaler_y.inverse_transform(test_targets)
    test_metrics = calculate_metrics(
        test_targets_inverted, test_predictions_inverted, OUTPUT_COLS
    )
    print("=" * 30)
    print("Test Results:")
    print(f"Test Loss (normalized): {test_loss:.4f}")
    print(f"Test R² Score: {test_metrics['Overall']['R2']:.4f}")
    print("=" * 30)

    # Export the model

    export_model_to_onnx(model, target_path)


if __name__ == "__main__":
    func_tags = ["hyperfaas-bfs-json:latest", "hyperfaas-thumbnailer-json:latest", "hyperfaas-echo:latest"]
    short_names = ["bfs", "thumbnailer", "echo"]
    db_name = "metrics.db"
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = curr_dir + "/../../benchmarks/" + db_name
    table_name = "training_data"

    for func_tag, short_name in zip(func_tags, short_names):
        target_path = curr_dir + "/" + short_name + ".onnx"
        main(table_name, func_tag, target_path, db_path=db_path)
