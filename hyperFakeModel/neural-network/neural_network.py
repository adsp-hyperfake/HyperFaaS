from datetime import datetime
from functools import partial
import copy
import json
import os
import sqlite3
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from pandas.errors import EmptyDataError
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna

BEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
COMPILATION_MODE = None
DATALOADER_WORKERS = 0
FLOAT32_PRECISION = "highest" # "highest", "high", "medium"
torch.set_float32_matmul_precision(FLOAT32_PRECISION)
torch.backends.cudnn.benchmark = True
PRELOAD_ALL_DATA_TO_DEVICE = True

torch.manual_seed(42)
np.random.seed(42)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

class CustomDataset(Dataset):
    def __init__(self, X, y, scaler_X=None, scaler_y=None, fit_scalers=True, cpu: bool = False):
        if scaler_X is None:
            self.scaler_X = StandardScaler()
        else:
            self.scaler_X = scaler_X

        if scaler_y is None:
            self.scaler_y = StandardScaler()
        else:
            self.scaler_y = scaler_y

        device = BEST_DEVICE
        if cpu:
            device = CPU_DEVICE

        # fit scalers for training data only!
        if fit_scalers:
            self.X = self.scaler_X.fit(X)
            self.y = self.scaler_y.fit(y)

        # convert to tensors
        if PRELOAD_ALL_DATA_TO_DEVICE:
            self.X = torch.tensor(X, dtype=torch.float32, device=device)
            self.y = torch.tensor(y, dtype=torch.float32, device=device)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(
        self,
        input_dim=5,
        output_dim=3,
        hidden_dims=[32, 16, 8],
        dropouts=[0.3, 0.23, 0.15],
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize scaling parameters as buffers (will be set later)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        self.register_buffer("output_mean", torch.zeros(output_dim))
        self.register_buffer("output_scale", torch.ones(output_dim))

        # Build the main network
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

    def forward(self, x):
        """Forward pass through the network with integrated scaling."""
        # Input scaling = (x - mean) / scale
        x_scaled = (x - self.input_mean) / self.input_scale

        # Forward pass through the network
        output_scaled = self.model(x_scaled)
        # Output inverse scaling = output * scale + mean
        output = output_scaled * self.output_scale + self.output_mean

        positive_output = torch.nn.functional.softplus(output)

        return positive_output

    def set_scalers(self, input_scaler, output_scaler):
        """Set the scalers after initialization"""
        if input_scaler is not None:
            input_mean_tensor = torch.tensor(input_scaler.mean_, dtype=torch.float32)
            input_scale_tensor = torch.tensor(input_scaler.scale_, dtype=torch.float32)
            self.input_mean.copy_(input_mean_tensor)
            self.input_scale.copy_(input_scale_tensor)

        if output_scaler is not None:
            output_mean_tensor = torch.tensor(output_scaler.mean_, dtype=torch.float32)
            output_scale_tensor = torch.tensor(
                output_scaler.scale_, dtype=torch.float32
            )
            self.output_mean.copy_(output_mean_tensor)
            self.output_scale.copy_(output_scale_tensor)


def create_sample_data(input_cols, output_cols, n_samples=10000):
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
        "function_ram_usage": np.random.randint(1, 10, n_samples),
    }
    df = pd.DataFrame(data)
    X = df[input_cols].values
    y = df[output_cols].values
    return X, y

def export_model_to_onnx(cpu, model, path):
    # Save the model
    model.eval()
    dummy_input = torch.zeros(1, model.input_dim)
    if cpu:
        device = CPU_DEVICE
    else:
        device = BEST_DEVICE
    print("cpu", cpu, "device", device)
    torch.onnx.export(
        model.to(device),
        dummy_input.to(device),
        path,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,  # Optimize the model
    )
    print(f"Exported model to {path}.")


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
    # plt.savefig('plot.png')
    plt.show(block=False)

    plt.pause(0.01)


def load_data_from_dbs(dbs_dir, table_name, input_cols, output_cols):
    """Load data from all SQLite databases in a directory returns a merged dataframe."""
    all_dfs = []
    query = f"SELECT {', '.join(input_cols + output_cols + ['function_image_tag'])} FROM {table_name}"
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
    # clean rows containing zero values
    safe_columns = ["active_function_calls_count", "function_instances_count"]
    columns_to_clean = [
        column for column in input_cols + output_cols if column not in safe_columns
    ]
    mask = (combined_df[columns_to_clean] == 0).any(axis=1)
    cleaned_df = combined_df[~mask]
    print(
        f"Loaded {len(cleaned_df)} rows from {len(all_dfs)} database{'s' if len(all_dfs) > 1 else ''} in directory '{dbs_dir}'\n"
        f"Cleaned a total of {len(combined_df) - len(cleaned_df)} rows containing zero values in any of the following columns:\n\t"
        f"{', '.join(columns_to_clean)}"
    )
    return cleaned_df


def get_targets_and_features_from_tag(df, image_tag, samples, input_cols, output_cols):
    """Takes a dataframe and returns only the rows for the relevant image tag."""
    image_tag_data_only = df[df["function_image_tag"] == image_tag]
    if samples > 0:
        image_tag_data_only = image_tag_data_only.sample(n=samples, random_state=42)
    X = image_tag_data_only[input_cols].values
    y = image_tag_data_only[output_cols].values
    print(f"Loaded {len(X)} data points for image {image_tag}.")
    return X, y


def split_data(X, y, test_size=0.35, val_size=0.5, seed=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataloaders(
    X_train, y_train, X_val, y_val, batch_size, X_test=None, y_test=None, cpu = False
):
    train_dataset = CustomDataset(X_train, y_train, fit_scalers=True, cpu=cpu)
    val_dataset = CustomDataset(
        X_val,
        y_val,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        fit_scalers=False,
        cpu=cpu
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=DATALOADER_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=DATALOADER_WORKERS)

    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = CustomDataset(
            X_test,
            y_test,
            scaler_X=train_dataset.scaler_X,
            scaler_y=train_dataset.scaler_y,
            fit_scalers=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=DATALOADER_WORKERS)
    else:
        test_dataset = None

    return train_loader, val_loader, test_loader, train_dataset


def train_model(
    cpu: bool,
    model,
    train_data_loader,
    val_data_loader,
    criterion,
    optimizer,
    scheduler,
    gradient_clipping,
    num_epochs=100,
    patience=25,
):
    """Train the model with early stopping."""

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state_dict = {}

    print("Starting training...")
    print(
        f"Training for up to {num_epochs} epochs with early stopping (patience={patience})"
    )
    print("-" * 60)

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", ncols=100):
        train_loss = train_epoch(cpu, model, train_data_loader, criterion, optimizer, gradient_clipping)

        # Evaluate on validation set
        val_loss, _, _ = evaluate_model(cpu, model, val_data_loader, criterion)
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
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stoppage after {epoch + 1} epochs")
            break

    # Load best model
    model.load_state_dict(best_model_state_dict)

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return train_losses, val_losses


def train_epoch(cpu, model, train_data_loader, criterion, optimizer, gradient_clipping):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for data, targets in train_data_loader:
        if cpu:
            device = CPU_DEVICE
        else:
            device = BEST_DEVICE
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        # forward pass
        predictions = model(data)
        prodictions_normalized = (predictions - model.output_mean) / model.output_scale
        targets_normalized = (targets - model.output_mean) / model.output_scale
        loss = criterion(prodictions_normalized, targets_normalized)
        # backward pass
        loss.backward()
        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_data_loader)


def evaluate_model(cpu, model, data_loader, criterion):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in data_loader:
            if cpu:
                device = CPU_DEVICE
            else:
                device = BEST_DEVICE
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            predictions = model(data)
            predictions_normalized = (
                predictions - model.output_mean
            ) / model.output_scale
            targets_normalized = (targets - model.output_mean) / model.output_scale
            # Loss calcs
            loss = criterion(predictions_normalized, targets_normalized)
            total_loss += loss.item()
            num_batches += 1
            # .cpu() copies from gpu to cpu mem if needed
            all_predictions.append(predictions_normalized.detach())
            all_targets.append(targets_normalized.detach())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    all_predictions = all_predictions.cpu().numpy()
    all_targets = all_targets.cpu().numpy()

    return total_loss / num_batches, all_predictions, all_targets


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


def predict(model, input_data, cpu):
    """Make predictions using the trained model."""
    # ensure input_data is a 2D array
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)

    # Convert input data to tensor
    device = BEST_DEVICE
    if cpu:
        device = CPU_DEVICE
    X_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    return predictions


def initialize_model(
    cpu: bool,
    input_dim,
    output_dim,
    hidden_dims,
    dropouts,
    lr,
    weight_decay,
    optimizer_name,
    scheduler_patience,
):
    if cpu:
        device = CPU_DEVICE
    else:
        device = BEST_DEVICE
    print(f"Initializing model on {device}")
    model = MLP(input_dim, output_dim, hidden_dims, dropouts).to(device)
    if COMPILATION_MODE is not None:
        print(f"Compiling model with {COMPILATION_MODE}")
        model = torch.compile(model, mode=COMPILATION_MODE)
    criterion = nn.MSELoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=scheduler_patience
    )
    return model, criterion, optimizer, scheduler


def setup_model_training(
    cpu: bool,
    identifier,
    onnx_export_path,
    X,
    y,
    epochs,
    output_cols,
    hyperparams=None,
):
    """Trains the model."""
    torch.manual_seed(42)
    np.random.seed(42)

    if cpu:
        device = CPU_DEVICE
    else:
        device = BEST_DEVICE

    print(f"Starting training for {identifier} using device {device}.")

    # train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    if hyperparams is None:
        hyperparams = {
            "hidden_dims": [95, 47, 9],
            "dropouts": [0.34, 0.018, 0.116],
            "lr": 0.001,
            "weight_decay": 1.05e-05,
            "batch_size": 64,
            "patience": 10,
            "optimizer": "Adam",
            "gradient_clipping": True,
        }

    print(hyperparams)

    train_loader, val_loader, test_loader, train_dataset = prepare_dataloaders(
        X_train, y_train, X_val, y_val, hyperparams["batch_size"], X_test, y_test, cpu=cpu
    )

    # Model configuration
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Create model, loss function, and optimizer
    model, criterion, optimizer, scheduler = initialize_model(
        cpu,
        input_dim,
        output_dim,
        hyperparams["hidden_dims"],
        hyperparams["dropouts"],
        hyperparams["lr"],
        hyperparams["weight_decay"],
        hyperparams["optimizer"],
        scheduler_patience=10,
    )
    print(hyperparams)
    gradient_clipping = hyperparams["gradient_clipping"]

    model.set_scalers(train_dataset.scaler_X, train_dataset.scaler_y)

    # Train the model
    train_losses, val_losses = train_model(
        cpu,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        gradient_clipping,
        num_epochs=epochs,
        patience=hyperparams["patience"],
    )

    # Evaluate on test set
    test_loss, test_predictions, test_targets = evaluate_model(
        cpu, model, test_loader, criterion
    )
    test_metrics = calculate_metrics(test_targets, test_predictions, output_cols)

    # Print test results
    print("=" * 30)
    print("Test Results:")
    print(f"Test Loss (normalized): {test_loss:.4f}")
    print(f"Test R² Score: {test_metrics['Overall']['R2']:.4f}")
    print("=" * 30)

    # Export the model
    export_model_to_onnx(cpu, model, onnx_export_path)

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)
    return min(val_losses)


def optuna_objective(trial, X, y, epochs, cpu: bool):
    """
    Optuna objective function for hyperparameter optimization.
    This function defines hyperparameters to be optimized and trains the model once and returns the validation loss.
    It is supposed to be used with Optuna to find very good hyperparameters for the model.
    """

    # Define hyperparameters and their search space
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = []
    dropouts = []
    for i in range(n_layers):
        hidden_dim = trial.suggest_int(f"hidden_size_l{i}", 4, 40)
        dropout = trial.suggest_float(f"dropout_l{i}", 0.01, 0.25)
        hidden_dims.append(hidden_dim)
        dropouts.append(dropout)

    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    # SGD and RMSprop didn't perform well
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    gradient_clipping = trial.suggest_categorical("gradient_clipping", [True, False])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    patience = trial.suggest_int("patience", 20, 20)

    # Split data and prepare dataloaders
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)
    train_loader, val_loader, _, train_dataset = prepare_dataloaders(
        X_train, y_train, X_val, y_val, batch_size, cpu
    )

    # Define model, loss function, and optimizer
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    model, criterion, optimizer, scheduler = initialize_model(
        cpu,
        input_dim,
        output_dim,
        hidden_dims,
        dropouts,
        lr,
        weight_decay,
        optimizer_name,
        scheduler_patience=5,
    )

    model.set_scalers(train_dataset.scaler_X, train_dataset.scaler_y)

    # Train the model
    train_model(
        cpu,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        gradient_clipping,
        num_epochs=epochs,
        patience=patience,
    )
    val_loss, _, _ = evaluate_model(cpu, model, val_loader, criterion)

    return val_loss


def run_optuna_study(
    cpu: bool, identifier, export_dir, trials, jobs, epochs, final_epochs, X, y, output_cols, state_db: None
):
    """Main function to run the training pipeline with Optuna hyperparameter optimization."""
    # Wrap the objective function with partial to pass additional arguments
    wrapped_objective = partial(optuna_objective, X=X, y=y, epochs=epochs, cpu=cpu)

    # Create an Optuna study and optimize
    if state_db:
        study = optuna.create_study(
            direction="minimize", study_name=identifier, storage=state_db
        )
    else:
        study = optuna.create_study(direction="minimize", study_name=identifier)
    study.optimize(wrapped_objective, n_trials=trials, n_jobs=jobs)

    print("Study over, the following optimized parameters were established:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # Send notification via ntfy
    requests.post(
        "https://ntfy.sh/hyperfake",
        data="\n".join(f"{k}: {v}" for k, v in study.best_params.items()).encode(
            "utf-8"
        ),
    )

    # Save the best hyperparameters
    n_layers = study.best_params["n_layers"]
    hyperparams = {
        "hidden_dims": [
            study.best_params[f"hidden_size_l{i}"] for i in range(n_layers)
        ],
        "dropouts": [study.best_params[f"dropout_l{i}"] for i in range(n_layers)],
        "lr": study.best_params["lr"],
        "weight_decay": study.best_params["weight_decay"],
        "batch_size": study.best_params["batch_size"],
        "patience": study.best_params["patience"],
        "optimizer": study.best_params["optimizer"],
        "gradient_clipping": study.best_params["gradient_clipping"],
    }

    onnx_export_path = os.path.join(export_dir, f"{identifier}.onnx")
    # Train the model with the best hyperparameters
    val_score_final_training = setup_model_training(
        cpu, identifier, onnx_export_path, X, y, final_epochs, output_cols, hyperparams
    )

    # Write the best hyperparameters to a file
    hyperparams_target = os.path.join(export_dir, f"{identifier}_hyperparams.json")
    export_dict = {
        "hyperparams": hyperparams,
        "val_score_optuna": study.best_value,
        "val_score_final_training": val_score_final_training,
    }
    with open(hyperparams_target, "w") as f:
        json.dump(export_dict, f, indent=4)


def manual_pipeline(
    cpu: bool,
    func_tags,
    short_names,
    dbs_dir,
    table_name,
    sample_data,
    export_dir,
    epochs,
    hyperparams,
    samples,
    input_cols,
    output_cols,
):
    """Main function to run the manual training pipeline."""
    if not sample_data:
        df = load_data_from_dbs(dbs_dir, table_name, input_cols, output_cols)
    for func_tag, short_name in zip(func_tags, short_names):
        if not sample_data:
            X, y = get_targets_and_features_from_tag(df, func_tag, samples, input_cols, output_cols)
            if X.size == 0:
                print(
                    f"\033[31mSkipping {func_tag}, no training data in the loaded dbs.\033[0m"
                )  # red text
                continue
        else:
            X, y = create_sample_data(input_cols=input_cols, output_cols=output_cols)
        identifier = short_name + "_" + datetime.now().strftime("%m%d_%H%M")
        onnx_export_path = os.path.join(export_dir, f"{short_name}.onnx")
        setup_model_training(cpu, identifier, onnx_export_path, X, y, epochs, output_cols, hyperparams)


def optuna_pipeline(
    cpu: bool,
    func_tags,
    short_names,
    dbs_dir,
    table_name,
    sample_data,
    export_dir,
    trials,
    jobs,
    epochs,
    final_epochs,
    samples,
    save_state,
    state_db,
    study_id,
    input_cols,
    output_cols,
):
    """Main function to run the Optuna training pipeline."""
    for func_tag, short_name in zip(func_tags, short_names):
        if not sample_data:
            df = load_data_from_dbs(dbs_dir, table_name, input_cols, output_cols)
            X, y = get_targets_and_features_from_tag(df, func_tag, samples, input_cols, output_cols)
            del df  # force garbage collector
            if X.size == 0:
                print(
                    f"\033[31mSkipping {func_tag}, no training data in the loaded dbs.\033[0m"
                )  # red text
                continue
        else:
            X, y = create_sample_data(input_cols=input_cols, output_cols=output_cols)
        if not study_id:
            study_id = datetime.now().strftime("%m%d_%H%M")
        identifier = short_name + "_" + study_id
        if save_state and not state_db:
            state_db_path = os.path.join(CURR_DIR, "models", f"{identifier}.db")
            state_db = f"sqlite:///{state_db_path}"
        run_optuna_study(
            cpu, identifier, export_dir, trials, jobs, epochs, final_epochs, X, y, output_cols, state_db
        )
