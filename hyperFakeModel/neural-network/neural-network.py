import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx as onnx
import pandas as pd

class ConstantOutputModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

        with torch.no_grad():
            self.linear.weight.fill_(0.0)
            self.linear.bias.fill_(1.0)

    def forward(self, x):
        return self.linear(x)

def create_test_database(path='testdatabase.db'):
    # Connect to SQLite database
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            request_body_size INTEGER,
            function_instances_count INTEGER,
            active_function_calls_count INTEGER,
            worker_cpu_usage REAL,
            worker_ram_usage INTEGER,
            function_runtime INTEGER,
            function_cpu_usage REAL,
            function_ram_usage INTEGER
        )
    ''')

    # Drop existing data
    cursor.execute('DELETE FROM training_data')

    # Insert mock data
    mock_data = [
        (1024, 3, 5, 1.5, 4_000_000_000, 5_000_000, 0.25, 512_000_000),
        (2048, 2, 4, 2.0, 8_000_000_000, 8_500_000, 0.5, 1_024_000_000),
        (4096, 1, 2, 0.8, 2_000_000_000, 2_000_000, 0.15, 256_000_000)
    ]
    cursor.executemany('INSERT INTO training_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)', mock_data)

    # Commit changes and close connection
    conn.commit()
    conn.close()

def prepare_data(db_path='../../benchmarks/metrics.db'):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    # Define relevant columns
    input_cols = ['request_body_size', 'function_instances_count', 'active_function_calls_count', 'worker_cpu_usage', 'worker_ram_usage']
    output_cols = ['function_runtime', 'function_cpu_usage', 'function_ram_usage']
    all_cols = input_cols + output_cols

    # Fetch columns from database
    query = f"SELECT {', '.join(all_cols)} FROM training_data"
    dataframe = pd.read_sql_query(query, conn)
    conn.close()

    # Set types for input and output columns
    for col in input_cols + output_cols:
        if 'cpu_usage' in col or 'worker_cpu_usage' in col:
            dataframe[col] = dataframe[col].astype('float32')
        else:
            dataframe[col] = dataframe[col].astype('int64')

    # Convert to tensors (deep learning models typically use float32)
    inputs = torch.tensor(dataframe[input_cols].values, dtype=torch.float32)
    targets = torch.tensor(dataframe[output_cols].values, dtype=torch.float32)
    
    return inputs, targets

def train(model, inputs, targets, epochs=1):
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    return model

def evaluate(model, inputs, targets):
    # Switch model to evaluation mode
    model.eval()

    # Evaluate model
    with torch.no_grad():
        predictions = model(inputs)

        # Calculate Mean Squared Error and Mean Absolute Error
        mse = nn.MSELoss()(predictions, targets).item()
        mae = nn.L1Loss()(predictions, targets).item()
    
    print(f"Evaluation Results:\nMSE: {mse:.4f}\nMAE: {mae:.4f}")

def export_model(model, path='model.onnx'):
    # Save the model
    model.eval()
    dummy_input = torch.zeros(1, 5)
    onnx.export(
        model,
        dummy_input,
        path,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True, # Optimize the model
    )

if __name__ == "__main__":
    create_test_database(path='testdatabase.db')

    inputs, targets = prepare_data(db_path='testdatabase.db')

    model = ConstantOutputModel(input_size=5, output_size=3)

    print("Training...")
    # train(model, inputs, targets, epochs=1)

    print("\nEvaluating...")
    evaluate(model, inputs, targets)

    path = 'model.onnx'
    print(f"\nExporting model to \'{path}\'...")
    export_model(model, path=path)
    
    print("\nDone!")