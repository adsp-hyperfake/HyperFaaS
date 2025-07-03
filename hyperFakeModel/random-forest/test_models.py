#!/usr/bin/env python3
"""
Manual testing script for Random Forest ONNX models
"""
import onnxruntime as ort
import numpy as np
from pathlib import Path

def test_model(model_path: str, model_name: str):
    """Test a single ONNX model with various inputs"""
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    model = ort.InferenceSession(model_path)
    input_name = model.get_inputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Input shape: {model.get_inputs()[0].shape}")
    print(f"Output shape: {model.get_outputs()[0].shape}")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Small Request",
            "input": [512, 1, 1, 0.1, 128],  # small body, 1 instance, 1 call, low CPU/RAM
        },
        {
            "name": "Medium Request", 
            "input": [2048, 3, 5, 0.4, 512],  # medium body, 3 instances, 5 calls, medium CPU/RAM
        },
        {
            "name": "Large Request",
            "input": [8192, 5, 10, 0.8, 1024],  # large body, 5 instances, 10 calls, high CPU/RAM
        },
        {
            "name": "High Load",
            "input": [4096, 10, 20, 0.9, 2048],  # high load scenario
        },
        {
            "name": "Zero Load",
            "input": [100, 0, 0, 0.0, 0],  # minimal load
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        test_input = np.array([scenario['input']], dtype=np.float32)
        
        # Run inference
        outputs = model.run(None, {input_name: test_input})
        predictions = outputs[0][0]  # Get first (and only) prediction
        
        print(f"Input:")
        print(f"  Body size: {test_input[0][0]:>8.0f} bytes")
        print(f"  Function instances: {test_input[0][1]:>3.0f}")
        print(f"  Active calls: {test_input[0][2]:>6.0f}")
        print(f"  CPU usage: {test_input[0][3]*100:>8.1f}%")
        print(f"  RAM usage: {test_input[0][4]:>8.0f} MB")
        
        print(f"Predictions:")
        print(f"  Function runtime: {predictions[0]:>8.3f} seconds")
        print(f"  CPU usage: {predictions[1]:>12.3f}")
        print(f"  RAM usage: {predictions[2]:>12.1f} MB")

def main():
    """Test all models"""
    models_dir = Path("./")
    
    models = [
        ("bfs.onnx", "BFS Function"),
        ("thumbnailer.onnx", "Thumbnailer Function"), 
        ("echo.onnx", "Echo Function")
    ]
    
    print("Random Forest Model Predictions Test")
    print("Features: [body_size, function_instances, active_calls, cpu_usage, ram_usage]")
    print("Outputs: [function_runtime, cpu_usage, ram_usage]")
    
    for model_file, model_name in models:
        model_path = models_dir / model_file
        if model_path.exists():
            test_model(str(model_path), model_name)
        else:
            print(f"\n❌ Model not found: {model_path}")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 