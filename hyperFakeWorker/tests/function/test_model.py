from pathlib import Path
import numpy as np
import pytest

from hyperFakeWorker.worker.function.managers.model import FunctionModelInferer, FunctionModelInput, FunctionModelOutput

test_path = Path(__file__).parent

# Auto-discover available test models
test_models = list(test_path.glob("*.onnx"))

def test_FunctionModelInferer():
    if not test_models:
        pytest.skip("No ONNX test models found")
    
    # Use the first available test model
    inferer = FunctionModelInferer(test_models[0])
    assert inferer.model is not None, "Failed to load ONNX model"

    input_data = FunctionModelInput(100, 5, 5, 5.5, 1024)
    input_tensor = input_data.as_tensor()
    assert input_tensor.shape == (1, 5), "Input tensor has invalid shape"
    assert input_tensor.dtype == np.float32, "Input tensor has invalid dtype"

    output_data = inferer.infer(input_data)