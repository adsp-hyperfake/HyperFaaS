import onnx
import onnxruntime as ort

def load_model(model_path: str) -> onnx.ModelProto:
    """
    Load an ONNX model from the specified path.
    Args:
        model_path (str): The path to the ONNX model file.
    Returns:
        onnx.ModelProto: The loaded ONNX model.
    """
    try:
        model = onnx.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model from {model_path}: {e}")
    
def infer_model(model: onnx.ModelProto, input_data) -> dict:
    """
    Perform inference using the loaded ONNX model.
    Args:
        model (onnx.ModelProto): The loaded ONNX model.
        input_data (dict): A dictionary containing input data for the model.
    Returns:
        dict: The output of the model inference.
    """
    try:
        session = ort.InferenceSession(model.SerializeToString())
        outputs = session.run(None, input_data)
        output_names = [output.name for output in session.get_outputs()]
        return {name: output for name, output in zip(output_names, outputs)}
    except Exception as e:
        raise RuntimeError(f"Failed to perform inference: {e}")

    
# Example usage with a simple ONNX model "add_model.onnx" that takes two inputs "a" and "b" and returns their sum.
if __name__ == "__main__":
    import numpy as np

    model_path = "model/add_model.onnx"
        
    model = load_model(model_path)
    
    input_data = {
        "a": np.array([[1.0]], dtype=np.float32), 
        "b": np.array([[2.0]], dtype=np.float32)
        }
    
    output = infer_model(model, input_data)
    
    print(f"Model output: {output}")
