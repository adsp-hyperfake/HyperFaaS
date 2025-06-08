# The purpose of this script is to test if the model that was exported by neural-network.py can be loaded and used for inference.

import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("model.onnx")

# Get input and output node information
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_size = session.get_inputs()[0].shape[1]

def infer_from_input():
    while True:
        try:
            raw = input(f"Enter {input_size} values separated by spaces (or type 'exit' to quit):\n> ")
            if raw.strip().lower() == "exit":
                break

            values = list(map(float, raw.strip().split()))
            if len(values) != input_size:
                print(f"You must enter exactly {input_size} values.")
                continue

            input_array = np.array([values], dtype=np.float32)
            output = session.run([output_name], {input_name: input_array})
            print("Output:", output[0][0])
        except Exception as e:
            print("Error during input or inference:", e)

if __name__ == "__main__":
    infer_from_input()