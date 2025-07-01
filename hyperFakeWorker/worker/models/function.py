from ..utils.onnx import ONNXModelInferer

from .input import FunctionModelInput
from .output import FunctionModelOutput

class FunctionModelInferer(ONNXModelInferer):

    def infer(self, input_data: FunctionModelInput) -> FunctionModelOutput:
        """
        Perform inference using the loaded ONNX model.
        Args:
            input_data (FunctionModelInput): A FunctionModelInput object
        Returns:
            FunctionModelOutput: The output of the model inference.
        """
        tensor_input = input_data.as_tensor()

        input_names = self.model.get_inputs()

        if len(input_names) != 1:
            raise ValueError(f"Expected exactly one input in the model, got {len(input_names)}.")

        input_name = input_names[0].name

        outputs = self.model.run(None, {input_name: tensor_input})

        output_tensor = outputs[0]

        return FunctionModelOutput.from_tensor(output_tensor)