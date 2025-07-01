from dataclasses import dataclass
import numpy as np

@dataclass
class FunctionModelOutput():
    function_runtime: int
    cpu_usage: float
    ram_usage: int

    @staticmethod
    def from_tensor(tensor: np.ndarray) -> "FunctionModelOutput":
        values = tensor.flatten()

        if values.size != 3:
            raise ValueError(f"Expected 3 otuput values, got {values.size}: {values}")

        return FunctionModelOutput(
            function_runtime=int(values[0]),
            cpu_usage=float(values[1]),
            ram_usage=int(values[2])
        )