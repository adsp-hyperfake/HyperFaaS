from dataclasses import dataclass
import numpy as np

@dataclass
class FunctionModelInput():
    body_size: int
    function_instances: int
    function_calls: int
    cpu_usage: float
    ram_usage: int

    def as_tensor(self) -> np.ndarray:
        return np.array(
            [[
                self.body_size,
                self.function_instances,
                self.function_calls,
                self.cpu_usage,
                self.ram_usage
            ]],
            dtype=np.float32
        )