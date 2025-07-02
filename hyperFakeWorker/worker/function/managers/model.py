from pathlib import Path

from ..image import FunctionImage

from .. import FunctionIdStr

class ModelManager():

    def __init__(self, models: list[Path]):

        self.function_model_paths = models
        self.function_models: dict[FunctionIdStr, Path] = {}

    def find_model(self, function_id: FunctionIdStr, image: FunctionImage) -> Path:
        # resolve model path
        if self.function_models.get(function_id) is None:
            for p in self.function_model_paths:
                if p.stem == image.image:
                    self.function_models[function_id] = p
                    break
        return self.function_models.get(function_id)