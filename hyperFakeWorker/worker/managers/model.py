from pathlib import Path

from ..function.image import FunctionImage
from ..function import FunctionIdStr

from ..log import logger

class ModelManager():

    def __init__(self, models: list[Path]):

        self.function_model_paths = models
        self.function_models: dict[FunctionIdStr, Path] = {}

    def find_model(self, function_id: FunctionIdStr, image: FunctionImage) -> Path:
        # resolve model path
        model_path = self.function_models.get(function_id)
        if model_path is None:
            for p in self.function_model_paths:
                if p.stem == image.image:
                    self.function_models[function_id] = p
                    return p
            logger.error(f"Failed to find a model corresponding to {function_id} | {image}")
            return None
        else:
            return model_path
        