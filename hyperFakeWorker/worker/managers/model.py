from pathlib import Path

from ..function.image import FunctionImage
from ..function import FunctionIdStr

from ..log import logger

class ModelManager():

    def __init__(self, models: dict[str, Path]):
        
        self.image_to_model_path = models

        self.function_models: dict[FunctionIdStr, Path] = {}

    def find_model(self, function_id: FunctionIdStr, image: FunctionImage) -> Path:
        model_path = self.function_models.get(function_id)
        if model_path is None:
            model_path = self.image_to_model_path.get(image.image)
            if model_path is not None:
                self.function_models[function_id] = model_path
                return model_path
            else:
                logger.error(f"Failed to find a model corresponding to {function_id} | {image.image}. Available: {list(self.image_to_model_path.keys())}")
                return None
        else:
            return model_path
        