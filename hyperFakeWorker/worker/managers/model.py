from pathlib import Path
import threading

from ..function.image import FunctionImage
from ..function import FunctionIdStr
from ..models.function import FunctionModelInferer

from ..log import logger

class ModelManager():

    def __init__(self, models: dict[str, Path]):
        
        self.image_to_model_path = models

        self.function_models: dict[FunctionIdStr, Path] = {}
        
        # Model cache to share loaded models between instances
        self.model_cache: dict[Path, FunctionModelInferer] = {}
        self.cache_lock = threading.RLock()

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
    
    def get_cached_model(self, model_path: Path) -> FunctionModelInferer:
        """Get a cached FunctionModelInferer instance. All instances share the same loaded model."""
        if model_path is None:
            raise ValueError("model_path cannot be None!")
            
        with self.cache_lock:
            if model_path in self.model_cache:
                logger.debug(f"Using cached model for {model_path}")
                return self.model_cache[model_path]
            
            logger.info(f"Loading and caching model from {model_path}")
            model_inferer = FunctionModelInferer(model_path)
            self.model_cache[model_path] = model_inferer
            
            return model_inferer
        