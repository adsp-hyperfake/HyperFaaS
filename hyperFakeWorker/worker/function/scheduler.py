from . import FunctionIdStr
from ..managers.function import FunctionManager
from .function import Function

from ..log import logger

class FunctionUnknownToSchedulerException(KeyError):
    pass

class FunctionScheduler():

    def __init__(self, function_manager: FunctionManager):
        self._function_manager = function_manager

    def get_instance_for_call(self, function_id: FunctionIdStr) -> Function:
        """Get the next instance for the given function_id using round-robin scheduling."""
        
        # Check if function_id exists
        if function_id not in self._function_manager.functionId_to_instances_map:
            raise FunctionUnknownToSchedulerException(f"The requested function {function_id} is unknown!")
        
        # Get instances for this function
        instances = self._function_manager.functionId_to_instances_map.get(function_id)
        if instances is None or len(instances) == 0:
            raise FunctionUnknownToSchedulerException(f"No instances available for function {function_id}!")
        
        # Get the next instance using round-robin
        selected_instance = self._function_manager.get_next_instance_round_robin(function_id)
        
        if selected_instance is None:
            raise FunctionUnknownToSchedulerException(f"Failed to get an instance for function {function_id}!")
        
        logger.debug(f"Scheduled instance {selected_instance.instance_id} for function {function_id}")
        return selected_instance