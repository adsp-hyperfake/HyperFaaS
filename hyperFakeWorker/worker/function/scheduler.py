from contextlib import contextmanager

from . import FunctionIdStr
from ..managers.function import FunctionManager
from .function import Function
from time import sleep

from ..log import logger

class FunctionScheduler():

    def __init__(self, function_manager: FunctionManager):
        self._function_manager = function_manager

    @contextmanager
    def schedule_call(self, function_id: FunctionIdStr, tries: int):
        running_instances = self._function_manager.functionId_to_instances_map[function_id]
        if len(running_instances) < 1:
            logger.critical(f"Insufficient instances for function {function_id} are scheduled!")
        scheduled_instance: Function = None
        for _ in range(tries):
            for instance in running_instances:
                if instance.lock() and scheduled_instance is None:
                    scheduled_instance = instance
                    break
            if scheduled_instance is not None:
                break
            sleep(2**(tries - 5))
        yield scheduled_instance
        if scheduled_instance is not None:
            scheduled_instance.unlock()