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
        scheduled_instance: Function = None

        running_instances: set[Function] = self._function_manager.functionId_to_instances_map[function_id]
        if len(running_instances) < 1:
            logger.critical(f"Insufficient instances for function {function_id} are scheduled!")
        for i in range(tries + 1):
            if self._function_manager.get_num_inactive_functions(function_id) < 1:
                sleep(0)
            for instance in running_instances:
                instance: Function
                if scheduled_instance is not None:
                    logger.debug("Instance already found, breaking the loop...")
                    break
                lock = instance.lock()
                if lock:
                    scheduled_instance = instance
                    break
            if scheduled_instance is not None:
                break
            sleep(2**(i - 5))
        if scheduled_instance is None and self._function_manager.get_num_inactive_functions(function_id) > 1:
            raise RuntimeError(f"I did not aquire a lock, even though there are {self._function_manager.get_num_inactive_functions(function_id)} inactive functions!!!")
        yield scheduled_instance
        if scheduled_instance is not None:
            scheduled_instance.unlock()