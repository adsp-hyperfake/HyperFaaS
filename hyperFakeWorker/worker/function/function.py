from pathlib import Path
import random
from hashlib import sha256
import threading
import time

from .abstract import AbstractFunction
from ..managers.function import FunctionManager
from ..managers.status import StatusManager
from .names import adjectives, names
from ..api.controller.controller_pb2 import StatusUpdate, Event, Status
from ..api.common.common_pb2 import InstanceID, FunctionID
from ..log import logger
from ..utils.time import get_timestamp, NANOSECONDS
from ..models.input import FunctionModelInput
from ..models.function import FunctionModelInferer
from .image import FunctionImage

class Function(AbstractFunction):

    def __init__(self, function_manager: FunctionManager, status_manager: StatusManager, name: str, function_id: str, instance_id: str, image: FunctionImage, model: FunctionModelInferer):
        self.function_manager: FunctionManager = function_manager
        self.status_manager: StatusManager = status_manager

        self.created_at = time.time_ns()
        self.last_worked_at = 0

        self.work_lock: threading.RLock = threading.RLock()

        self.name = name
        self.function_id = function_id
        self.instance_id = instance_id
        self.image = image

        self.model = model

        self.cpu = 0.0
        self.ram = 0

        self.is_cold = True

    @property
    def was_recently_active(self):
        return self.time_since_last_work <= 8 * NANOSECONDS # or self.is_active

    @property
    def is_active(self):
        # return self.was_recently_active
        if self.work_lock.acquire(False):
            self.work_lock.release()
            return False
        return True

    @property
    def uptime(self):
        current_time = time.time_ns()
        return current_time - self.created_at
    
    @property
    def time_since_last_work(self):
        current_time = time.time_ns()
        return current_time - self.last_worked_at
    
    def coldstart(self):
        logger.debug(f"Waiting for coldstart of function {self.function_id} - {self.instance_id}")
        # time.sleep(1)
        self.is_cold = False
        self.status_manager.send_status_update(update=StatusUpdate(
            instance_id=InstanceID(id=self.instance_id),
            event=Event.Value("EVENT_RUNNING"),
            status=Status.Value("STATUS_SUCCESS"),
            function_id=FunctionID(id=self.function_id),
            timestamp=get_timestamp()
        ))
        logger.debug(f"Finished coldstart of function {self.function_id} - {self.instance_id}")

    def timeout(self):
        self.status_manager.send_status_update(update=StatusUpdate(
            instance_id=InstanceID(id=self.instance_id),
            event=Event.Value("EVENT_TIMEOUT"),
            function_id=FunctionID(id=self.function_id),
        ))
        return None
    
    def lock(self):
        return self.work_lock.acquire(blocking=False)

    def unlock(self):
        self.work_lock.release()

    def work(self, body_size: int, bytes: int):
        with self.work_lock:
            self.last_worked_at = time.time_ns()
            logger.debug(f"Executing function {self.function_id} - {self.instance_id}")
            if self.is_cold:
                self.coldstart()
            results = self.model.infer(
                FunctionModelInput(body_size, 
                                   self.function_manager.get_num_function_instances(self.function_id), 
                                   self.function_manager.get_num_active_functions(self.function_id), 
                                   self.function_manager.total_cpu_usage, 
                                   self.function_manager.total_ram_usage
                                   )
            )
            self.cpu = results.cpu_usage
            self.ram = results.ram_usage
            self.last_worked_at = time.time_ns() + results.function_runtime # Write estimated time
            time.sleep(results.function_runtime / 1_000_000_000)
            timeout = False
            self.last_worked_at = time.time_ns() # Set correct time in case of simulated errors
            if timeout:
                self.timeout()
                return None, results.function_runtime
            self.cpu = 0
            self.ram = 0
            return random.randbytes(bytes), results.function_runtime
           
    def __eq__(self, value):
        if not isinstance(value, Function):
            return False
        return self.function_id == value.function_id and self.instance_id == value.instance_id
    
    def __hash__(self):
        return self.instance_id.__hash__()
    
    @staticmethod
    def create_new(function_manager: FunctionManager, status_manager: StatusManager, function_id: str, image: FunctionImage, model: Path) -> "Function":
        if model is None:
            raise ValueError(f"model cannot be None!")
        hash_source = function_id + str(random.randint(1, 2**31))
        return Function(
            function_manager=function_manager,
            status_manager=status_manager,
            name=f"{random.choice(adjectives)}-{random.choice(names)}",
            function_id=function_id,
            instance_id=sha256(hash_source.encode(errors="ignore")).hexdigest()[0:12],
            image=image,
            model=FunctionModelInferer(model)
        )