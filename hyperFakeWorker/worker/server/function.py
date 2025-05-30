import datetime
import time
import random
from hashlib import sha256
import threading
from queue import Queue, Empty

from blinker import signal

from .._grpc.controller.controller_pb2 import StatusUpdate, Event, Status, FunctionState, InstanceState
from .._grpc.common.common_pb2 import InstanceID, FunctionID
from ..log import logger
from ..utils.time import get_timestamp

InstanceIdStr=str
FunctionIdStr=str

class Function():

    def __init__(self, function_id: str, instance_id: str):
        self.created_at = int(datetime.datetime.now().timestamp())
        self.last_worked_at = int(datetime.datetime.now().timestamp())
        self.work_lock = threading.Lock()
        self.function_id = function_id
        self.instance_id = instance_id

        self.is_cold = True

        self.status_signal = signal("status")

    @property
    def is_active(self):
        return self.work_lock.locked()

    @property
    def uptime(self):
        current_time = int(datetime.datetime.now().timestamp())
        return current_time - self.created_at
    
    @property
    def time_since_last_work(self):
        current_time = int(datetime.datetime.now().timestamp())
        return current_time - self.last_worked_at

    @staticmethod
    def create_new(function_id: str):
        hash_source = function_id + str(random.randint(1, 2^31))
        return Function(
            function_id=function_id,
            instance_id=sha256(hash_source.encode(errors="ignore")).hexdigest(),
        )
    
    def work(self, bytes: int):
        logger.debug(f"Executing function {self.function_id} - {self.instance_id}")
        if self.is_cold:
            logger.debug(f"Waiting for coldstart of function {self.function_id} - {self.instance_id}")
            time.sleep(1)
            self.is_cold = False
            self.status_signal.send(self, update=StatusUpdate(
                instance_id=InstanceID(id=self.instance_id),
                event=Event.EVENT_RUNNING,
                status=Status.STATUS_SUCCESS,
                function_id=FunctionID(id=self.function_id),
                timestamp=get_timestamp()
            ))
            logger.debug(f"Finished coldstart of function {self.function_id} - {self.instance_id}")
        with self.work_lock:
            time.sleep(2)
            timeout = False
            if timeout:
                self.status_signal.send(self, update=StatusUpdate(
                    instance_id=InstanceID(id=self.instance_id),
                    event=Event.EVENT_TIMEOUT,
                    function_id=FunctionID(id=self.function_id),
                ))
            self.last_worked_at = int(datetime.datetime.now().timestamp())
            return random.randbytes(bytes)
           
    def __eq__(self, value):
        if not isinstance(value, Function):
            return False
        return self.function_id == value.function_id and self.instance_id == value.instance_id
    
    def __hash__(self):
        return self.instance_id.__hash__()

class FunctionManager():

    def __init__(self):
        # instance_id : Function
        self.active_functions: dict[InstanceIdStr, Function] = {}
        # function_id : Function
        self.instances: dict[FunctionIdStr, set[Function]] = {}

        self.status_signal = signal("status")

    def add_function(self, function: Function):
        self.active_functions[function.instance_id] = function

        if self.instances.get(function.function_id) is None:
            self.instances[function.function_id] = set()
        self.instances[function.function_id].add(function)

    def remove_function(self, instance_id: InstanceIdStr):
        if self.active_functions.get(instance_id) is None:
            return
        function = self.active_functions[instance_id]
        self.instances[function.function_id].remove(function)
        return self.active_functions.pop(instance_id)

    def get_function(self, instance_id: InstanceIdStr):
        try:
            return self.active_functions[instance_id]
        except KeyError as e:
            logger.critical(f"Failed to find instance_id {instance_id} in:\n{self.active_functions.keys()}")
            raise e
    
    def send_status_update(self, update: StatusUpdate):
        self.status_signal.send(self, update=update)

    def get_status_updates(self):
        updates: Queue[StatusUpdate] = Queue(2**6)

        def handler(sender, update):
            updates.put(update)

        self.status_signal.connect(handler)

        try:
            while True:
                update = updates.get()
                yield update
        finally:
            self.status_signal.disconnect(handler)


    def get_state(self) -> list[FunctionState]:
        state = []
        for function_id in self.instances.keys():
            functions = self.instances[function_id]
            running, idle = [], []
            for func in functions:
                instance_state = InstanceState(
                    instance_id=func.instance_id,
                    is_active=func.is_active,
                    time_since_last_work=func.time_since_last_work,
                    uptime=func.uptime
                )
                if func.is_active:
                    running.append(
                        instance_state
                    )
                else:
                    idle.append(
                        instance_state
                    )
            state.append(FunctionState(
                function_id=function_id,
                running=running,
                idle=idle
            ))
        return state