import datetime
import time
import random
from hashlib import sha256
import threading
from queue import Queue

from blinker import signal

from . import FunctionIdStr, InstanceIdStr
from ..api.controller.controller_pb2 import StatusUpdate, Event, Status, FunctionState, InstanceState
from ..api.common.common_pb2 import InstanceID, FunctionID
from ..log import logger
from ..utils.time import get_timestamp

from ..kvstore.client import KVStoreClient
from .image import FunctionImage

class Function():

    def __init__(self, function_id: str, instance_id: str, image: FunctionImage):
        self.created_at = int(datetime.datetime.now().timestamp())
        self.last_worked_at = int(datetime.datetime.now().timestamp())
        self.work_lock: threading._RLock = threading.RLock()
        self.function_id = function_id
        self.instance_id = instance_id
        self.image = image

        self.is_cold = True

        self.status_signal = signal("status")

    @property
    def is_active(self):
        got_lock = self.work_lock.acquire(blocking=False)
        if got_lock:
            self.work_lock.release()
        return not got_lock

    @property
    def uptime(self):
        current_time = int(datetime.datetime.now().timestamp())
        return current_time - self.created_at
    
    @property
    def time_since_last_work(self):
        current_time = int(datetime.datetime.now().timestamp())
        return current_time - self.last_worked_at

    @staticmethod
    def create_new(function_id: str, image: FunctionImage):
        hash_source = function_id + str(random.randint(1, 2^31))
        return Function(
            function_id=function_id,
            instance_id=sha256(hash_source.encode(errors="ignore")).hexdigest(),
            image=image
        )
    
    def work(self, bytes: int):
        with self.work_lock:
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
            time.sleep(1)
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
        self.kvs_client = KVStoreClient("127.0.0.1:8999")
        self.images: dict[FunctionIdStr, FunctionImage] = {}
        self.data_lock = threading.RLock()

        self.status_signal = signal("status")

    def get_image(self, function_id: FunctionIdStr):#
        with self.data_lock:
            if self.images.get(function_id) is None:
                self.images[function_id] = self.kvs_client.get_image(function_id)
            return self.images[function_id]

    def get_num_active_functions(self, function_id: FunctionIdStr) -> int:
        with self.data_lock:
            return len(list(filter(lambda i: i.is_active, self.instances.get(function_id))))

    @property
    def num_active_functions(self) -> dict[FunctionIdStr, int]:
        with self.data_lock:
            active_funcs = {}
            for key, value in self.instances.items():
                active = list(filter(lambda i: i.is_active, value))
                active_funcs[key] = len(active)
            return active_funcs
    
    def get_num_functions(self, function_id: FunctionIdStr) -> int:
        with self.data_lock:
            return len(self.instances.get(function_id))

    @property
    def num_functions(self) -> dict[FunctionIdStr, int]:
        with self.data_lock:
            active_funcs = {}
            for key, value in self.instances.items():
                active_funcs[key] = len(value)
            return active_funcs

    def add_function(self, function: Function):
        with self.data_lock:
            self.active_functions[function.instance_id] = function

            if self.instances.get(function.function_id) is None:
                self.instances[function.function_id] = set()
            self.instances[function.function_id].add(function)

    def remove_function(self, instance_id: InstanceIdStr):
        with self.data_lock:
            if self.active_functions.get(instance_id) is None:
                return
            function = self.active_functions[instance_id]
            self.instances[function.function_id].remove(function)
            return self.active_functions.pop(instance_id)

    def get_function(self, instance_id: InstanceIdStr):
        with self.data_lock:
            try:
                return self.active_functions[instance_id]
            except KeyError as e:
                logger.critical(f"Failed to find instance_id {instance_id} in:\n{self.active_functions.keys()}")
                raise e
    
    def send_status_update(self, update: StatusUpdate):
        with self.data_lock:
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
        with self.data_lock:
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