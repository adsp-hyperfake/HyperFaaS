from pathlib import Path
import threading
from queue import Queue
from weakref import WeakSet, WeakValueDictionary

from . import FunctionIdStr, InstanceIdStr
from ..api.controller.controller_pb2 import StatusUpdate, FunctionState, InstanceState
from ..api.common.common_pb2 import FunctionID
from ..log import logger

from . import AbstractFunction
from ..kvstore.client import KVStoreClient
from .image import FunctionImage

class FunctionManager():

    def __init__(self, db_address: str, update_buffer_size: int):
        self.function_lock = threading.RLock()
        # instance_id : Function
        self.active_functions: dict[InstanceIdStr, AbstractFunction] = {}
        # function_id : set[Function]
        self.instances: dict[FunctionIdStr, set[AbstractFunction]] = {}
        self.images: dict[FunctionIdStr, FunctionImage] = {}

        self.kvs_client = KVStoreClient(db_address)
        
        self.status_lock = threading.RLock()
        self.status_queues: WeakSet[Queue] = WeakSet()
        
        self.update_buffer_size = update_buffer_size

    @property
    def total_cpu_usage(self) -> float:
        with self.function_lock:
            return sum([func.cpu for func in self.active_functions.values()], 0)
    
    @property
    def total_ram_usage(self) -> int:
        with self.function_lock:
            return sum([func.ram for func in self.active_functions.values()], 0)

    def get_image(self, function_id: FunctionIdStr):
        with self.function_lock:
            if self.images.get(function_id) is None:
                self.images[function_id] = self.kvs_client.get_image(function_id)
            return self.images[function_id]

    def get_num_recently_active_functions(self, function_id: FunctionIdStr) -> int:
        with self.function_lock:
            return len(list(filter(lambda i: i.was_recently_active, self.instances.get(function_id))))

    @property
    def num_recently_active_functions(self) -> dict[FunctionIdStr, int]:
        with self.function_lock:
            active_funcs = {}
            for key, value in self.instances.items():
                active = list(filter(lambda i: i.was_recently_active, value))
                active_funcs[key] = len(active)
            return active_funcs

    def get_num_active_functions(self, function_id: FunctionIdStr) -> int:
        with self.function_lock:
            return len(list(filter(lambda i: i.is_active, self.instances.get(function_id))))

    @property
    def num_active_functions(self) -> dict[FunctionIdStr, int]:
        with self.function_lock:
            active_funcs = {}
            for key, value in self.instances.items():
                active = list(filter(lambda i: i.is_active, value))
                active_funcs[key] = len(active)
            return active_funcs
    
    def get_num_functions(self, function_id: FunctionIdStr) -> int:
        with self.function_lock:
            return len(self.instances.get(function_id))

    @property
    def num_functions(self) -> dict[FunctionIdStr, int]:
        with self.function_lock:
            active_funcs = {}
            for key, value in self.instances.items():
                active_funcs[key] = len(value)
            return active_funcs

    def add_function(self, function: AbstractFunction):
        with self.function_lock:
            # Add to function instance map
            self.active_functions[function.instance_id] = function

            # Add to set of all instances of an image
            if self.instances.get(function.function_id) is None:
                self.instances[function.function_id] = set()
            self.instances[function.function_id].add(function)

    def remove_function(self, instance_id: InstanceIdStr):
        with self.function_lock:
            if self.active_functions.get(instance_id) is None:
                return
            function = self.active_functions[instance_id]
            self.instances[function.function_id].remove(function)
            return self.active_functions.pop(instance_id)

    def get_function(self, instance_id: InstanceIdStr):
        with self.function_lock:
            try:
                return self.active_functions[instance_id]
            except KeyError as e:
                logger.critical(f"Failed to find instance_id {instance_id} in:\n{self.active_functions.keys()}")
                raise e
            
    def choose_function(self, function_id: FunctionIdStr):
        with self.function_lock:
            available_functions = [func for func in self.instances[function_id] if not func.is_active]
            if len(available_functions) > 0:
                return available_functions[0]
            return None
    
    def send_status_update(self, update: StatusUpdate):
        if not isinstance(update, StatusUpdate):
            raise TypeError("The sent update must be an actual status update!")
        with self.status_lock:
            for q in self.status_queues:
                q.put(update)
            
    def get_status_updates(self):
        updates_queue: Queue[StatusUpdate] = Queue(maxsize=self.update_buffer_size)
        with self.status_lock:
            self.status_queues.add(updates_queue)
        
        try:
            while True:
                update = updates_queue.get()
                yield update
                updates_queue.task_done()
        finally:
            with self.status_lock:
                self.status_queues.remove(updates_queue)


    def get_state(self) -> list[FunctionState]:
        with self.function_lock:
            state = []
            for function_id in self.instances.keys():
                functions = self.instances[function_id]
                running = []
                idle = []
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
                if len(idle) <= 0 and len(running) <= 0:
                    state.append(FunctionState(
                        function_id=FunctionID(id=function_id),
                    ))
                elif len(idle) > 0 and len(running) <= 0:
                    state.append(FunctionState(
                        function_id=FunctionID(id=function_id),
                        idle=idle
                    ))
                elif len(idle) > 0 and len(running) > 0:
                    state.append(FunctionState(
                        function_id=FunctionID(id=function_id),
                        running=running,
                    ))
                else:
                    state.append(FunctionState(
                        function_id=FunctionID(id=function_id),
                        running=running,
                        idle=idle
                    ))
            return state