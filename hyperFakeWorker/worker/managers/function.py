from statistics import mean
import threading
import time
from weakref import WeakSet

from ..function import FunctionIdStr, InstanceIdStr
from ..api.controller.controller_pb2 import StatusUpdate, FunctionState, InstanceState
from ..api.common.common_pb2 import FunctionID
from ..log import logger
from ..function.abstract import AbstractFunction

class FunctionManager():

    def __init__(self):
        self.function_lock = threading.RLock()
        # instance_id : Function
        self.instanceId_to_instance_map: dict[InstanceIdStr, AbstractFunction] = {}
        # function_id : set[Function]
        self.functionId_to_instances_map: dict[FunctionIdStr, WeakSet[AbstractFunction]] = {}
        # Round-robin state tracking
        self.round_robin_index: dict[FunctionIdStr, int] = {}

    @property
    def avg_remaining_runtime(self) -> float:
        now = time.time_ns()
        running_functions = [e for e in self.instanceId_to_instance_map.values() if e.is_active]
        if len(running_functions) < 1:
            return 0
        remaining_runtime = [e.last_worked_at - now for e in running_functions]
        average_remaining_runtime = mean(remaining_runtime)
        return average_remaining_runtime

    @property
    def total_cpu_usage(self) -> float:
        return sum([func.cpu for func in self.instanceId_to_instance_map.values()], 0)
    
    @property
    def total_ram_usage(self) -> int:
        return sum([func.ram for func in self.instanceId_to_instance_map.values()], 0)

    def get_num_recently_active_functions(self, function_id: FunctionIdStr) -> int:
        return len([e for e in self.functionId_to_instances_map.get(function_id) if e.was_recently_active])

    @property
    def num_recently_active_functions(self) -> int:
        return len([e for e in self.instanceId_to_instance_map.values() if e.was_recently_active])

    def get_num_inactive_functions(self, function_id: FunctionIdStr) -> int:
        return self.get_num_function_instances(function_id) - self.get_num_active_functions(function_id)

    def get_num_active_functions(self, function_id: FunctionIdStr) -> int:
        return len([e for e in self.functionId_to_instances_map.get(function_id) if e.is_active])

    @property
    def num_inactive_functions(self) -> int:
        return self.num_functions - self.num_active_functions

    @property
    def num_active_functions(self) -> int:
        return len([e for e in self.instanceId_to_instance_map.values() if e.is_active])
    
    def get_num_function_instances(self, function_id: FunctionIdStr) -> int:
        return len(self.functionId_to_instances_map.get(function_id))

    @property
    def num_functions(self) -> int:
        return len(self.instanceId_to_instance_map.values())

    def get_next_instance_round_robin(self, function_id: FunctionIdStr) -> AbstractFunction:
        """Get the next instance for the given function_id using round-robin scheduling."""
        with self.function_lock:
            instances = self.functionId_to_instances_map.get(function_id)
            if instances is None or len(instances) == 0:
                return None
            
            # Convert WeakSet to list for indexing
            instances_list = list(instances)
            if len(instances_list) == 0:
                return None
            
            # Get current round-robin index
            current_index = self.round_robin_index.get(function_id, 0)
            
            # Get the instance at current index
            selected_instance = instances_list[current_index]
            
            # Update round-robin index for next call
            self.round_robin_index[function_id] = (current_index + 1) % len(instances_list)
            
            return selected_instance

    def add_function(self, function: AbstractFunction):
        with self.function_lock:
            # Add to function instance map
            self.instanceId_to_instance_map[function.instance_id] = function

            # Add to set of all instances of an image
            if self.functionId_to_instances_map.get(function.function_id) is None:
                self.functionId_to_instances_map[function.function_id] = WeakSet()
                # Initialize round-robin index for new function
                self.round_robin_index[function.function_id] = 0
            self.functionId_to_instances_map[function.function_id].add(function)

    def remove_function(self, instance_id: InstanceIdStr):
        with self.function_lock:
            if self.instanceId_to_instance_map.get(instance_id) is None:
                return
            function = self.instanceId_to_instance_map[instance_id]
            self.functionId_to_instances_map[function.function_id].remove(function)
            
            # Reset round-robin index if no instances left
            if len(self.functionId_to_instances_map[function.function_id]) == 0:
                self.round_robin_index[function.function_id] = 0
            # Adjust round-robin index if it's out of bounds
            elif self.round_robin_index[function.function_id] >= len(self.functionId_to_instances_map[function.function_id]):
                self.round_robin_index[function.function_id] = 0
                
            return self.instanceId_to_instance_map.pop(instance_id)

    def get_function(self, instance_id: InstanceIdStr):
        try:
            return self.instanceId_to_instance_map[instance_id]
        except KeyError as e:
            logger.critical(f"Failed to find instance_id {instance_id} in:\n{self.instanceId_to_instance_map.keys()}")
            raise e
        
    def get_state(self) -> list[FunctionState]:
        with self.function_lock:
            state = []
            for function_id in self.functionId_to_instances_map.keys():
                functions = self.functionId_to_instances_map[function_id]
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