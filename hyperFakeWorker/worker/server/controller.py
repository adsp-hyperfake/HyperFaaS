import grpc

from .._grpc.controller import controller_pb2_grpc
from .._grpc.common.common_pb2 import FunctionID, InstanceID, CallRequest, CallResponse, Error
from .._grpc.controller.controller_pb2 import StatusRequest, StatusUpdate, MetricsRequest, MetricsUpdate, StateRequest, StateResponse
from .._grpc.controller.controller_pb2 import VirtualizationType, Event, Status

from ..log import logger

from .function import FunctionManager, Function
from ..utils.time import get_timestamp

class ControllerServicer(controller_pb2_grpc.ControllerServicer):

    def __init__(self):
        super().__init__()
        self._fn_mngr = FunctionManager()

    def Start(self, request: FunctionID, context: grpc.ServicerContext):
        new_function = Function.create_new(request.id)
        self._fn_mngr.add_function(new_function)
        self._fn_mngr.send_status_update(
            StatusUpdate(
                instance_id=InstanceID(id=new_function.instance_id),
                event=Event.EVENT_START,
                status=Status.STATUS_SUCCESS,
                function_id=FunctionID(id=new_function.function_id),
            )
        )
        return InstanceID(id=new_function.instance_id)
    
    def Call(self, request: CallRequest, context: grpc.ServicerContext):
        func = self._fn_mngr.get_function(request.instance_id.id)
        response = func.work(10)
        if response:
            self._fn_mngr.send_status_update(
                StatusUpdate(
                    instance_id=InstanceID(id=func.instance_id),
                    event=Event.EVENT_RESPONSE,
                    status=Status.STATUS_SUCCESS,
                    function_id=FunctionID(id=func.function_id),
                )
            )
        else:
            self._fn_mngr.send_status_update(
                StatusUpdate(
                    instance_id=InstanceID(id=func.instance_id),
                    event=Event.EVENT_DOWN,
                    function_id=FunctionID(id=func.function_id),
                )
            )
        response = CallResponse(
            data=response
        )
        logger.debug(f"Returning response {response.__str__()}")
        context.set_trailing_metadata(
            (
                ("gotResponseTimestamp".lower(), str(get_timestamp().ToSeconds())),
				("callQueuedTimestamp".lower(),  str(get_timestamp().ToSeconds())),
            )
        )
        return response
    
    def Stop(self, request: InstanceID, context: grpc.ServicerContext):
        func = self._fn_mngr.remove_function(request.id)
        self._fn_mngr.send_status_update(
            StatusUpdate(
                instance_id=InstanceID(id=func.instance_id),
                event=Event.EVENT_STOP,
                status=Status.STATUS_SUCCESS,
                timestamp=get_timestamp()
            )
        )
        return InstanceID(request.id)
    
    def Status(self, request: StatusRequest, context: grpc.ServicerContext):
        # Collect functions...
        for update in self._fn_mngr.get_status_updates():
            yield update
    
    def Metrics(self, request: MetricsRequest, context: grpc.ServicerContext):
        return MetricsUpdate(
            used_ram_percent=1000.0,
            cpu_percent_percpu=[1.0,1.0,0.3]
        )
    
    def State(self, request: StateRequest, context: grpc.ServicerContext):
        return StateResponse(
            functions=self._fn_mngr.get_state()
        )