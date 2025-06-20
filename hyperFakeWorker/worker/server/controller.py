import grpc
from traceback import print_exc
from time import sleep

from ..api.controller import controller_pb2_grpc
from ..api.common.common_pb2 import FunctionID, InstanceID, CallRequest, CallResponse, Error
from ..api.controller.controller_pb2 import StatusRequest, StatusUpdate, MetricsRequest, MetricsUpdate, StartResponse
from ..api.controller.controller_pb2 import VirtualizationType, Event, Status

from ..log import logger

from ..function.function import FunctionManager, Function
from ..utils.time import get_timestamp

class ControllerServicer(controller_pb2_grpc.ControllerServicer):

    def __init__(self):
        super().__init__()
        self._fn_mngr = FunctionManager()

    def Start(self, request: FunctionID, context: grpc.ServicerContext):
        logger.debug(f"Got Start call for function {request.id}")
        new_function = Function.create_new(self._fn_mngr, request.id, self._fn_mngr.get_image(request.id))
        with self._fn_mngr.function_lock:
            self._fn_mngr.add_function(new_function)
            self._fn_mngr.send_status_update(
                StatusUpdate(
                    instance_id=InstanceID(id=new_function.instance_id),
                    event=Event.Value("EVENT_START"),
                    status=Status.STATUS_SUCCESS,
                    function_id=FunctionID(id=new_function.function_id),
                )
            )
            logger.info(f"Scheduled instance {new_function.name} - {new_function.instance_id} for function {new_function.image} - {new_function.function_id}")
            return StartResponse(instance_id=InstanceID(id=new_function.instance_id), instance_ip="127.0.0.1", instance_name=new_function.name)
    
    def Call(self, request: CallRequest, context: grpc.ServicerContext):
        logger.debug(f"Got Call for function instance {request.instance_id.id} | {request.function_id.id}")
        tries = 0
        while True:
            if tries > 5:
                response = CallResponse(
                    request_id=request.request_id,
                    error=Error(message="Unable to schedule function call!")
                )
                return response
            func = self._fn_mngr.choose_function(request.function_id.id)
            tries += 1
            if func is None:
                sleep(0.1)
            else:
                break
        with func.work_lock:
            try:
                queued_ts = get_timestamp().ToNanoseconds()
                response = func.work(10)
                response_ts = get_timestamp().ToNanoseconds()
                if response:
                    self._fn_mngr.send_status_update(
                        StatusUpdate(
                            instance_id=InstanceID(id=func.instance_id),
                            event=Event.Value("EVENT_RESPONSE"),
                            status=Status.Value("STATUS_SUCCESS"),
                            function_id=FunctionID(id=func.function_id),
                        )
                    )
                else:
                    self._fn_mngr.send_status_update(
                        StatusUpdate(
                            instance_id=InstanceID(id=func.instance_id),
                            event=Event.Value("EVENT_DOWN"),
                            function_id=FunctionID(id=func.function_id),
                        )
                    )
                response = CallResponse(
                    request_id=request.request_id,
                    data=response,
                    instance_id=InstanceID(id=func.instance_id)
                )
                logger.debug(f"Returning response {response.__str__()}")
                context.set_trailing_metadata(
                    (
                        ("gotResponseTimestamp".lower(), str(response_ts)),
                        ("callQueuedTimestamp".lower(), str(queued_ts)),
                        ("functionProcessingTime".lower(), str(response_ts - queued_ts))
                    )
                )
                
            except Exception as e:
                print("Encountered error!")
                print_exc()
                response = CallResponse(
                    request_id=request.request_id,
                    error=Error(message="Encountered Unexpected error when executing function call!")
                )
            finally:
                return response
    
    def Stop(self, request: InstanceID, context: grpc.ServicerContext):
        logger.info(f"Got Stop call for function instance {request.id}")
        func = self._fn_mngr.remove_function(request.id)
        self._fn_mngr.send_status_update(
            StatusUpdate(
                instance_id=InstanceID(id=func.instance_id),
                event=Event.Value("EVENT_STOP"),
                status=Status.STATUS_SUCCESS,
                timestamp=get_timestamp()
            )
        )
        del func
        return InstanceID(request.id)
    
    def Status(self, request: StatusRequest, context: grpc.ServicerContext):
        # Collect functions...
        for update in self._fn_mngr.get_status_updates():
            logger.debug(f"Sent Status update:\nFunction: {update.function_id}\nInstance: {update.instance_id}\nEvent: {update.event.__str__()}\nStatus: {update.status.__str__()}\ntime: {update.timestamp.__str__()}")
            yield update
    
    def Metrics(self, request: MetricsRequest, context: grpc.ServicerContext):
        logger.debug(f"Got Metrics request!")
        return MetricsUpdate(
            used_ram_percent=1000.0,
            cpu_percent_percpu=[1.0,1.0,0.3]
        )