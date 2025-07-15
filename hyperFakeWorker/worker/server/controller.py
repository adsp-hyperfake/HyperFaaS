import grpc
from traceback import format_exc
from time import sleep
from pathlib import Path

from ..exceptions import SchedulingException

from ..config import WorkerConfig

from ..api.controller import controller_pb2_grpc
from ..api.common.common_pb2 import FunctionID, InstanceID, CallRequest, CallResponse, Error
from ..api.controller.controller_pb2 import StatusRequest, StatusUpdate, MetricsRequest, MetricsUpdate, StartResponse
from ..api.controller.controller_pb2 import VirtualizationType, Event, Status

from ..log import logger

from ..managers.function import FunctionManager
from ..managers.model import ModelManager
from ..managers.image import ImageManager
from ..managers.status import StatusManager
from ..function.function import Function
from ..function.scheduler import FunctionScheduler, FunctionUnknownToSchedulerException
from ..utils.time import get_timestamp

class ControllerServicer(controller_pb2_grpc.ControllerServicer):

    def __init__(self, config: WorkerConfig):
        super().__init__()
        self._function_manager = FunctionManager()
        self._status_manager = StatusManager(config.update_buffer_size)
        self._scheduler = FunctionScheduler(self._function_manager)
        self._image_manager = ImageManager(db_address=config.db_address)
        self._model_manager = ModelManager(models=config.models)

    def Start(self, request: FunctionID, context: grpc.ServicerContext):
        logger.debug(f"Got Start call for function {request.id}")

        function_image = self._image_manager.get_image(request.id)
        model_path = self._model_manager.find_model(request.id, function_image)
        new_function = Function.create_new(self._function_manager, self._status_manager, request.id, function_image, model_path, self._model_manager)

        self._function_manager.add_function(new_function)

        self._status_manager.send_status_update(
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
        # Note: instance_id parameter is ignored, using function_id for round-robin scheduling
        logger.debug(f"Got Call for function {request.function_id.id} (ignoring instance_id {request.instance_id.id})")

        initial_trailers = context.trailing_metadata()
        initial_trailers_count = 0 if (initial_trailers is None) else len(context.trailing_metadata())

        try:
            # Use round-robin scheduler to get next available instance
            func = self._scheduler.get_instance_for_call(request.function_id.id)
            
            call_queued_timestamp = get_timestamp().ToNanoseconds()
            
            response_data, runtime = func.work(
                len(request.SerializeToString()), # Body size
                10 # number of return bytes
            )

            call_response_timestamp = get_timestamp().ToNanoseconds()

            if response_data: # Got a response
                self._status_manager.send_status_update(
                    StatusUpdate(
                        instance_id=InstanceID(id=func.instance_id),
                        event=Event.Value("EVENT_RESPONSE"),
                        status=Status.Value("STATUS_SUCCESS"),
                        function_id=FunctionID(id=func.function_id),
                    )
                )
            else: # Got no response
                self._status_manager.send_status_update(
                    StatusUpdate(
                        instance_id=InstanceID(id=func.instance_id),
                        event=Event.Value("EVENT_DOWN"),
                        function_id=FunctionID(id=func.function_id),
                    )
                )

            response = CallResponse(
                request_id=request.request_id,
                instance_id=InstanceID(id=func.instance_id),
                data=response_data
            )
            
            context.set_trailing_metadata(
                (
                    ("gotResponseTimestamp".lower(), str(call_response_timestamp)),
                    ("callQueuedTimestamp".lower(), str(call_queued_timestamp)),
                    ("functionProcessingTime".lower(), str(runtime))
                )
            )

            logger.debug(f"Returning response {response.__str__()}")
            return response
            
        except FunctionUnknownToSchedulerException as e:
            logger.error(f"Function {request.function_id.id} is not registered: {str(e)}")

            response = CallResponse(
                request_id=request.request_id,
                instance_id=request.instance_id,
                error = Error(message=f"Function {request.function_id.id} is not registered!")
            )
        except Exception as e:
            logger.error("Encountered error!")
            logger.error(format_exc())

            response = CallResponse(
                request_id=request.request_id,
                instance_id=request.instance_id,
                error = Error(message="Encountered unexpected error when executing function call!")
            )
        finally:
            current_trailers = context.trailing_metadata()
            current_trailers_count = 0 if (current_trailers is None) else len(context.trailing_metadata())
            if current_trailers_count == initial_trailers_count:
                logger.debug("The required grpc trailers are not set, setting dummy trailers...")
                context.set_trailing_metadata(
                    (
                        ("gotResponseTimestamp".lower(), str(0)),
                        ("callQueuedTimestamp".lower(), str(0)),
                        ("functionProcessingTime".lower(), str(0))
                    )
                )
                
        return response
    
    def Stop(self, request: InstanceID, context: grpc.ServicerContext):
        logger.info(f"Got Stop call for function instance {request.id}")
        func = self._function_manager.remove_function(request.id)
        self._status_manager.send_status_update(
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
        for update in self._status_manager.get_status_updates():
            logger.debug(f"Sent Status update:\nFunction: {update.function_id}\nInstance: {update.instance_id}\nEvent: {update.event.__str__()}\nStatus: {update.status.__str__()}\ntime: {update.timestamp.__str__()}")
            yield update
    
    def Metrics(self, request: MetricsRequest, context: grpc.ServicerContext):
        logger.debug(f"Got Metrics request!")
        return MetricsUpdate(
            used_ram_percent=self._function_manager.total_ram_usage,
            cpu_percent_percpu=[self._function_manager.total_cpu_usage]
        )