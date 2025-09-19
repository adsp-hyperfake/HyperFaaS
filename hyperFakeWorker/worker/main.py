#!/bin/python
import logging
from pathlib import Path

import click

from .api import add_proto_definitions
from .config import WorkerConfig
from .log import setup_logger

add_proto_definitions()

@click.group()
@click.option('--address', default='', help='Worker address.')
@click.option('--database-type', default='http', help='Type of the database.')
@click.option('--runtime', default='docker', help='Container runtime type.')
@click.option('-w', '--workers', default=32, type=int)
@click.option('--max-rpcs', 'maxrpcs', default=256, type=int)
@click.option('--timeout', default=20, type=int, help='Timeout in seconds before leafnode listeners are removed from status stream updates.')
@click.option('--auto-remove', is_flag=True, help='Auto remove containers.')
@click.option('--log-level', default='info', help='Log level (debug, info, warn, error)')
@click.option('--log-format', default='text', help='Log format (json or text)')
@click.option('--log-file', default=None, help='Log file path (defaults to stdout)')
@click.option('--containerized', is_flag=True, help='Use socket to connect to Docker.')
@click.option("-m", "--model", "model", multiple=True, nargs=2, default=[], help='Add a model: <model_path> <image_name>')
@click.option('--update-buffer-size', default=None, type=int, help='Update buffer size.')  
@click.pass_context
def main(ctx, address, database_type, runtime, workers, maxrpcs, timeout, auto_remove, log_level, log_format, log_file, containerized, update_buffer_size, model):
    setup_logger(log_level, log_file)

    db_address = "localhost:8999"
    if containerized:
        db_address = "database:8999"

    if update_buffer_size is None:
        # If maxsize is <= 0, the queue size is infinite.
        update_buffer_size = -1

    models_dict = {}
    for model_path_str, image_name in model:
        model_path = Path(model_path_str)
        if not model_path.exists():
            raise click.BadParameter(f"Model file does not exist: {model_path}")
        models_dict[image_name] = model_path

    # Pass context to other commands
    ctx.obj = WorkerConfig(
        # General
        address=address or "[::]:50051",
        database_type=database_type or "http",
        timeout=timeout,  # not used
        max_workers=workers,
        max_rpcs=maxrpcs,

        # Runtime
        runtime=runtime,  # not used
        auto_remove=auto_remove,  # not used
        containerized=containerized,

        # Log
        log_level=log_level,
        log_format=log_format,  # not implemented
        log_file=log_file,

        update_buffer_size=update_buffer_size,

        # Extra parameters
        db_address=db_address,

        # Models
        models=models_dict
    )

@main.command()
@click.pass_obj
def server(config: WorkerConfig):
    from .server.server import serve

    serve(config)

@main.command()
def client():
    
    from .log import logger
    import grpc
    from .api.controller.controller_pb2_grpc import ControllerStub
    from .api.controller.controller_pb2 import InstanceStateRequest, InstanceState
    from .api.common.common_pb2 import FunctionID, InstanceID, CallRequest, CallResponse
    channel = grpc.insecure_channel("localhost:50051")
    stub = ControllerStub(channel)

    logger.info("Testing grpc server...")

    state: StateResponse = stub.State(StateRequest(node_id="12345"))
    for function in state.functions:
        print(f"State of functions: {function.function_id}")
        print(f"Checking {len(function.idle)} idle function instances:")
        for func in function.idle:
            print(f"Checking instance {func.instance_id}: {func}")
            call_future = stub.Call(CallRequest(instance_id=InstanceID(id=func.instance_id), function_id=function.function_id))
            print(call_future)
        print(f"Checking {len(function.running)} running function instances:")
        for func in function.running:
            print(f"Checking instance {func.instance_id}: {func}")
            call_future = stub.Call(CallRequest(instance_id=InstanceID(id=func.instance_id), function_id=function.function_id))
            print(call_future)


@main.command()
@click.pass_context
def test_call(ctx):
    """Calls the worker with an echo function."""
    import grpc
    import httpx

    from .api.common.common_pb2 import CallRequest, FunctionID, InstanceID
    from .api.controller.controller_pb2_grpc import ControllerStub

    logger = logging.getLogger()
    config: WorkerConfig = ctx.obj

    # Register the function in the key-value store first  
    # Use the first available model for testing
    models_dir = Path("models")
    available_models = list(models_dir.glob("*.onnx"))
    if not available_models:
        logger.error("No ONNX models found for testing")
        return
    
    # Use the first model found (e.g. hyperfaas-echo.onnx -> hyperfaas-echo:latest)
    model_file = available_models[0]
    function_name = model_file.stem  # e.g. hyperfaas-echo
    function_id_str = f"{function_name}:latest"
    kv_url = config.db_address
    if not kv_url.startswith("http://"):
        kv_url = f"http://{kv_url}"
    function_metadata = {
        "image_tag": function_id_str,
        "config": {
            "mem_limit": 128,
            "cpu_quota": 100000,
            "cpu_period": 100000,
            "timeout": 10,
            "max_concurrency": 1
        }
    }
    try:
        response = httpx.post(kv_url, json=function_metadata)
        response.raise_for_status()
        data = response.json()
        # Expecting the store to return a JSON with the function id, e.g. {"FunctionID": "..."}
        real_function_id = data.get("function_id", function_id_str)
        logger.info(
            f"Registered function {real_function_id} in key-value store.")
    except Exception as e:
        logger.error(f"Failed to register function in key-value store: {e}")
        real_function_id = function_id_str

    # Connect to the worker
    worker_port = config.address.split(":")[-1]
    channel = grpc.insecure_channel(f"localhost:{worker_port}")
    stub = ControllerStub(channel)

    # Start a function instance
    function_id = FunctionID(id=real_function_id)
    start_response = stub.Start(function_id)
    instance_id = start_response.instance_id.id

    # Call the function
    call_request = CallRequest(
        function_id=function_id,
        instance_id=InstanceID(id=instance_id),
        data=b"hello"
    )
    response = stub.Call(call_request)
    logger.info(f"Function call response: {response}")

@main.command()
@click.option('--leaf-address', default='localhost:50050', help='Address of the leaf server.')
@click.pass_context
def test_call_leaf(ctx, leaf_address):
    import grpc
    
    from .api.leaf.leaf_pb2_grpc import LeafStub
    from .api.leaf.leaf_pb2 import CreateFunctionRequest, ScheduleCallRequest
    from .api.common.common_pb2 import ImageTag, Config, CPUConfig
    
    logger = logging.getLogger()
    
    
    channel = grpc.insecure_channel(leaf_address)
    stub = LeafStub(channel)
    
    request = CreateFunctionRequest(
        image_tag=ImageTag(tag="hyperfaas-echo:latest"),
        config=Config(
            memory=100 * 1024 * 1024,  # 100MB
            cpu=CPUConfig(
                period=100000,
                quota=50000
            ),
            max_concurrency=500,
            timeout=10
        )
    )
    
    # Call CreateFunction
    response = stub.CreateFunction(request)
    logger.info(f"Created function with ID: {response.functionID.id}")
    
    call_request = ScheduleCallRequest(
        functionID=response.functionID,
        data=b"hello"
    )
    
    # Call ScheduleCall
    call_response = stub.ScheduleCall(call_request)
    if "error" in call_response:
        logger.error(f"Error in call response: {call_response.error.message}")
        return
    logger.info(f"Call response: {call_response}")
    
    
    

if __name__ == "__main__":
    main()
