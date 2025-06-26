#!/bin/python
from pathlib import Path
import logging

import click

from .api import add_proto_definitions
from .config import WorkerConfig
from .log import setup_logger

add_proto_definitions()


@click.group()
@click.option('--address', default='', help='Worker address.')
@click.option('--database-type', default='http', help='Type of the database.')
@click.option('--runtime', default='docker', help='Container runtime type.')
@click.option('--timeout', default=20, type=int, help='Timeout in seconds before leafnode listeners are removed from status stream updates.')
@click.option('--auto-remove', is_flag=True, help='Auto remove containers.')
@click.option('--log-level', default='info', help='Log level (debug, info, warn, error)')
@click.option('--log-format', default='text', help='Log format (json or text)')
@click.option('--log-file', default=None, help='Log file path (defaults to stdout)')
@click.option('--containerized', is_flag=True, help='Use socket to connect to Docker.')
@click.option("-m", "--model", "model", multiple=True, default=[], type=click.Path(resolve_path=True, path_type=Path, dir_okay=False, exists=True))
@click.option('--update-buffer-size', default=None, type=int, help='Update buffer size.')  
@click.pass_context
def main(ctx, address, database_type, runtime, timeout, auto_remove, log_level, log_format, log_file, containerized, update_buffer_size, model):
    setup_logger(log_level, log_file)

    db_address = "localhost:8999"
    if containerized:
        db_address = "database:8999/"

    if update_buffer_size is None:
        # If maxsize is <= 0, the queue size is infinite.
        update_buffer_size = -1

    # Pass context to other commands
    ctx.obj = WorkerConfig(
        # General
        address=address or "[::]:50051",
        database_type=database_type or "http",
        timeout=timeout,  # not used

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
        models=model
    )


@main.command()
@click.pass_obj
def server(config: WorkerConfig):
    from .server.server import serve

    serve(config)


@main.command()
@click.pass_context
def client(ctx):
    import grpc

    from .api.common.common_pb2 import CallRequest, CallResponse, FunctionID, InstanceID
    from .api.controller.controller_pb2 import StatusRequest, StatusUpdate
    from .api.controller.controller_pb2_grpc import ControllerStub
    channel = grpc.insecure_channel("localhost:50051")
    stub = ControllerStub(channel)

    logger = logging.getLogger()
    logger.info("Testing grpc server with Status stream...")

    # Example: stream status updates for a node
    status_stream = stub.Status(StatusRequest(nodeID="12345"))
    for status_update in status_stream:
        print(
            f"Status update: instance_id={status_update.instance_id.id}, function_id={status_update.function_id.id}, event={status_update.event}, status={status_update.status}")


@main.command()
@click.pass_context
def test_call(ctx):
    import grpc
    import httpx

    from .api.common.common_pb2 import CallRequest, FunctionID, InstanceID
    from .api.controller.controller_pb2_grpc import ControllerStub

    logger = logging.getLogger()
    config: WorkerConfig = ctx.obj

    # Register the function in the key-value store first
    function_id_str = "hyperfaas-hello:latest"
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
    channel = grpc.insecure_channel("localhost:50051")
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
    print("Function call response:", response)


if __name__ == "__main__":
    main()
