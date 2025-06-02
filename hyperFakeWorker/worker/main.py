#!/bin/python
from .api import add_proto_definitions
add_proto_definitions()

import click

@click.group()
def main():
    pass

@main.command()
def server():
    from .server.server import serve

    serve()

@main.command()
def client():
    import grpc
    from .api.controller.controller_pb2_grpc import ControllerStub
    from .api.controller.controller_pb2 import StatusRequest
    channel = grpc.insecure_channel("localhost:50051")
    stub = ControllerStub(channel)

    status_future = stub.Status(StatusRequest(nodeID="0"))
    status_future.result()


if __name__ == "__main__":
    main()
