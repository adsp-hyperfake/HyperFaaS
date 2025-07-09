from time import time_ns

from google.protobuf import timestamp_pb2 as _timestamp_pb2

NANOSECONDS = 1_000_000_000

def get_timestamp() -> _timestamp_pb2.Timestamp:
    now_ns = time_ns()
    seconds = now_ns // 1_000_000_000
    nanoseconds = now_ns % 1_000_000_000
    return _timestamp_pb2.Timestamp(seconds=seconds, nanos=nanoseconds)