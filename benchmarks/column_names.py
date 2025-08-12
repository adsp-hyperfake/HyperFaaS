"""
Column name constants for the metrics table.
These are the column names as defined in new_import.py.
"""

# Primary key
REQUEST_ID = "request_id"

# Basic metadata
TIMESTAMP = "timestamp"
FUNCTION_ID = "function_id"
IMAGE_TAG = "image_tag"
INSTANCE_ID = "instance_id"

# Request status and response
STATUS = "status"
ERROR = "error"
GRPC_REQ_DURATION = "grpc_req_duration"

# Data size metrics
REQUEST_SIZE_BYTES = "request_size_bytes"
RESPONSE_SIZE_BYTES = "response_size_bytes"

# Timestamp metrics (all in unix nanoseconds)
CALL_QUEUED_TIMESTAMP = "call_queued_timestamp"
GOT_RESPONSE_TIMESTAMP = "got_response_timestamp"
LEAF_GOT_REQUEST_TIMESTAMP = "leaf_got_request_timestamp"
LEAF_SCHEDULED_CALL_TIMESTAMP = "leaf_scheduled_call_timestamp"

# Processing time metrics
FUNCTION_PROCESSING_TIME_NS = "function_processing_time_ns"

# Table names
METRICS_TABLE = "metrics"
CPU_MEM_STATS_TABLE = "cpu_mem_stats"

# CPU/Memory stats table columns
CPU_USAGE_PERCENT = "cpu_usage_percent"
MEMORY_USAGE = "memory_usage"
MEMORY_USAGE_LIMIT = "memory_usage_limit"
MEMORY_USAGE_PERCENT = "memory_usage_percent"


# Training data table columns . all of the duplicated ones are the same as the metrics table columns
FUNCTION_INSTANCES_COUNT = "function_instances_count"
ACTIVE_FUNCTION_CALLS_COUNT = "active_function_calls_count"
WORKER_CPU_USAGE = "worker_cpu_usage"
WORKER_RAM_USAGE = "worker_ram_usage"
FUNCTION_CPU_USAGE = "function_cpu_usage"
FUNCTION_RAM_USAGE = "function_ram_usage"