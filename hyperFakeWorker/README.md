# HyperFake Worker (Python Implementation)

A standalone Python implementation of the HyperFake worker that simulates HyperFaaS function execution using machine learning models. This implementation provides a development and testing environment for the HyperFake simulation framework.

## Overview

The Python HyperFake Worker implements a complete gRPC-based worker that:

- **Simulates Function Execution**: Uses ONNX models to predict execution time and resource usage instead of running actual functions
- **Maintains Worker State**: Tracks function instances, resource consumption, and scheduling decisions
- **Provides gRPC API**: Implements the same controller interface as the real HyperFaaS worker
- **Supports Hot/Cold Starts**: Simulates container lifecycle and cold start delays

## Architecture

```
hyperFakeWorker/
├── worker/
│   ├── api/           # Protocol buffer definitions and gRPC stubs
│   ├── function/      # Function simulation and lifecycle management
│   ├── managers/      # Resource and state management
│   ├── models/        # ONNX model inference and data structures
│   ├── server/        # gRPC server implementation
│   └── utils/         # Utilities (ONNX inference, timing)
├── models/            # Pre-trained ONNX models
└── tests/             # Unit tests
```

## Usage

### Running the Worker

The worker can be started using the justfile commands:

```bash
just run-local
```

### Configuration Options

```bash
uv run worker/main.py server [OPTIONS]

Options:
  --address TEXT              Worker address (default: [::]:50051)
  --database-type TEXT        Database type (default: http)
  --runtime TEXT              Container runtime (default: docker)
  -w, --workers INTEGER       Max worker threads (default: 32)
  --max-rpcs INTEGER          Max concurrent RPCs (default: 256)
  --timeout INTEGER           Timeout for leaf listeners (default: 20)
  --auto-remove              Auto remove containers
  --log-level TEXT           Log level (debug, info, warn, error)
  --log-format TEXT          Log format (json or text)
  --log-file TEXT            Log file path
  --containerized            Use Docker socket connection
  -m, --model TEXT TEXT      Add model: <model_path> <image_name>
  --update-buffer-size INTEGER Update buffer size
```

### Adding Models

Models can be added via command line or by placing ONNX files in the `models/` directory:

```bash
# Via command line
uv run worker/main.py server -m ./models/echo.onnx hyperfaas-echo:latest

# Via models directory (automatically discovered)
cp your-model.onnx ./models/hyperfaas-yourfunction.onnx
```

## Model Format

The worker expects ONNX or JSON models with the following interface:

### Input Format
- **Shape**: `[1, 5]` (batch size 1, 5 features)
- **Features**:
  1. Request body size (bytes)
  2. Number of function instances
  3. Number of active function calls
  4. Worker CPU usage (cores)
  5. Worker RAM usage (bytes)

### Output Format
- **Shape**: `[1, 3]` (batch size 1, 3 predictions)
- **Predictions**:
  1. Function runtime (nanoseconds)
  2. Function CPU usage (cores)
  3. Function RAM usage (bytes)

## Development

### Setup Development Environment

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run with development logging
uv run worker/main.py server --log-level debug
```

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/function/           # Function simulation tests
uv run pytest tests/utils/              # Utility function tests
uv run pytest tests/test_kvstore_client.py  # Key-value store tests
```


