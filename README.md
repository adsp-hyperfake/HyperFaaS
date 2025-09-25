# HyperFaaS

HyperFaaS is a serverless platform with a tree-like load balancing structure. It consists of load balancer nodes that forward calls to worker nodes, which execute serverless functions.
The load balancer nodes that forward calls to worker nodes are called "leaf nodes".

## Architecture

HyperFaaS consists of three main components:

- **Load Balancer Nodes**: Schedule function calls to other nodes or workers and handle load balancing
- **Worker Nodes**: Execute the serverless functions in containers
- **Database**: Manages function metadata and configurations like resource limits

The platform can be run in two modes:
- **Containerized Mode**: All components (load balancer nodes, workers and database) run in Docker containers
- **Native Mode**: All components run directly on the host

## Getting Started
To get started with HyperFaaS, follow these steps:

### Prerequisites

- [Go](https://go.dev/doc/install)
- [Docker](https://docs.docker.com/get-docker/)
- [Protoc](https://protobuf.dev/installation/)
- [Just](https://github.com/casey/just?tab=readme-ov-file#installation)

> **Note**
> If you are running Windows, we heavily recommend using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to run HyperFaaS / justfile commands.

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/3s-rg-codes/HyperFaaS.git
   cd HyperFaaS
   ```

2. Build components and the Go functions:
   ```
   just build
   ```

### Running the Platform

#### Containerized Mode

Start all components with Docker Compose:
```
just d
```

Or with automatic rebuilding:
```
just start-rebuild
```

#### Native Mode

Run database, leaf, and worker components separately:

1. Start the database:
   ```
   cd ./cmd/database && go run .
   ```

2. Start a leaf node:
   ```
   just run-local-leaf
   ```

3. Start a worker node:
   ```
   just run-local-worker
   ```

## Developing Functions

Currently, HyperFaaS only supports Go as a language for serverless functions. Functions are executed as Docker containers.

To build a Go function:
```
just build-function-go function_name
```

To build all Go functions:
```
just build-functions-go
```

## Testing

Run all tests:
```
just test-all
```

Run integration tests:
```
just test-integration-containerized-all
```

## Cleanup

Remove all Docker containers/images and logs:
```
just clean
```

# HyperFake

HyperFake is a simulation framework for HyperFaaS that provides a simulated worker node as close as possible to the real worker. Instead of executing actual serverless functions, HyperFake uses machine learning models to predict function execution time and resource usage, enabling large-scale performance testing and analysis without the overhead of real function execution.

## Overview

HyperFake simulates the most compute and memory-intensive part of the HyperFaaS worker: function execution. For each function image, we train models to predict:

- **Execution Time**: Function runtime in nanoseconds
- **CPU Usage**: CPU cores consumed during execution  
- **Memory Usage**: RAM bytes consumed during execution

The framework includes tools for data generation, model training, and deployment of the simulated workers.

## Architecture

HyperFake consists of several key components:

- **Data Collection Pipeline** ([`benchmarks/`](./benchmarks/), [`pkg/loadgen`](./pkg/loadgen/)): Tools for generating training data from real HyperFaaS runs
- **Machine Learning Models** ([`hyperFakeModel/`](./hyperFakeModel/)): Multiple algorithm implementations for function behavior prediction
- **Fake Workers**: Simulated worker implementations in both Python ([`hyperFakeWorker/`](./hyperFakeWorker/)) and Go
- **Benchmarking Tools**: Load generators and metrics collection and plotting systems

## Quick Start

### 1. Data Generation

First, build the functions:

```bash
just build-functions-go
```

Then, generate training data by running HyperFaaS with metrics collection:

```bash
# Start HyperFaaS
just start

# In another terminal, start metrics client
just metrics-client

# Run load generator (optionally with config file)
just run-local-pipeline <config_file> <output_file>
```

This will generate metrics data for the functions specified in the used config.

#### 1.1 Different VMs Setup

To improve the quality of the results, you probably want to run HyperFaaS and the load generator on different VMs/machines. For that purpose, we have a script designed specifically for SUT and Client.

On the machine where you run the load generator, create or use a config file. You can see the existing config files in the `benchmarks/configs` folder. We have a script `pull-metrics.sh` that will pull the metrics from the SUT and save them to a local database. Make sure to configure it correctly.

```bash
just run-full-pipeline <config_file> <output_file>
```

The `output_file` is the CSV file where the load generator will save the results. The results will automatically be processed and saved to a SQLite database inside the `benchmarks` folder. After that, everything gets moved to the `~/training_data` folder.

Now you are ready to train models.

### 2. Model Training

HyperFake supports multiple machine learning algorithms:

- [**Linear Regression**](./hyperFakeModel/linear-training/README.md)
- [**Ridge Regression**](./hyperFakeModel/ridge-regression/README.md)
- [**Deep Neural Network**](./hyperFakeModel/neural-network/README.md)
- [**Random Forest**](./hyperFakeModel/random-forest/README.md)

Each README contains information on how to train models on generated data and use them in the fake worker.

There is also a [folder](./hyperFakeModel/examples/) with other algorithm examples. They are deprecated but can be used as inspiration for the development of other algorithms.

## Training Data Format

Models are trained on the following input features:
- Request body size (bytes)
- Number of function instances
- Number of active function calls
- Worker CPU usage (cores)
- Worker RAM usage (bytes)

And predict these output values:
- Function runtime (nanoseconds)
- Function CPU usage (cores)
- Function RAM usage (bytes)

Each row should contain the information for one function call.

| Parameter | Body Size | Number of Function Instances | Number of active Function Calls | Worker CPU Usage | Worker RAM Usage | Function Runtime | Function CPU Usage | Function RAM Usage |
| --------- | --------- | ---------------------------- | ------------------------------- | ---------------- | ---------------- | ---------------- | ------------------ | ------------------ |
| Unit      | Byte      | Count                        | Count                           | Cores            | Bytes            | Nanoseconds      | Cores              | Bytes              |
| Type      | Int64     | Int64                        | Int64                           | Float64          | Int64            | Int64            | Float64            | Int64              |


### 3. Running HyperFake

HyperFake provides two worker implementations:

- **Go Implementation**: Production-ready fake worker integrated with the main HyperFaaS codebase
- **Python Implementation** ([`hyperFakeWorker/`](./hyperFakeWorker/)): Standalone Python implementation for development and testing

The Go implementation is recommended for experiments and production use.

#### Usage
First, move the generated models (.onnx or .json) to `./hyperFakeWorker/models`.
Then start the simulated HyperFaaS system:

```bash
# Run with ONNX models
just fake-start fake-onnx

# Or run with linear regression json models
just fake-start fake-linear
```

Now you can send requests to the worker. The worker is started with all functions saved in `./hyperFakeWorker/models`.

## Adding New Functions

To add support for a new function in the pipeline:

1. **Create the function**: Add your function to [`functions/go/`](./functions/go/) and build it with `just build-function-go <function_name>`

2. **Add to configuration**: Add your function to the load generation config file (e.g., [`benchmarks/configs/local.yaml`](./benchmarks/configs/local.yaml)) with appropriate resource limits and data providers:
   ```yaml
   function_config:
     hyperfaas-my-function:latest:
       memory: 512MB
       cpu:
         period: 100000
         quota: 50000
   data_providers:
     hyperfaas-my-function:latest:
       type: "my_function"
       # Add function-specific parameters
   patterns:
     my_function:
       image_tag: hyperfaas-my-function:latest
       # Add load pattern configuration
   ```