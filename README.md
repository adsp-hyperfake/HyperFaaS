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

2. Build components and the go functions:
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
The HyperFake project aims to provide a simulated HyperFaaS worker that is as close as possible to the real worker.
To do this, we use different models to simulate the behavior of the real worker. Because the most compute and memory intensive part of the worker is (most likely) the function execution, we train a model per function image to predict the execution time and its resource usage.
Additionally, we provide a set of tools to generate load, measure metrics and train models.

## Generating Data
We have modified the normal HyperFaaS components to collect metrics. To generate data, you have to first run HyperFaaS:
```
just start
```
In another terminal, run the metrics client:
```
just metrics-client
```

To improve the quality of the results, you probably want to run HyperFaaS and the load generator in different VMs/ machines.

In the machine where you run the load generator, create or use a config file. You can see the existing config files in the `benchmarks/configs` folder.
We have a script `pull-metrics.sh` that will pull the metrics from the SUT and save them to a local database. Make sure to configure it correctly.

Then, you can run the load generator:
```
just run-full-pipeline <config_file> <out_file>
```
The `out_file` is the csv file where the load generator will save the results.
The results will automatically be processed and saved to a sqlite database inside the `benchmarks` folder.
After that, everything gets moved to the `~/training_data` folder.

Now you are ready to train models.

## Training Models - Neural Network

For each function, we use [Optuna][0] to establish hyperparameters, then train and export the final model in the [ONNX][1] format. This can be done in a few steps:

1. Copy the database (or databases) to train the models on to `./hyperFakeModel/training_dbs`.
2. Run `just neural-clean` to prepare the `./hyperFakeModel/neural-network/models` folder. Its contents will get moved to a subfolder.
3. Set up the venv by running `just neural-setup-venv`
4. Optionally test the setup, e.g. by running `just neural-optuna-test echo`. This will perform a short Optuna optimization for the `echo` function and automatically cleans up after itself.
5. Establish the hyperparameters for each function.
   In seperate tmux windows, run the following commands.
   - `just neural-optuna bfs-json`
   - `just neural-optuna thumbnailer-json`
   - `just neural-optuna echo`

   This process will take many hours, depending on the hardware setup.
6. Finally, train the models by running the following commands in seperate tmux windows:
   - `just neural-train-model bfs-json`
   - `just neural-train-model thumbnailer-json`
   - `just neural-train-model echo`

   This will result in a `$function.onnx` and `$function.onnx.data` file for each function.
7. Copy the models to the target folder: `just neural-copy-models`

### Training on a subset of the training data

By default, step 6 will train on all the following columns of the training data:

- "request_body_size"
- "function_instances_count"
- "active_function_calls_count"
- "worker_cpu_usage"
- "worker_ram_usage"

In case you want to train on a subset of the columns, run

`just neural-train-model-cols function "space-separated columns"`.

For example, run

`just neural-train-model-cols bfs-json "worker_cpu_usage worker_ram_usage"`

to train the bfs-json function on just the two columns "worker_cpu_usage worker_ram_usage".
### [hyperFake Model](./hyperFakeModel/README.md)

## Training Models - Random Forest

TODO

## HyperFake Workers
There is two implementations of the HyperFake worker, one in Python (./hyperFakeWorker/) and one in Go, which is a modification of the real worker.
The Go implementation is the one that was last used for the experiments.
In order to run the Go fake worker, you just have to provide the appropiate flags to it.
We recommend doing so with the justfile command:

```justfile
# make sure that the onnx models are in hyperFakeModel/
fake-start runtime_type:
    FAKE_RUNTIME_TYPE={{runtime_type}} WORKER_TYPE=fake-worker docker compose up --scale worker=0 fake-worker leaf database -d --build
```
If the runtime type is `fake-onnx`, the worker will load the onnx models using a hard coded mapping for simplicity:

```
"hyperfaas-echo:latest":             "echo.onnx",
"hyperfaas-bfs-json:latest":         "bfs-json.onnx",
"hyperfaas-thumbnailer-json:latest": "thumbnailer-json.onnx",
```
### Adding new models

To add new models, you have to modify the main.go of the worker and include a new mapping of the image name to the model name, and make sure to add the model to the `./hyperFakeModel/` folder.


[0]: https://optuna.org/
[1]: https://onnx.ai/