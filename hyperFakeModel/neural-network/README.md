# Neural Network Model (MLP)
Trains a multi-layer perceptron and exports the model as .ONNX.

For each function, we use [Optuna][0] to establish hyperparameters, then train and export the final model in the [ONNX][1] format. This can be done in a few steps:

1. Copy the database (or databases) to train the models on to `./hyperFakeModel/training_dbs`.
2. In the project root, run `just neural-clean` to prepare the `./hyperFakeModel/neural-network/models` folder. Its contents will get moved to a subfolder.
3. Set up the venv by running `just neural-setup-venv`
4. Optionally test the setup, e.g. by running `just neural-optuna-test hyperfaas-echo`. This will perform a short Optuna optimization for the `hyperfaas-echo` function and automatically cleans up after itself.
5. Establish the hyperparameters for each function.
   In separate tmux windows, run the following command for each function (function name as specified in `data-providers` section in your config, e.g. `hyperfaas-bfs-json`):
   - `just neural-optuna <function-name>`

   This process can take many hours, depending on the hardware setup and size of the metrics database.
6. Finally, train the models.
   In separate tmux windows, run the following command for each function (function name as specified in `data-providers` section in your config, e.g. `hyperfaas-bfs-json`):
   - `just neural-train-model <function-name>`

   This will result in a `$function.onnx` model file for each function.
7. Copy the models to the target folder: `just neural-copy-models`

## Training on a subset of the training data

By default, step 6 will train on all the following columns of the training data:

- "request_body_size"
- "function_instances_count"
- "active_function_calls_count"
- "worker_cpu_usage"
- "worker_ram_usage"

In case you want to train on a subset of the columns, run

`just neural-train-model-cols <function-name> "<space-separated columns>"`.

For example, run

`just neural-train-model-cols hyperfaas-bfs-json "worker_cpu_usage worker_ram_usage"`

to train the hyperfaas-bfs-json function on just the two columns "worker_cpu_usage worker_ram_usage".

## Parameters

```text
Usage: neural_net_cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  manual
  optuna
```

```text
Usage: neural_net_cli.py optuna [OPTIONS]

Options:
  --cpu                         Use CPU for training instead of GPU
  --dbs-dir PATH                Path to directory containing database files
                                [default: /path/to/HyperFaaS/hyperFakeModel/training_dbs]
  --export-dir PATH             Target directory for saving the trained models
                                [default: /path/to/HyperFaaS/hyperFakeModel/neural-network/models]
  --func-tag TEXT               Function tags to train (can be specified
                                multiple times)  [default: hyperfaas-bfs-
                                json:latest, hyperfaas-thumbnailer-
                                json:latest, hyperfaas-echo:latest]
  --short-name TEXT             Short names corresponding to function tags
                                [default: bfs, thumbnailer, echo]
  --table-name TEXT             The db's table name containing the training
                                data  [default: training_data]
  --sample-data                 Use sample data instead of real data
  --input-cols TEXT             Input columns for training (can be specified
                                multiple times)  [default: request_body_size,
                                function_instances_count,
                                active_function_calls_count, worker_cpu_usage,
                                worker_ram_usage]
  --output-cols TEXT            Output columns for training (can be specified
                                multiple times)  [default: function_runtime,
                                function_cpu_usage, function_ram_usage]
  --trials INTEGER RANGE        Number of trials  [default: 20; x>=1]
  --jobs INTEGER RANGE          Number of parallel jobs (-1 for # of CPUs)
                                [default: 5; x>=-1]
  --epochs INTEGER RANGE        Number of training epochs  [default: 50; x>=1]
  --final-epochs INTEGER RANGE  Number of training epochs for the final
                                exported model  [default: 100; x>=1]
  --samples INTEGER RANGE       Number of samples to train on [-1 uses all
                                data]  [x>=-1]
  --save-state                  Save the state of the study to a db
  --shared-db PATH              Optional path to a db saving the state of the
                                study
  --study-id TEXT               Unique ID of the study
  --help                        Show this message and exit.
```

```text
Usage: neural_net_cli.py manual [OPTIONS]

Options:
  --cpu                    Use CPU for training instead of GPU
  --dbs-dir PATH           Path to directory containing database files
                           [default: /path/to/HyperFaaS/hyperFakeModel/training_dbs]
  --export-dir PATH        Target directory for saving the trained models
                           [default: /path/to/HyperFaaS/hyperFakeModel/neural-network/models]
  --func-tag TEXT          Function tags to train (can be specified multiple
                           times)  [default: hyperfaas-bfs-json:latest,
                           hyperfaas-thumbnailer-json:latest, hyperfaas-
                           echo:latest]
  --short-name TEXT        Short names corresponding to function tags
                           [default: bfs, thumbnailer, echo]
  --table-name TEXT        The db's table name containing the training data
                           [default: training_data]
  --sample-data            Use sample data instead of real data
  --input-cols TEXT        Input columns for training (can be specified
                           multiple times)  [default: request_body_size,
                           function_instances_count,
                           active_function_calls_count, worker_cpu_usage,
                           worker_ram_usage]
  --output-cols TEXT       Output columns for training (can be specified
                           multiple times)  [default: function_runtime,
                           function_cpu_usage, function_ram_usage]
  --hyperparams PATH       Path to hyperparameters json file
  --epochs INTEGER RANGE   Number of training epochs  [default: 100; x>=1]
  --samples INTEGER RANGE  Number of samples to train on [-1 uses all data]
                           [x>=-1]
  --help                   Show this message and exit.
```

## Performance

We applied the improvements mentioned in the [official performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

We achieved performance improvements of up to 20% for training on GPUs and 30% for training on CPU.

You can read more about that [here](./PERFORMANCE.md)

[0]: https://optuna.org/
[1]: https://onnx.ai/