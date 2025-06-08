# Neural Network Model (MLP)

(wip readme)

Trains a multi-layer perceptron and exports the model as .ONNX.

## Usage

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
  --dbs-dir PATH                Path to directory containing database files
                                [default: /path/to/HyperFaaS/hyperFakeModel/
                                training_dbs]
  --export-dir PATH             Target directory for saving the trained models
                                [default: /path/to/HyperFaaS/hyperFakeModel/
                                neural-network/models]
  --func-tag TEXT               Function tags to train (can be specified
                                multiple times)  [default: hyperfaas-bfs-
                                json:latest, hyperfaas-thumbnailer-
                                json:latest, hyperfaas-echo:latest]
  --short-name TEXT             Short names corresponding to function tags
                                [default: bfs, thumbnailer, echo]
  --table-name TEXT             The db's table name containing the training
                                data  [default: training_data]
  --sample-data                 Use sample data instead of real data
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
  --dbs-dir PATH           Path to directory containing database files
                           [default: /path/to/HyperFaaS/hyperFak
                           eModel/training_dbs]
  --export-dir PATH        Target directory for saving the trained models
                           [default: /path/to/HyperFaaS/hyperFakeModel/
                           neural-network/models]
  --func-tag TEXT          Function tags to train (can be specified multiple
                           times)  [default: hyperfaas-bfs-json:latest,
                           hyperfaas-thumbnailer-json:latest, hyperfaas-
                           echo:latest]
  --short-name TEXT        Short names corresponding to function tags
                           [default: bfs, thumbnailer, echo]
  --table-name TEXT        The db's table name containing the training data
                           [default: training_data]
  --sample-data            Use sample data instead of real data
  --hyperparams PATH       Path to hyperparameters json file
  --epochs INTEGER RANGE   Number of training epochs  [default: 100; x>=1]
  --samples INTEGER RANGE  Number of samples to train on [-1 uses all data]
                           [x>=-1]
  --help                   Show this message and exit.
```