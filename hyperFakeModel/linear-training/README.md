# Linear Regression Training

Simple linear regression model training for HyperFake function performance prediction. This module provides a lightweight alternative to neural networks for cases where linear relationships are sufficient or when interpretability is important.

## Usage

### Training Models

Train linear regression models from a database of collected metrics:

```bash
# Using justfile (from project root)
just train-linear-regression ../../benchmarks/metrics.db models.json

Options:
  db      Path to SQLite database file (default: ./benchmarks/metrics)
  output  Output path for trained models (default: models.json)
```

### Command Line Options

```bash
uv run train_models.py [OPTIONS]

Options:
  --db     PATH      Path to SQLite database file (required)
  --output PATH  Output path for trained models (default: models.json)
```


## Troubleshooting
Linear regression models are lightweight, so if you try to run the load generator on the fake worker with linear regression models, you should probably reduce the load in config files. Otherwise, most requests will time out.