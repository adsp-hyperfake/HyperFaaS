# Model Training Parameters Guide

This guide explains how to use the parameterized model training commands in the HyperFaaS project.

## Overview

Both Random Forest and Ridge Regression training scripts now support customizable parameters through command-line arguments. The justfile has been updated to pass these parameters through, with sensible defaults matching the original hardcoded values.

## Random Forest Parameters

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of trees in the forest |
| `max_depth` | int | None | Maximum depth of trees (None = unlimited) |
| `min_samples_split` | int | 2 | Minimum samples required to split an internal node |
| `min_samples_leaf` | int | 1 | Minimum samples required to be at a leaf node |
| `max_features` | str | "sqrt" | Number of features to consider when looking for the best split |
| `random_state` | int | 42 | Random state for reproducibility |
| `n_jobs` | int | -1 | Number of jobs to run in parallel (-1 = use all cores) |

### Usage Examples

#### Using justfile commands:

```bash
# Default parameters (same as before)
just train-random-forest

# Custom parameters
just train-random-forest ../../benchmarks/metrics.db training_data 200 15 5 2 sqrt 42 -1
```

#### Direct script usage:

```bash
cd hyperFakeModel/random-forest
uv run random_forest.py --n-estimators 150 --max-depth 20 --min-samples-split 5
```

## Ridge Regression Parameters

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha_min` | float | -3 | Minimum alpha exponent for logspace (10^alpha_min) |
| `alpha_max` | float | 3 | Maximum alpha exponent for logspace (10^alpha_max) |
| `alpha_num` | int | 25 | Number of alpha values to try |
| `cv_folds` | int | 5 | Number of cross-validation folds |
| `test_size` | float | 0.2 | Fraction of data to use for testing |
| `val_size` | float | 0.25 | Fraction of remaining data to use for validation |
| `random_state` | int | 42 | Random state for reproducibility |

### Usage Examples

#### Using justfile commands:

```bash
# Default parameters (same as before)
just train-ridge-regression

# Custom parameters
just train-ridge-regression ../../benchmarks/metrics.db training_data -2 2 30 10 0.15 0.3 42
```

#### Direct script usage:

```bash
cd hyperFakeModel/ridge-regression
uv run ridge_regression.py --alpha-min -2 --alpha-max 2 --alpha-num 30 --cv-folds 10
```

## Parameter Selection Guidelines

### Random Forest

- **Small datasets** (< 1000 samples): Use fewer trees (50-100) and limit depth (10-15)
- **Large datasets** (> 10000 samples): Use more trees (200-500) and deeper trees (20-30)
- **High noise**: Increase `min_samples_split` and `min_samples_leaf`
- **Many features**: Consider `max_features="log2"` instead of `"sqrt"`

### Ridge Regression

- **Start with wide alpha search** (-5 to 5) to find the right order of magnitude
- **Then narrow down** around the best performing alpha
- **More CV folds** (10+) for smaller datasets, fewer (3-5) for larger datasets
- **Adjust data splits** based on dataset size and validation needs

## Monitoring Training

Both scripts now print the parameters being used:

```
Using model parameters: {'n_estimators': 200, 'max_depth': 20, ...}
Training Random Forest with parameters: {...}
```

This helps verify that your parameters are being applied correctly.

## Integration with Existing Workflow

The parameterized training integrates seamlessly with the existing HyperFaaS workflow:

1. **Data collection**: Use existing load generation and metrics collection
2. **Training**: Use new parameterized commands to experiment with different models
3. **Deployment**: Export to ONNX as before, use with fake workers
4. **Evaluation**: Compare performance across different parameter combinations

## Tips for Experimentation

1. **Start with defaults** to establish a baseline
2. **Change one parameter at a time** to understand its impact
3. **Use the convenience presets** to quickly test different configurations
4. **Monitor validation metrics** to avoid overfitting
5. **Save promising configurations** by documenting the parameter combinations that work well
