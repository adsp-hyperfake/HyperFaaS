# Ridge Regression Training

Ridge regression model training for HyperFake function performance prediction. This module provides regularized linear regression with L2 penalty, offering better generalization than simple linear regression while maintaining interpretability.

## Usage

### Training Models

Train ridge regression models from a database of collected metrics:

```bash
# Using justfile (from project root)
just train-ridge-regression ../../benchmarks/metrics.db training_data

Options:
  db_path       Path to SQLite database file (default: ../../benchmarks/metrics.db)
  table         Table name in database (default: training_data)
  alpha_min     Min alpha exponent for regularization (default: -3)
  alpha_max     Max alpha exponent for regularization (default: 3)
  alpha_num     Number of alpha values to test (default: 25)
  cv_folds      Cross-validation folds (default: 5)
  test_size     Test set fraction (default: 0.2)
  val_size      Validation set fraction (default: 0.25)
  random_state  Random seed (default: 42)
```

## Parameter Selection

### Key Parameters

Choose optimal parameters based on your data characteristics and regularization needs:

#### **alpha_min / alpha_max** (Regularization Range)
- **Small datasets**: Use wider range (-4 to 4) for more exploration
- **Medium datasets**: Standard range (-3 to 3) works well
- **Large datasets**: Narrow range (-2 to 2) for efficiency
- **Rule**: Larger range = more thorough search but slower training

#### **alpha_num** (Number of Alpha Values)
- **Quick testing**: 10-15 values
- **Standard training**: 25 values (default)
- **Thorough search**: 50-100 values
- **Rule**: More values = better alpha selection but longer training

#### **cv_folds** (Cross-Validation Folds)
- **Small datasets**: 3-5 folds
- **Medium datasets**: 5 folds (default)
- **Large datasets**: 3 folds for speed
- **Rule**: More folds = better validation but slower training

#### **test_size / val_size** (Data Split Ratios)
- **Small datasets**: Reduce test_size to 0.1, val_size to 0.2
- **Standard datasets**: Keep defaults (0.2, 0.25)
- **Large datasets**: Can use smaller fractions for speed

### Performance Tuning

For **fast training** (quick iteration):
```bash
just train-ridge-regression ../../benchmarks/metrics.db training_data -2 2 10 3 0.15 0.2
```

For **thorough optimization** (best model):
```bash
just train-ridge-regression ../../benchmarks/metrics.db training_data -4 4 50 5 0.2 0.25
```

## Troubleshooting

