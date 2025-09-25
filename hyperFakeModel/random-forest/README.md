# Random Forest Training

Random forest model training for HyperFake function performance prediction. This module provides ensemble learning with decision trees, offering robust non-linear predictions while handling complex relationships in the data.

## Usage

### Training Models

Train random forest models from a database of collected metrics:

```bash
# Using justfile (from project root)
just train-random-forest ../../benchmarks/metrics.db training_data

Options:
  db_path           Path to SQLite database file (default: ../../benchmarks/metrics.db)
  table             Table name in database (default: training_data)
  n_estimators      Number of trees in forest (default: 100)
  max_depth         Maximum tree depth (default: None)
  min_samples_split Minimum samples to split node (default: 2)
  min_samples_leaf  Minimum samples in leaf (default: 1)
  max_features      Features to consider for splits (default: sqrt)
  random_state      Random seed (default: 42)
  n_jobs           Number of parallel jobs (default: -1)
```

## Parameter Selection

### Key Parameters

Choose optimal parameters based on your data size and performance requirements:

#### **n_estimators** (Number of Trees)
- **Small datasets**: 50-100 trees
- **Medium datasets**: 100-200 trees  
- **Large datasets**: 200-500 trees
- **Rule**: More trees = better accuracy but slower training/prediction

#### **max_depth** (Tree Depth)
- **Default**: None (unlimited depth)
- **Recommendation**: Start with 10-20 for controlled overfitting
- **Small datasets**: 5-10 to prevent overfitting
- **Large datasets**: 15-25 for complex patterns

#### **min_samples_split** (Minimum Samples to Split)
- **Default**: 2
- **Recommendation**: 5-10 for noisy data
- **Higher values**: Prevent overfitting but may underfit

#### **min_samples_leaf** (Minimum Samples in Leaf)
- **Default**: 1  
- **Recommendation**: 2-5 for smoother predictions
- **Higher values**: More regularization, less overfitting

### Performance Tuning

For **fast inference** in fake worker:
```bash
just train-random-forest ../../benchmarks/metrics.db training_data 50 10 5 2
```

For **high accuracy** (slower):
```bash
just train-random-forest ../../benchmarks/metrics.db training_data 200 "" 2 1
```

## Troubleshooting

Random forest models provide excellent accuracy but are computationally expensive. Consider reducing the number of estimators or tree depth if performance becomes an issue in the fake worker.