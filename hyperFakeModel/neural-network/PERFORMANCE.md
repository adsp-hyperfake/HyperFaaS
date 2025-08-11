# Performance Tuning

The [official performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) described three types of tuning, general, CPU specific and GPU specific improvments.

## Already included improvements

- [Disabling gradient calculation for validation or inference](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-gradient-calculation-for-validation-or-inference)

## Further interesting improvements

- [Enabling asynchronous data loading and augmentation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation)
- [Utilizing OpenMP](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#utilize-openmp)
- [Using TZMalloc](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#utilize-openmp)
- [Reducing floating point precision](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-tensor-cores)
- [Using CUDA graphs](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-cuda-graphs)
- [Using cuDNN auto-tuner](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-cuda-graphs)
- [Avoiding unnecessary CPU-GPU synchronisation](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)

## Testing Environment

CPU: 9950X3D \
RAM: 192GB 2666MHZ DDR5 RAM \
GPU: RTX2070 SUPER

original database: 8_997_754 rows \
removed zeros: 1_085_890 rows \
bfs: 2_375_594 rows

We ran the experiments by executing the manual training, utilizing a fixed hyperparameter set and running 10 training epochs for the bfs function.

```json
{
    "hyperparams": {
        "hidden_dims": [
            39,
            21
        ],
        "dropouts": [
            0.03267819348951357,
            0.02272203322672002
        ],
        "lr": 0.004580708150974148,
        "weight_decay": 1.0540556598931587e-06,
        "batch_size": 64,
        "patience": 20,
        "optimizer": "Adam"
    },
    "val_score_optuna": 0.5603743326065578,
    "val_score_final_training_optuna": 0.5593990597609929,
    "val_score_training_full_data": 0.5014,
    "r2_score_training_full_data": 0.4984
}
```

## "Default" Settings (before experiment)

- no compilation
- 0 workers (for data loading)
- 0 OpenMP Threads
- no TCMalloc implementation
- f32 highest precision

## Results

|                  | GPU (jobs=1)  | CPU (jobs=1)  |
|------------------|---------------|---------------|
| none (current)   | 20s/it        | 10s/it        |
| default          | 22s/it        | 10s/it        |
| max-autotune     | 17s/it        | 10s/it        |
| reduce-overhead  | 17s/it        | 10s/it        |
| 1 workers        | 46s/it        | 36s/it        |
| 2 workers        | 43s/it        | 33s/it        |
| 4 workers        | 44s/it        | 33s/it        |
| OpenMP 1 Thread  | not possible  | 7s/it         |
| OpenMP 2 Threads | not possible  | 7s/it         |
| OpenMP 4 Threads | not possible  | 7s/it         |
| TCMalloc         | not possible  | 10s/it        |
| f32 medium precision | 19s/it    | not possible  |
| cudnn benchmark  | 20s/it        | not possible  |

The above results are *before* improving the data loading!
some improvements in data loading (total 4s/it reduction):

- `all_predictions` / `all_targets` lists are now only copied when evaluation finishes => 1s/it reduction
- CustomData loads data into GPU -> 3s/it reduction


## Prerequisites

### OpenMP

OpenMP needs to be installed (included in libgcc). Then set `export OMP_NUM_THREADS=1`

To unset use `unset OMP_NUM_THREADS`

### TCMalloc

TCMalloc is part of googles performance tools. Install with `libgoogle-perftools` or `libgoogle-perftools-dev`.
Then set `export LD_PRELOAD=<path/to/libtcmalloc.so>:$LD_PRELOAD`