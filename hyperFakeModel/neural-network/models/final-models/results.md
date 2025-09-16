## All Data Results

| Scenario | Metric | bfs-json | echo | thumbnailer-json |
| --- | --- | --- | --- | --- |
| Full Columns | Test Loss | 0.4016 | 0.8374 | 0.6944 |
|  | R² Score | 0.5993 | 0.1576 | 0.3064 |
| Leave-one-out: request_size_bytes | Test Loss | 0.4052 | 0.8386 | 0.4562 |
|  | R² Score | 0.5956 | 0.1563 | 0.5416 |
| Leave-one-out: function_instances_count | Test Loss | 0.4140 | 0.8604 | 0.4526 |
|  | R² Score | 0.5868 | 0.1343 | 0.5454 |
| Leave-one-out: active_function_calls_count | Test Loss | 0.5162 | 0.8902 | 0.5130 |
|  | R² Score | 0.4848 | 0.1044 | 0.4847 |
| Leave-one-out: worker_cpu_usage | Test Loss | 0.5479 | 0.8925 | 0.5595 |
|  | R² Score | 0.4529 | 0.1029 | 0.4380 |
| Leave-one-out: worker_ram_usage | Test Loss | 0.4678 | 0.8960 | 0.4864 |
|  | R² Score | 0.5331 | 0.0990 | 0.5114 |

## Averaged 1-to-n Results

| Scenario | Metric | bfs-json | echo | thumbnailer-json |
| --- | --- | --- | --- | --- |
| Full Columns | Test Loss | 0.3479 | 0.5363 | 0.2395 |
|  | R² Score | 0.6630 | 0.4729 | 0.7600 |
| Leave-one-out: request_size_bytes | Test Loss | 0.3488 | 0.5405 | 0.2681 |
|  | R² Score | 0.6621 | 0.4687 | 0.7314 |
| Leave-one-out: function_instances_count | Test Loss | 0.3692 | 0.6801 | 0.2772 |
|  | R² Score | 0.6431 | 0.3293 | 0.7222 |
| Leave-one-out: active_function_calls_count | Test Loss | 0.4412 | 0.6067 | 0.3112 |
|  | R² Score | 0.5726 | 0.4038 | 0.6881 |
| Leave-one-out: worker_cpu_usage | Test Loss | 0.4506 | 0.6518 | 0.3656 |
|  | R² Score | 0.5624 | 0.3588 | 0.6336 |
| Leave-one-out: worker_ram_usage | Test Loss | 0.3852 | 0.6145 | 0.2868 |
|  | R² Score | 0.6267 | 0.3959 | 0.7126 |