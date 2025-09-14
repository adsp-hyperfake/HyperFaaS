#!/usr/bin/python

import click
import json
import os
from functools import wraps
from neural_network import optuna_pipeline, manual_pipeline


def shared_training_options(func):
    @click.option(
        "--cpu",
        is_flag=True,
        default=False,
        show_default=True,
        help="Use CPU for training instead of GPU",
    )
    @click.option(
        "--dbs-dir",
        type=click.Path(exists=True),
        default=os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "training_dbs")
        ),
        show_default=True,
        help="Path to directory containing database files",
    )
    @click.option(
        "--export-dir",
        type=click.Path(exists=True),
        default=os.path.join(os.path.dirname(__file__), "models"),
        show_default=True,
        help="Target directory for saving the trained models",
    )
    @click.option(
        "--func-tag",
        multiple=True,
        default=(
            "hyperfaas-bfs-json:latest",
            "hyperfaas-thumbnailer-json:latest",
            "hyperfaas-echo:latest",
        ),
        show_default=True,
        help="Function tags to train (can be specified multiple times)",
    )
    @click.option(
        "--short-name",
        multiple=True,
        default=("bfs-json", "thumbnailer-json", "echo"),
        show_default=True,
        help="Short names corresponding to function tags",
    )
    @click.option(
        "--table-name",
        default="training_data_avg",
        show_default=True,
        help="The db's table name containing the training data",
    )
    @click.option(
        "--sample-data",
        is_flag=True,
        default=False,
        help="Use sample data instead of real data",
    )
    @click.option(
        "--input-cols",
        multiple=True,
        show_default=True,
        default=(
            "request_size_bytes",
            "function_instances_count",
            "active_function_calls_count",
            "worker_cpu_usage",
            "worker_ram_usage",
        ),
        help="Input columns for training (can be specified multiple times)",
    )
    @click.option(
        "--output-cols",
        multiple=True,
        show_default=True,
        default=(
            "function_processing_time_ns",
            "function_cpu_usage",
            "function_ram_usage",
        ),
        help="Output columns for training (can be specified multiple times)"
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    pass


@cli.command()
@shared_training_options
@click.option(
    "--trials",
    default=20,
    type=click.IntRange(1, None),
    show_default=True,
    help="Number of trials",
)
@click.option(
    "--jobs",
    default=5,
    type=click.IntRange(-1, None),
    show_default=True,
    help="Number of parallel jobs (-1 for # of CPUs)",
)
@click.option(
    "--epochs",
    default=50,
    type=click.IntRange(1, None),
    show_default=True,
    help="Number of training epochs",
)
@click.option(
    "--final-epochs",
    default=100,
    type=click.IntRange(0, None),
    show_default=True,
    help="Number of training epochs for the final exported model",
)
@click.option(
    "--samples",
    default=-1,
    type=click.IntRange(-1, None),
    help="Number of samples to train on [-1 uses all data]",
)
@click.option(
    "--save-state",
    is_flag=True,
    default=False,
    show_default=True,
    help="Save the state of the study to a db",
)
@click.option(
    "--shared-db",
    type=click.Path(),
    help="Optional path to a db saving the state of the study",
)
@click.option(
    "--study-id",
    help="Unique ID of the study",
)
def optuna(
    cpu: bool,
    dbs_dir,
    export_dir,
    func_tag,
    short_name,
    table_name,
    sample_data,
    trials,
    jobs,
    epochs,
    final_epochs,
    samples,
    save_state,
    shared_db,
    study_id,
    input_cols,
    output_cols,
):
    if len(func_tag) != len(short_name):
        raise click.ClickException(
            "Number of function tags must match number of short names"
        )
    assert jobs != 0, "Number of parallel jobs must not be 0"
    click.echo(f"Trials: {trials}, Jobs: {jobs}")
    optuna_pipeline(
        cpu,
        func_tag,
        short_name,
        dbs_dir,
        table_name,
        sample_data,
        export_dir,
        trials,
        jobs,
        epochs,
        final_epochs,
        samples,
        save_state,
        shared_db,
        study_id,
        list(input_cols),
        list(output_cols),
    )


@cli.command()
@shared_training_options
@click.option(
    "--hyperparams",
    type=click.Path(),
    help="Path to hyperparameters json file",
)
@click.option(
    "--epochs",
    default=100,
    type=click.IntRange(1, None),
    show_default=True,
    help="Number of training epochs",
)
@click.option(
    "--samples",
    default=-1,
    type=click.IntRange(-1, None),
    help="Number of samples to train on [-1 uses all data]",
)
def manual(
    cpu,
    dbs_dir,
    export_dir,
    func_tag,
    short_name,
    table_name,
    sample_data,
    hyperparams,
    epochs,
    samples,
    input_cols,
    output_cols,
):
    if len(func_tag) != len(short_name):
        raise click.ClickException(
            "Number of function tags must match number of short names"
        )
    click.echo("Starting manual mode...")
    click.echo(f"Database path: {dbs_dir}")
    click.echo(f"Target path: {export_dir}")
    hyperparameters = None
    if hyperparams:
        hyperparameters = _read_and_validate_hyperparams(hyperparams)
    manual_pipeline(
        cpu,
        func_tag,
        short_name,
        dbs_dir,
        table_name,
        sample_data,
        export_dir,
        epochs,
        hyperparameters,
        samples,
        list(input_cols),
        list(output_cols),
    )


def _read_and_validate_hyperparams(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    hyperparams = data["hyperparams"]
    required_keys = [
        "hidden_dims",
        "dropouts",
        "lr",
        "weight_decay",
        "batch_size",
        "patience",
        "optimizer",
        "gradient_clipping",
    ]
    for key in required_keys:
        assert key in hyperparams, f"Missing key: {key}"

    # Check hiddem_dims and dropouts are lists of equal length
    assert isinstance(hyperparams["hidden_dims"], list), "hidden_dims should be a list"
    assert isinstance(hyperparams["dropouts"], list), "dropouts should be a list"
    assert len(hyperparams["hidden_dims"]) == len(hyperparams["dropouts"]), (
        "hidden_dims and dropouts must have the same length"
    )

    return hyperparams


if __name__ == "__main__":
    cli()
