# Standard Library
import os
import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable

# Third Party
import mlflow
import pandas as pd
from omegaconf import DictConfig

# Local Folder
from .gcp import access_secret_version
from .helper import flatten_dict, set_credential_env
from .logging import get_logger

logger = get_logger(__name__)


def mlflow_run(fn: Callable) -> Callable:
    """
    A decorator of run which logs configs before run and metrics return by the run.
    If mlflow is set as the logger, it will use mlflow,
    or, metrics will be logged to local output dir.
    """

    @wraps(fn)
    def wrapper(cfg: DictConfig, *args, **kwargs):
        if "logger" in cfg:
            uri = cfg.logger.tracking_uri or cfg.logger.save_dir
            exp_name = cfg.logger.experiment_name
            logger.info("Logging with mlflow")
            logger.info(f"Set mlflow tracking uri: {uri}")
            logger.info(f"Set experiment: {exp_name}")

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(exp_name)

            # Start a run
            nested = mlflow.active_run() is not None
            with mlflow.start_run(
                run_name=cfg.logger.run_name,
                nested=nested,
            ):
                if tags := cfg.logger.tags:
                    mlflow.set_tags(tags)

                mlflow.log_params(flatten_dict(cfg))
                metrics = fn(cfg, *args, **kwargs)

                if metrics is not None and len(metrics) > 0:
                    mlflow.log_metrics(metrics)

                if (output_dir := cfg.paths.get("output_dir")) and os.path.exists(
                    output_dir
                ):  # noqa: W503
                    mlflow.log_artifacts(output_dir)
        else:
            metrics = fn(cfg, *args, **kwargs)
            metrics_path = cfg.paths.get(
                "metrics_path", os.path.join(cfg.paths.output_dir, "metrics.log")
            )
            log_metrics(metrics=metrics, path=metrics_path)

    return wrapper


def log_metrics(metrics: dict[str, float], path: str) -> None:
    """
    Log metrics to local file.

    Args:
        metrics (dict): Metrics of experiment.
        path (str): File path to save the metrics.
    """
    with open(path, "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")


def log_artifacts(artifacts: dict[str, Any], dir_path: str = ".") -> None:
    # Log artifacts
    for artifact_name, artifact in artifacts.items():
        if re.search("confusion_matrix", artifact_name):
            mlflow.log_figure(artifact, Path(dir_path) / f"{artifact_name}.png")
        elif isinstance(artifact, pd.DataFrame):
            csv_path = Path(dir_path) / "outputs" / f"{artifact_name}.csv"
            artifact.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)


def init_mlflow(cfg: DictConfig):
    """Set up uri and authentication to mlflow.
    Args:
        cfg (DictConfig): Configs include infomation to access mlflow secrets.
    """
    mlflow_creds = {
        "MLFLOW_TRACKING_USERNAME": access_secret_version(
            project_id=cfg["gcp"]["project_id"],
            secret_id=cfg["gcp"]["mlflow_secrets"]["username"],
            version_id="latest",
        ),
        "MLFLOW_TRACKING_PASSWORD": access_secret_version(
            project_id=cfg["gcp"]["project_id"],
            secret_id=cfg["gcp"]["mlflow_secrets"]["password"],
            version_id="latest",
        ),
    }

    set_credential_env(mlflow_creds)
    mlflow.set_tracking_uri(cfg["logger"]["tracking_uri"])
    experiment_name = cfg["logger"]["experiment_name"]
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def get_experiment_id(experiment_name: str) -> str:
    """Get experiment ID.
    Args:
        experiment_name (str): Target experiment name.
    Returns:
        str: MLflow experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def get_mlflow_run_id(cfg: dict, run_id: str, nested_run_name: str) -> tuple[str, str]:
    """Create MLflow experiment along with run ID or nested run ID if run ID is specified.
    Args:
        cfg (dict): Hydra main config.
        run_id (str): Parent run ID. If empty, the component will create a new run ID.
        nested_run_name (str): Nested run name. Will be referenced only if run_id is not empty.
    Returns:
        run_id (str): Run ID.
        nested_run_id (str): Nested run ID.
    """
    # Set experiment
    experiment_id = get_experiment_id(experiment_name=cfg["logger"]["experiment_name"])

    # Get run ID
    nested_run_id = ""
    if run_id == "":
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_id=run_id,
            run_name=cfg["logger"]["run_name"],
        ) as run:
            run_id = run.info.run_id
            mlflow.set_tags(cfg["logger"]["tags"])
    else:
        with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=nested_run_name,
                nested=True,
            ) as sub_run:
                nested_run_id = sub_run.info.run_id
                mlflow.set_tags(cfg["logger"]["tags"])

    return run_id, nested_run_id