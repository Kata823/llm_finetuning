# Standard Library
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

# Third Party
import mlflow
import polars as pl
from omegaconf import DictConfig, OmegaConf

# Local Folder
from ..utils import file, helper, logging
from ..utils.mlflow import init_mlflow

logger = logging.get_logger(__name__)


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, cfg: DictConfig):
        """Initialize the BaseTrainer with a configuration.

        Args:
            cfg (DictConfig): The configuration object.
        """
        self.cfg = cfg
        self._setup_mlflow()

    def fit_test(self):
        """Fit the model and test it."""
        self.fit()
        self.test()

    def fit(self): ...

    def test(self): ...

    def _setup_mlflow(self):
        """Setup MLflow session."""
        if run := mlflow.active_run():
            logger.warning(f"An active mlflow run detected: {run.info.run_id}")
            return

        self.experiment_id = init_mlflow(self.cfg)
        run_name = self.cfg["logger"]["run_name"]
        exp_name = self.cfg["logger"]["experiment_name"]
        self.run_id = mlflow.start_run(
            run_name=run_name, experiment_id=self.experiment_id
        ).info.run_id
        logger.info(
            f"Starting MLflow run: [{self.run_id}]{run_name} in experiment: [{self.experiment_id}]{exp_name}"
        )

        # Log tags and params
        mlflow.set_tags(self.cfg["logger"]["tags"])
        mlflow.log_params(
            helper.flatten_dict(OmegaConf.to_container(self.cfg, resolve=True))
        )

    def _close_mlflow(self):
        """Close the MLflow session."""
        mlflow.end_run()

    def __del__(self):
        self._close_mlflow()