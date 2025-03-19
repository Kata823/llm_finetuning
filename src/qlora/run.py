# Third Party
import hydra
from omegaconf import DictConfig

# Local Folder
from src.qlora.trainer import Trainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="run")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    if cfg["train_flg"]:
        logger.info(f"Start Training")
        trainer.train()
    if cfg["eval_flg"]:
        logger.info(f"Start Evaluation")
        trainer.evaluate(cfg)
    if cfg["test_flg"]:
        logger.info(f"Start Testing")
        trainer.predict(cfg)

if __name__ == "__main__":
    main()