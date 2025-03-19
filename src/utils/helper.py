# Standard Library
import os
from collections.abc import MutableMapping
from typing import Union

# Third Party
from omegaconf import DictConfig

# Local Folder
from .logging import get_logger

logger = get_logger(__name__)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten nested dictionary
    Args:
        d (MutableMapping): the source dictionary
        parent_key (str): prefix added to the root keys
        sep (str): separator between keys
    Returns:
        dict: flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = str(parent_key) + sep + str(k) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_credential_env(config: Union[dict, DictConfig]) -> None:
    """Set env var from credential config file.
    config (DictConfig): credential config
    """
    if config is None:
        return

    for k, v in config.items():
        if isinstance(v, MutableMapping):
            set_credential_env(v)
        else:
            os.environ[k] = v
            logger.info(f"Set env {k}=****")