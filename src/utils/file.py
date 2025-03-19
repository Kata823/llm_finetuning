# Standard Library
import json
import pickle
from pathlib import Path
from typing import Any, Union

# Third Party
import numpy as np
import ruamel.yaml
import yaml

# Local Folder
from .logging import get_logger

logger = get_logger(__name__)


def save_npy(arr: np.ndarray, filepath: Union[str, Path]):
    """Save a NumPy array to a .npy file.

    Args:
        arr (np.ndarray): The NumPy array to save.
        filepath (Union[str, Path]): The file path where the array will be saved.
    """
    with open(filepath, "wb") as f:
        np.save(f, arr)
    logger.info(f"Saved {str(filepath)}")


def load_npy(filepath: Union[str, Path]) -> np.ndarray:
    """Load a NumPy array from a .npy file.

    Args:
        filepath (Union[str, Path]): The file path from which the array will be loaded.

    Returns:
        np.ndarray: The loaded NumPy array.
    """
    with open(filepath, "rb") as f:
        arr = np.load(f)
    logger.info(f"Loaded {str(filepath)}")
    return arr


def save_pickle(obj: Any, filepath: Union[str, Path]):
    """Save an object to a pickle file.

    Args:
        obj (Any): The object to save.
        filepath (Union[str, Path]): The file path where the object will be saved.
    """
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved {str(filepath)}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load an object from a pickle file.

    Args:
        filepath (Union[str, Path]): The file path from which the object will be loaded.

    Returns:
        Any: The loaded object.
    """
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded {str(filepath)}")
    return obj


def save_yaml(data: dict, file_path: str) -> None:
    """Save a dictionary to a YAML file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The file path where the dictionary will be saved.
    """
    yml = ruamel.yaml.YAML()
    yml.indent(mapping=2, sequence=4, offset=2)
    with open(file_path, "w") as f:
        logger.info(f"save {file_path}")
        yml.dump(data, f)


def load_yaml(file_path: str) -> dict:
    """Load a dictionary from a YAML file.

    Args:
        file_path (str): The file path from which the dictionary will be loaded.

    Returns:
        dict: The loaded dictionary.
    """
    with open(file_path) as f:
        logger.info(f"load {file_path}")
        return yaml.safe_load(f)


def save_json(obj: Any, filepath: Union[str, Path]) -> None:
    """Save an object to a JSON file.

    Args:
        obj (Any): The object to save.
        filepath (Union[str, Path]): The file path where the object will be saved.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved to {str(filepath)}")


def load_json(filepath: Union[str, Path]) -> Any:
    """Load an object from a JSON file.

    Args:
        filepath (Union[str, Path]): The file path from which the object will be loaded.

    Returns:
        Any: The loaded object.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        obj = json.load(f)
    logger.info(f"Loaded {str(filepath)}")
    return obj


def load_list_json(filepath: Union[str, Path]) -> list[Any]:
    obj = [
        json.loads(line)
        for line in open(
            filepath,
            "r",
            encoding="utf-8",
        )
    ]
    logger.info(f"Loaded {str(filepath)}")
    return obj