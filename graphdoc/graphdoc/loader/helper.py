# system packages
import logging
import os
import yaml
from yaml import SafeLoader
from pathlib import Path
from typing import Literal, Optional, Union

# internal packages

# external packages
import mlflow

# logging
log = logging.getLogger(__name__)


def check_directory_path(directory_path: Union[str, Path]) -> None:
    _directory_path = Path(directory_path).resolve()
    if not _directory_path.is_dir():
        raise ValueError(
            f"The provided path does not resolve to a valid directory: {directory_path}"
        )


def check_file_path(file_path: Union[str, Path]) -> None:
    _file_path = Path(file_path).resolve()
    if not _file_path.is_file():
        raise ValueError(
            f"The provided path does not resolve to a valid file: {file_path}"
        )


def _env_constructor(loader, node):
    value = loader.construct_scalar(node)
    env_value = os.getenv(value)
    if env_value is None:
        raise ValueError(f"Environment variable '{value}' is not set.")
    return env_value


def load_yaml_config(file_path: Union[str, Path], use_env: bool = True) -> dict:
    if use_env:
        SafeLoader.add_constructor("!env", _env_constructor)

    _file_path = Path(file_path).resolve()
    if not _file_path.is_file():
        raise ValueError(
            f"The provided path does not resolve to a valid file: {file_path}"
        )
    with open(_file_path, "r") as file:
        return yaml.load(file, Loader=SafeLoader)


def setup_logging(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    log_file_path: Optional[str] = "logs/app.log",
):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

def load_dspy_model(
    model_name: str,
    latest_version: bool = True,
    version: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
):
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow_client = mlflow.MlflowClient()
    try:
        if latest_version:
            model_latest_version = mlflow_client.get_latest_versions(model_name)
            log.info(
                f"Loaded latest version of {model_name} at {model_latest_version[0].source}"
            )
            return mlflow.dspy.load_model(model_latest_version[0].source)
        else:
            log.info(f"Tried to loading dspy model {model_name} with version {version}")
            log.warning(
                "Unsupported loading of dspy model by version number. Please use latest_version=True"
            )
    except Exception as e:
        log.error(f"Error loading dspy model {model_name}: {e}")
        raise e
