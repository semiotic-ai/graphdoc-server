# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc.loader.helper import load_dspy_model

# external packages
import dspy

# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent

class TestLoader:
    def test_pass(self):
        assert True

    def test_load_dspy_model_by_version(self):
        mlflow_tracking_uri = Path(BASE_DIR) / "assets" / "mlruns"
        log.info(f"Loading model from {mlflow_tracking_uri}")
        mlflow_model = load_dspy_model(
            model_name="doc_generator_model",
            mlflow_tracking_uri=mlflow_tracking_uri,
            latest_version=True,
        )
        assert mlflow_model is not None
        assert isinstance(mlflow_model, dspy.ChainOfThought)
