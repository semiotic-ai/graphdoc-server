# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc.loader.helper import load_dspy_model
from graphdoc import FlowLoader

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

    def test_load_flow_latest_version(self, fl: FlowLoader):
        flow = fl.load_latest_version(model_name="doc_generator_model")
        assert flow is not None
        assert isinstance(flow, dspy.ChainOfThought)

    def test_load_model_by_uri(self, fl: FlowLoader):
        model_uri = "file:///Users/denver/Documents/code/graph/graphdoc/mlruns/513408250948216117/976d330558344c41b30bd1531571de18/artifacts/model"
        flow = fl.load_model_by_uri(model_uri)
        assert flow is not None
        assert isinstance(flow, dspy.ChainOfThought)
