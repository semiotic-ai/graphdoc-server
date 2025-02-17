# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import FlowLoader
from mlflow import MlflowClient

# external packages
import dspy
# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MLFLOW_DIR = Path(BASE_DIR) / "graphdoc" / "tests" / "assets" / "mlruns"

class TestFlowLoader:
    def test_init_flow_loader(self):
        fl = FlowLoader(mlflow_tracking_uri=MLFLOW_DIR)
        log.info(f"Using MLflow tracking URI: {MLFLOW_DIR}")
        assert fl is not None
        assert isinstance(fl, FlowLoader)   

    def test_load_model_by_name_and_version(self):
        log.info("Starting test_load_model_by_name_and_version")
        fl = FlowLoader(mlflow_tracking_uri=MLFLOW_DIR)
        log.info(f"FlowLoader initialized with MLflow tracking URI: {MLFLOW_DIR}")

        client = MlflowClient(tracking_uri=str(MLFLOW_DIR))
        log.info(f"MLflow directory exists: {MLFLOW_DIR.exists()}")
        
        models_dir = MLFLOW_DIR / "models"
        log.info(f"Models directory exists: {models_dir.exists()}")
        if models_dir.exists():
            log.info(f"Models directory contents: {list(models_dir.iterdir())}")
            
        registered_models = client.search_registered_models()
        log.info(f"Found {len(registered_models)} registered models")
        
        for model in registered_models:
            log.info(f"Found registered model: {model.name}")
            versions = client.search_model_versions(f"name='{model.name}'")
            log.info(f"Found {len(list(versions))} versions for model {model.name}")
            for version in versions:
                log.info(f"  Version {version.version} (Stage: {version.current_stage})")

        model = fl.load_model_by_name_and_version(model_name="doc_generator_model", model_version="1")
        log.info("Successfully loaded doc_generator_model version 1")
        assert model is not None
        assert isinstance(model, dspy.ChainOfThought)

        model = fl.load_model_by_name_and_version(model_name="doc_quality_model", model_version="1")
        log.info("Successfully loaded doc_quality_model version 1")
        assert model is not None
        assert isinstance(model, dspy.ChainOfThought)