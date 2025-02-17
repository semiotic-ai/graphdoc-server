# system packages 

# internal packages 
from mlflow_manager import MLFlowManager

# external packages 
import mlflow

class TestMLFlowManager:
    def test_init(self):
        mlflow_manager = MLFlowManager(
            source_tracking_uri="http://localhost:5000",
            target_tracking_uri="http://localhost:5001"
        )
        assert mlflow_manager is not None
