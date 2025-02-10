# system packages
import logging

# internal packages

# external packages
import mlflow

# logging
log = logging.getLogger(__name__)

class FlowLoader:
    def __init__(self, mlflow_tracking_uri: str):
        mlflow_tracking_uri = str(mlflow_tracking_uri)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_client = mlflow.MlflowClient(tracking_uri=str(mlflow_tracking_uri))
        mlflow.set_tracking_uri(str(mlflow_tracking_uri))    

    def load_latest_version(self, model_name: str):
        pass
