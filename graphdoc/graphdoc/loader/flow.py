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
        model_latest_version = self.mlflow_client.get_latest_versions(model_name)        
        return mlflow.dspy.load_model(model_latest_version[0].source)
    
    def load_model_by_uri(self, model_uri: str):
        try:
            return mlflow.dspy.load_model(model_uri)
        # TODO: we should handle this better based on the error
        except Exception as e:
            log.error(f"Error loading model from {model_uri}: {e}")
            raise e
