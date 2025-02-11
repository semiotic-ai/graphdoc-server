# system packages
import ast
import logging

# internal packages
import dspy

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

    def get_prompt_type(self, prompt): 
        if isinstance(prompt, dspy.Predict):
            return "predict"
        elif isinstance(prompt, dspy.ChainOfThought):
            return "chain_of_thought"
        else:
            raise ValueError(f"Not a registered prompt type: {type(prompt)}")

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

    def run_parameters(self, run_id: str):
        run = self.mlflow_client.get_run(run_id)
        
        # go through and convert the nested dictionaries to actual dictionaries (as they are currently strings)
        for key, value in run.data.params.items():
            run.data.params[key] = ast.literal_eval(value)
        return run.data.params
    
    @staticmethod
    def get_prompt_signature(prompt) -> dspy.Signature:
        if isinstance(prompt, dspy.ChainOfThought):
            return prompt.predict.signature
        elif isinstance(prompt, dspy.Predict):
            return prompt.signature
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
