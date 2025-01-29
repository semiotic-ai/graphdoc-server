# system packages
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

# internal packages
from ..prompts import SinglePrompt, DocQualityPrompt

# external packages
import dspy
import mlflow
from mlflow.models import ModelSignature

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class SinglePromptTrainerRunner(ABC):

    def __init__(
        self,
        prompt: SinglePrompt,
        optimizer_type: str,
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        mlflow_tracking_uri: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        self.prompt = prompt
        self.optimizer_type = optimizer_type
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_experiment_name = mlflow_experiment_name
        self.trainset = trainset
        self.evalset = evalset

        # mlflow related
        log.info(f"Setting MLFlow tracking URI to {self.mlflow_tracking_uri}")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.dspy.autolog()
        log.info(f"Setting MLFlow experiment to {self.mlflow_experiment_name}")
        mlflow.set_experiment(self.mlflow_experiment_name)
        self.mlflow_client = mlflow.MlflowClient()

    # mlflow related methods
    @abstractmethod
    def get_signature(self) -> ModelSignature:
        pass

    @abstractmethod
    def get_prompt_signature(self, prompt) -> dspy.Signature:
        pass

    # TODO: we should update this to enable a remote model to be loaded
    def load_model(self):
        try:
            latest_version = self.mlflow_client.get_latest_versions(
                self.mlflow_model_name
            )
            loaded_model = mlflow.dspy.load_model(latest_version[0].source)
            return loaded_model
        except Exception as e:
            print(f"No model found in MLFlow, creating a new one")
            mlflow.dspy.log_model(
                dspy_model=self.prompt.infer,
                artifact_path="model",
                signature=self.get_signature(),
                task=None,
                registered_model_name=self.mlflow_model_name,
            )  # TODO: add metadata related to trainset and evalset
            return self.prompt.infer

    def save_model(self, model: dspy.Signature):
        mlflow.dspy.log_model(
            dspy_model=model,
            artifact_path="model",
            signature=self.get_signature(),
            task=None,
            registered_model_name=self.mlflow_model_name,
        )  # TODO: add metadata related to trainset and evalset

    # trainer related methods
    def initialize_trainer(self, optimizer_type: Optional[str] = None):
        if optimizer_type is None:
            optimizer_type = self.optimizer_type
        if optimizer_type == "miprov2":
            return dspy.MIPROv2(metric=self.prompt.evaluate_metric, auto="light")
        else:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    def run_trainer(self, optimizer_type: Optional[str] = None):
        if optimizer_type is None:
            optimizer_type = self.optimizer_type
        optimizer = self.initialize_trainer(optimizer_type)
        if optimizer_type == "miprov2":
            print(
                f"compiling model (type: {type(self.prompt.infer)}): {self.prompt.infer}"
            )
            print(f"trainset type: {type(self.trainset)}")
            optimized_model = optimizer.compile(
                self.prompt.infer,
                trainset=self.trainset,
                max_labeled_demos=0,
                max_bootstrapped_demos=0,
            )
            return optimized_model

    @abstractmethod
    def _log_evaluation_metrics(self, base_evaluation, optimized_evaluation):
        pass
    
    @abstractmethod
    def evaluate_training(self, base_model, optimized_model) -> Tuple[float, float]:
        pass

    def _compare_models(
        self,
        base_evaluation,
        optimized_evaluation,
        comparison_value: str = "overall_score",
    ) -> bool:
        """Compare the metrics of the base and optimized models

        returns true if the optimized model is better than the base model
        """
        return self.prompt._compare_metrics(
            base_evaluation, optimized_evaluation, comparison_value
        )

    # run methods
    @abstractmethod
    def run_training(self, load_model: bool = True, save_model: bool = True):
        pass
