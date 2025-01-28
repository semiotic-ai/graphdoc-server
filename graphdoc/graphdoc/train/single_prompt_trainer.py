# system packages
import logging
from typing import List, Optional
from abc import ABC, abstractmethod

# internal packages
from ..prompts import SinglePrompt, DocQualityPrompt

# external packages
import dspy
import mlflow

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class SinglePromptTrainerRunner(ABC):

    def __init__(
        self,
        prompt: SinglePrompt,
        optimizer_type: str,
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        self.prompt = prompt
        self.optimizer_type = optimizer_type
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_experiment_name = mlflow_experiment_name
        self.trainset = trainset
        self.evalset = evalset

        # mlflow related
        mlflow.dspy.autolog()
        mlflow.set_experiment(self.mlflow_experiment_name)
        self.mlflow_client = mlflow.MlflowClient()

    # mlflow related methods
    @abstractmethod
    def get_signature(self):
        pass

    @abstractmethod
    def get_prompt_signature(self, prompt):
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
            print(f"compiling model (type: {type(self.prompt.infer)}): {self.prompt.infer}")
            print(f"trainset type: {type(self.trainset)}")
            optimized_model = optimizer.compile(
                self.prompt.infer,
                trainset=self.trainset,
                max_labeled_demos=0,
                max_bootstrapped_demos=0,
            )
            return optimized_model

    @abstractmethod
    def evaluate_training(
        self, base_model, optimized_model, type: str, metric_type: str
    ):
        pass
        # # TODO: we should type this better
        # base_prompt = SinglePrompt(
        #     prompt=base_model,
        #     type=self.prompt.type,  # type: ignore
        #     metric_type=self.prompt.metric_type,  # type: ignore
        # )
        # optimized_prompt = SinglePrompt(
        #     prompt=optimized_model,
        #     type=self.prompt.type,  # type: ignore
        #     metric_type=self.prompt.metric_type,  # type: ignore
        # )
        # base_evaluation = base_prompt.evaluate_evalset(self.evalset)
        # optimized_evaluation = optimized_prompt.evaluate_evalset(self.evalset)
        # return base_evaluation, optimized_evaluation

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
        # if load_model:
        #     base_model = self.load_model()
        #     self.prompt = SinglePrompt(
        #         prompt=base_model,
        #         type=self.prompt.type,
        #         metric_type=self.prompt.metric_type,
        #     )
        # else:
        #     base_model = self.prompt.infer
        # optimized_model = self.run_trainer()
        # self.save_model(optimized_model)
        # base_evaluation, optimized_evaluation = self.evaluate_training(
        #     base_model, optimized_model
        # )
        # if self._compare_models(base_evaluation, optimized_evaluation):
        #     self.save_model(optimized_model)
        #     log.info("Model training successful, saving model")
