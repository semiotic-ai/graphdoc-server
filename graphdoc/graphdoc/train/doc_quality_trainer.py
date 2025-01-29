# system packages
import logging
from typing import List, Tuple

# internal packages
from .single_prompt_trainer import SinglePromptTrainerRunner
from ..prompts import DocQualityPrompt

# external packages
import dspy
import mlflow
from mlflow.models import infer_signature
from mlflow.models import ModelSignature

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DocQualityTrainer(SinglePromptTrainerRunner):
    def __init__(
        self,
        prompt: DocQualityPrompt,
        optimizer_type: str,
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        mlflow_tracking_uri: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        super().__init__(
            prompt=prompt,
            optimizer_type=optimizer_type,
            mlflow_model_name=mlflow_model_name,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            trainset=trainset,
            evalset=evalset,
        )

    def get_signature(self) -> ModelSignature:
        example = self.trainset[0].toDict()
        example.pop("category")
        example.pop("rating")
        return infer_signature(example)

    def get_prompt_signature(self, prompt) -> dspy.Signature:
        if isinstance(prompt, dspy.ChainOfThought):
            return prompt.predict.signature
        elif isinstance(prompt, dspy.Predict):
            return prompt.signature
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def evaluate_training(self, base_model, optimized_model) -> Tuple[float, float]:
        print(f"eval training base_model (type: {type(base_model)}): {base_model}")
        print(
            f"eval training optimized_model (type: {type(optimized_model)}): {optimized_model}"
        )
        base_prompt = DocQualityPrompt(
            prompt=self.get_prompt_signature(base_model),
            type=self.prompt.type,  # type: ignore
            metric_type=self.prompt.metric_type,  # type: ignore
        )
        optimized_prompt = DocQualityPrompt(
            prompt=self.get_prompt_signature(optimized_model),
            type=self.prompt.type,  # type: ignore
            metric_type=self.prompt.metric_type,  # type: ignore
        )
        base_evaluation = base_prompt.evaluate_evalset(self.evalset)
        optimized_evaluation = optimized_prompt.evaluate_evalset(self.evalset)
        return base_evaluation, optimized_evaluation

    def run_training(self, load_model: bool = True, save_model: bool = True):
        if load_model:
            base_model = self.load_model()
            self.prompt = DocQualityPrompt(
                type=self.prompt.type,  # type: ignore
                metric_type=self.prompt.metric_type,  # type: ignore
            )
        else:
            base_model = self.prompt.infer
        optimized_model = self.run_trainer()
        base_evaluation, optimized_evaluation = self.evaluate_training(
            base_model, optimized_model
        )
        if self._compare_models(base_evaluation, optimized_evaluation):
            if save_model and optimized_model:
                self.save_model(optimized_model)
                log.info("Model training successful, saving model")
            return optimized_model
        else:
            log.info("Trained model did not improve on base model")
            return optimized_model
