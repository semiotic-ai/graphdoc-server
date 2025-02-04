# system packages
import io
import logging
from typing import Any, Dict, List, Tuple

# internal packages
from .single_prompt_trainer import SinglePromptTrainerRunner
from ..prompts import DocGeneratorPrompt

# external packages
import dspy
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models import ModelSignature

# logging
log = logging.getLogger(__name__)


class DocGeneratorTrainer(SinglePromptTrainerRunner):
    def __init__(
        self,
        prompt: DocGeneratorPrompt,
        optimizer_type: str,
        optimizer_kwargs: Dict[str, Any],    
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        mlflow_tracking_uri: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        super().__init__(
            prompt=prompt,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            mlflow_model_name=mlflow_model_name,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            trainset=trainset,
            evalset=evalset,
        )

    def get_signature(self) -> ModelSignature:
        example = self.trainset[0].toDict()
        example.pop("documented_schema")
        return infer_signature(example)

    # def get_prompt_signature(self, prompt) -> dspy.Signature:
    #     pass

    def _log_evaluation_metrics(self, base_evaluation, optimized_evaluation) -> None:
        base_evaluation_overall_score = base_evaluation["overall_score"]
        optimized_evaluation_overall_score = optimized_evaluation["overall_score"]

        mlflow.log_metrics(
            {
                "base_evaluation_overall_score": base_evaluation_overall_score,
                "optimized_evaluation_overall_score": optimized_evaluation_overall_score,
            }
        )
        log.info(f"Base Evaluation: {base_evaluation}")
        log.info(f"Optimized Evaluation: {optimized_evaluation}")

    def evaluate_training(
        self, base_model, optimized_model
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        base_prompt = DocGeneratorPrompt(
            prompt=self.get_prompt_signature(base_model),
            type=self.prompt.type,  # type: ignore
            metric_type=self.prompt.metric_type,  # type: ignore
        )
        optimized_prompt = DocGeneratorPrompt(
            prompt=self.get_prompt_signature(optimized_model),
            type=self.prompt.type,  # type: ignore
            metric_type=self.prompt.metric_type,  # type: ignore
        )
        base_evaluation = base_prompt.evaluate_evalset(self.evalset)
        optimized_evaluation = optimized_prompt.evaluate_evalset(self.evalset)

        log.info(f"base_evaluation: {base_evaluation}")
        log.info(f"optimized_evaluation: {optimized_evaluation}")
        self._log_evaluation_metrics(base_evaluation, optimized_evaluation)
        return base_evaluation, optimized_evaluation

    def run_training(self, load_model: bool = True, save_model: bool = True):
        if load_model:
            log.info("Loading model from mlflow")
            base_model = self.load_model()
            self.prompt = DocGeneratorPrompt(
                type=self.prompt.type,  # type: ignore
                metric_type=self.prompt.metric_type,  # type: ignore
            )  # we could have this be compained with run_trainer to have one function mapped together
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
