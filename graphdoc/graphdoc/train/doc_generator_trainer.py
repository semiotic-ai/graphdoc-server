# system packages
import io
import logging
import math
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

    def _calculate_average_score(self, evaluation): 
        examples = evaluation["results"]
        total = 0
        for ex in examples: 
            rating = math.sqrt(ex[2]) * 25
            total += rating
        return round(total / len(examples), 6)

    def _log_evaluation_metrics(self, base_evaluation, optimized_evaluation) -> None:
        base_evaluation_overall_score = self._calculate_average_score(base_evaluation)
        optimized_evaluation_overall_score = self._calculate_average_score(optimized_evaluation)

        mlflow.log_metrics(
            {
                "base_evaluation_overall_score": base_evaluation_overall_score,
                "optimized_evaluation_overall_score": optimized_evaluation_overall_score,
            }
        )
        log.info(f"Base Evaluation: {base_evaluation}")
        log.info(f"Optimized Evaluation: {optimized_evaluation}")
        mlflow.log_dict(base_evaluation, "base_evaluation.json")
        mlflow.log_dict(optimized_evaluation, "optimized_evaluation.json")


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
                prompt=self.get_prompt_signature(base_model) # TODO: double check this is what we want, but i am pretty sure it is
            )  # we could have this be compained with run_trainer to have one function mapped together
        else:
            base_model = self.prompt.infer

        optimized_model = self.run_trainer()
        base_evaluation, optimized_evaluation = self.evaluate_training(
            base_model, optimized_model
        )

        # log the prompts
        base_signature = self.get_prompt_signature(base_model)
        optimized_signature = self.get_prompt_signature(optimized_model)
        base_prompt = self.par.format_signature_prompt(
            signature=base_signature, signature_type="doc_generation"
        )
        optimized_prompt = self.par.format_signature_prompt(
            signature=optimized_signature, signature_type="doc_generation"
        )
        mlflow.log_text(base_prompt, "base_prompt.txt")
        mlflow.log_text(optimized_prompt, "optimized_prompt.txt")

        if save_model and optimized_model:
            self.save_model(optimized_model)

        if self._compare_models(base_evaluation, optimized_evaluation):
                log.info("Model training successful, saving model")
        else:
            log.info("Trained model did not improve on base model")
        return optimized_model
