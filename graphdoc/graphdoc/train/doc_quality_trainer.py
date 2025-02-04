# system packages
import io
import logging
from typing import Any, Dict, List, Tuple

# internal packages
from .single_prompt_trainer import SinglePromptTrainerRunner
from ..prompts import DocQualityPrompt

# external packages
import dspy
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models import ModelSignature

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DocQualityTrainer(SinglePromptTrainerRunner):
    def __init__(
        self,
        prompt: DocQualityPrompt,
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
        example.pop("category")
        example.pop("rating")
        return infer_signature(example)

    def _log_evaluation_metrics(self, base_evaluation, optimized_evaluation) -> None:
        base_evaluation_overall_score = base_evaluation["overall_score"]
        optimized_evaluation_overall_score = optimized_evaluation["overall_score"]

        mlflow.log_metrics(
            {
                "base_evaluation_overall_score": base_evaluation_overall_score,
                "optimized_evaluation_overall_score": optimized_evaluation_overall_score,
            }
        )

        metrics_data = {
            "Evaluation Type": ["Base Evaluation", "Optimized Evaluation"],
            "Overall Score": [
                base_evaluation_overall_score,
                optimized_evaluation_overall_score,
            ],
        }

        for key, value in base_evaluation["per_category_scores"].items():
            metrics_data[f"{key} Percent Correct"] = [
                value["percent_correct"],
                optimized_evaluation["per_category_scores"][key]["percent_correct"],
            ]
            metrics_data[f"{key} Total"] = [
                value["total"],
                optimized_evaluation["per_category_scores"][key]["total"],
            ]
            metrics_data[f"{key} Correct"] = [
                value["correct"],
                optimized_evaluation["per_category_scores"][key]["correct"],
            ]

        df = pd.DataFrame(metrics_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "evaluation_comparison.csv")

    def evaluate_training(
        self, base_model, optimized_model
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        log.info(f"base_evaluation: {base_evaluation}")
        log.info(f"optimized_evaluation: {optimized_evaluation}")
        self._log_evaluation_metrics(base_evaluation, optimized_evaluation)
        return base_evaluation, optimized_evaluation

    def run_training(self, load_model: bool = True, save_model: bool = True):
        if load_model:
            log.info("Loading model from mlflow")
            base_model = self.load_model()
            self.prompt = DocQualityPrompt(
                type=self.prompt.type,  # type: ignore
                metric_type=self.prompt.metric_type,  # type: ignore
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
            signature=base_signature, signature_type="doc_quality"
        )
        optimized_prompt = self.par.format_signature_prompt(
            signature=optimized_signature, signature_type="doc_quality"
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
