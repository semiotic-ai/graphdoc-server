# system packages
from typing import Any, Dict, List

# internal packages
from .single_prompt_trainer import SinglePromptTrainerRunner
from .doc_quality_trainer import DocQualityTrainer
from .doc_generator_trainer import DocGeneratorTrainer
from ..prompts import SinglePrompt

# external packages
import dspy

# optimizer:
#   optimizer_type: miprov2
#   # metric: this is set in the prompt
#   auto: light # miprov2 setting
#   # student: this is the prompt.infer object
#   # trainset: this is the dataset we are working with
#   max_labeled_demos: 0
#   max_bootstrapped_demos: 4


class TrainerFactory:
    @staticmethod
    def get_single_prompt_trainer(
        trainer_class: str,
        prompt: SinglePrompt,
        optimizer_type: str,
        optimizer_kwargs: Dict[str, Any],
        mlflow_tracking_uri: str,
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        """
        Returns an instance of the specified trainer class.
        """
        # update any potentially missing or conflicting values
        optimizer_kwargs["metric"] = prompt.evaluate_metric
        optimizer_kwargs["student"] = prompt.infer
        optimizer_kwargs["trainset"] = trainset

        trainer_classes = {
            "DocQualityTrainer": DocQualityTrainer,
            "DocGeneratorTrainer": DocGeneratorTrainer,
        }
        if trainer_class not in trainer_classes:
            raise ValueError(f"Unknown trainer class: {trainer_class}")

        try:
            return trainer_classes[trainer_class](
                prompt=prompt,  # type: ignore # TODO: we should have better type checking here
                optimizer_type=optimizer_type,
                optimizer_kwargs=optimizer_kwargs,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_model_name=mlflow_model_name,
                mlflow_experiment_name=mlflow_experiment_name,
                trainset=trainset,
                evalset=evalset,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize trainer class ({trainer_class}): {e}"
            )
