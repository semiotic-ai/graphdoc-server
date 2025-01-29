# system packages
from typing import List

# internal packages
from .single_prompt_trainer import SinglePromptTrainerRunner
from .doc_quality_trainer import DocQualityTrainer
from ..prompts import SinglePrompt

# external packages
import dspy


class TrainerFactory:
    @staticmethod
    def get_single_prompt_trainer(
        trainer_class: str,
        prompt: SinglePrompt,
        optimizer_type: str,
        mlflow_tracking_uri: str,
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        """
        Returns an instance of the specified trainer class.
        """
        trainer_classes = {
            "DocQualityTrainer": DocQualityTrainer,
        }
        if trainer_class not in trainer_classes:
            raise ValueError(f"Unknown trainer class: {trainer_class}")

        try:
            return trainer_classes[trainer_class](
                prompt=prompt,  # type: ignore # TODO: we should have better type checking here
                optimizer_type=optimizer_type,
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
