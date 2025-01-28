from .single_prompt_trainer import SinglePromptTrainerRunner
from .doc_quality_trainer import DocQualityTrainer


class TrainerFactory:
    @staticmethod
    def get_trainer(trainer_class: str):
        """
        Returns an instance of the specified trainer class.
        """
        trainer_classes = {
            "DocQualityTrainer": DocQualityTrainer,
        }
        if trainer_class not in trainer_classes:
            raise ValueError(f"Unknown trainer class: {trainer_class}")

        try:
            return trainer_classes[trainer_class]
        except Exception as e:
            raise ValueError(
                f"Failed to initialize trainer class ({trainer_class}): {e}"
            )
