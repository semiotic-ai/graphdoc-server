from .schema_doc_quality import DocQualityPrompt, DocQualitySignature
from .single_prompt import SinglePrompt


class PromptFactory:
    @staticmethod
    def get_single_prompt(prompt_class: str, prompt_type: str, prompt_metric: str) -> SinglePrompt:
        """
        Returns an instance of the specified prompt class.
        """
        prompt_classes = {
            "SchemaDocQualityPrompt": DocQualityPrompt,
        }
        if prompt_class not in prompt_classes:
            raise ValueError(f"Unknown prompt class: {prompt_class}")
        try:
            return prompt_classes[prompt_class](type=prompt_type, metric_type=prompt_metric)
        except Exception as e:
            raise ValueError(f"Failed to initialize prompt class ({prompt_class}): {e}")
