from typing import Union
from .schema_doc_quality import DocQualityPrompt, DocQualitySignature
from .schema_doc_generation import DocGeneratorPrompt, DocGeneratorSignature
from .single_prompt import SinglePrompt


class PromptFactory:
    @staticmethod
    def get_single_prompt(
        prompt_class: str, prompt_type: str, prompt_metric: Union[str, DocQualityPrompt]
    ) -> SinglePrompt:
        """
        Returns an instance of the specified prompt class.
        """
        prompt_classes = {
            "SchemaDocQualityPrompt": DocQualityPrompt,
            "DocGeneratorPrompt": DocGeneratorPrompt,
        }
        if prompt_class not in prompt_classes:
            raise ValueError(f"Unknown prompt class: {prompt_class}")
        try:
            # TODO: we should be able to have better type checking here
            return prompt_classes[prompt_class](
                type=prompt_type, metric_type=prompt_metric  # type: ignore
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize prompt class ({prompt_class}): {e}")
