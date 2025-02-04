# system packages 
from typing import Union

# internal packages
from .schema_doc_quality import DocQualityPrompt, DocQualitySignature
from .schema_doc_generation import DocGeneratorPrompt, DocGeneratorSignature
from .single_prompt import SinglePrompt

# external packages
import dspy


class PromptFactory:
    @staticmethod
    def get_single_prompt(
        prompt: Union[str, dspy.Signature],
        prompt_class: str,
        prompt_type: str,
        prompt_metric: Union[str, DocQualityPrompt, SinglePrompt],
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
                prompt=prompt, type=prompt_type, metric_type=prompt_metric  # type: ignore
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize prompt class ({prompt_class}): {e}")


# David Cheriton(?) - one of the first google developers
# avoid uneccessary verbs (get_single_prompt -> single_prompt): this is a readability thing. oop is nouns, not verbs.
