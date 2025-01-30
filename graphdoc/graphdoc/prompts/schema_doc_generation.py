# system packages
import logging
from typing import Any, Dict, List, Literal, Optional

# internal packages
from .single_prompt import SinglePrompt
from .schema_doc_quality import DocQualityPrompt
from ..parser import Parser

# external packages
import dspy
from graphql import parse, print_ast

# logging
log = logging.getLogger(__name__)


###################
# DSPy Signatures #
###################
class DocGeneratorSignature(dspy.Signature):
    """
    ### TASK:
    Given a GraphQL Schema, generate a precise description for the columns of the tables in the database.

    ### Requirements:
    - Focus solely on confirmed details from the provided schema.
    - Keep the description concise and factual.
    - Exclude any speculative or additional commentary.
    - DO NOT return the phrase "in the { table } table" in your description.

    ### Formatting
    - Ensure that the schema maintains proper documentation formatting, as is provided.
    """

    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField(
        desc="The database schema with proper documentation, ensuring that the underlying schema is not altered."
    )


#######################
# Single Prompt Class #
#######################
class DocGeneratorPrompt(SinglePrompt):
    def __init__(
        self,
        metric_type: DocQualityPrompt,
        type: Literal["predict", "chain_of_thought"] = "predict",
        prompt: Optional[dspy.Signature] = None,
    ) -> None:
        if prompt is None:
            prompt = DocGeneratorSignature  # type: ignore
        super().__init__(prompt=prompt, type=type, metric_type=metric_type)  # type: ignore

        # initialize the parser
        self.par = Parser()

    # metric functions
    def evaluate_documentation_quality(
        self, schema: dspy.Example, pred: dspy.Prediction, trace=None
    ) -> int:
        """
        A helper function to evaluate the quality of the documentation.

        :param pred: The prediction to evaluate
        :type pred: Prediction
        :param trace: The trace of the prediction
        :type trace: Optional[str]
        :return: The quality of the documentation
        :rtype: int
        """
        try:
            gold_schema = parse(schema.database_schema)
            pred_schema = parse(pred.documented_schema)
        except Exception as e:
            log.warning(
                f"evaluate_documentation_quality: An exception occurred while parsing the schema: {e}"
            )
            return 1
        if not self.par.schema_equality_check(gold_schema, pred_schema):
            log.warning("evaluate_documentation_quality: Schema equality check failed")
            return 1

        # we use the instantiated metric type to evaluate the quality of the documentation
        evaluation = self.metric_type.infer(database_schema=pred.documented_schema)
        log.info(f"evaluate_documentation_quality: Evaluation: {evaluation.rating}")
        return evaluation.rating

    # abstract methods
    def evaluate_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> Any:
        return self.evaluate_documentation_quality(example, prediction, trace)

    def _format_metric(
        self,
        examples: List[dspy.Example],
        overall_score: float,
        results: List,
        scores: List,
    ) -> Dict[str, Any]:
        return {
            "overall_score": overall_score,
            "scores": scores,
            "results": results,
        }

    def _compare_metrics(
        self, base_metrics, optimized_metrics, comparison_value: str = "overall_score"
    ) -> bool:
        if comparison_value == "overall_score":
            return optimized_metrics["overall_score"] > base_metrics["overall_score"]
        else:
            raise ValueError(f"Invalid comparison value: {comparison_value}")
