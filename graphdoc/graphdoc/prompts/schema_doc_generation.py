# system packages
import logging
from typing import Any, Dict, List, Literal, Optional, Union

# internal packages
from .single_prompt import SinglePrompt
from .schema_doc_quality import DocQualityPrompt
from ..parser import Parser

# external packages
import dspy
from graphql import parse, print_ast
from dspy.utils.parallelizer import ParallelExecutor

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

class DocGeneratorHelperSignature(dspy.Signature):
    """
    ### TASK:
    Analyze the provided GraphQL Schema and generate detailed yet concise descriptions for each field within the database tables and enums. 
    
    ### Requirements:
    - If the field is unclear, and the documentation result is ambiguous, request additional information: "WARNING: Please provide additional information to avoid confusion".
    - Utilize only the verified information from the schema to ensure accuracy.
    - Descriptions should be factual, straightforward, and avoid any speculative language.
    - Refrain from using the phrase "in the { table } table" within your descriptions.
    - Ensure that the documentation adheres to standard schema formatting without modifying the underlying schema structure.
    
    ### Formatting:
    - Maintain consistency with the existing documentation style and structure.
    - Focus on clarity and precision to aid developers and system architects in understanding the schema's components effectively. 
    """

    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField(
        desc="The database schema with proper documentation, ensuring that the underlying schema is not altered."
    )

class BadDocGeneratorSignature(dspy.Signature):
    """
    ### TASK:
    Given a GraphQL Schema, generate intentionally incorrect documentation for the columns of the tables in the database.

    ### Requirements:
    - Every table, entity, enum, etc. must have at least one column with a description that is obviosly incorrect.
    - The documentation must be incorrect and misleading.
    - The documentation should be scattered, with only some columns having documentation.

    ### Formatting
    - Ensure that the schema maintains proper documentation formatting, as is provided.
    """

    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField(
        desc="The database schema with intentionally incorrect documentation, ensuring that the underlying schema is not altered."
    )


def doc_gen_factory(key: Union[str, dspy.Signature]):
    if not isinstance(key, str):  # TODO: we can handle this in a much better way
        return key
    factory = {
        "zero_shot_doc_gen": DocGeneratorSignature,
        "doc_gen_helper": DocGeneratorHelperSignature,
        "bad_doc_gen": BadDocGeneratorSignature,
    }
    return factory[key]


#######################
# Single Prompt Class #
#######################
class DocGeneratorPrompt(SinglePrompt):
    def __init__(
        self,
        metric_type: DocQualityPrompt,  # factory function here would unify our types
        type: Literal["predict", "chain_of_thought"] = "chain_of_thought",
        prompt: Union[str, dspy.Signature] = "zero_shot_doc_gen",
        # prompt: Optional[dspy.Signature] = None,
    ) -> None:
        # if prompt is None:
        # prompt = DocGeneratorSignature  # type: ignore
        prompt_signature = doc_gen_factory(prompt)
        super().__init__(prompt=prompt_signature, type=type, metric_type=metric_type)  # type: ignore

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
        return evaluation.rating ** 2 # MSE: not really, but the same idea, scale the value based on difference from descired score

    # abstract methods
    def evaluate_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> Any:
        return self.evaluate_documentation_quality(example, prediction, trace)

    def _format_metric(  # this should be public
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
        }  # WIP

    def _compare_metrics(
        self, base_metrics, optimized_metrics, comparison_value: str = "overall_score"
    ) -> bool:
        if comparison_value == "overall_score":
            return optimized_metrics["overall_score"] > base_metrics["overall_score"]
        else:
            raise ValueError(f"Invalid comparison value: {comparison_value}")

    #########################################
    # Schema Generation
    #########################################
    def decompose_and_document_schema(self, schema: str) -> Union[str, None]:
        """
        Decompose the schema into smaller components and document each component.
        """
        try:
            components = parse(schema)
            examples = []
            for component in components.definitions:
                component = self.par.fill_empty_descriptions(component)
                example = dspy.Example(database_schema=component, documented_schema="")
                examples.append(example)

            executor = ParallelExecutor(
                num_threads=4,
                disable_progress_bar=False,
                max_errors=4,
                provide_traceback=False,
                compare_results=False,
            )

            def process_item(example):
                prediction = self.infer(**example.inputs())
                if not self.par.schema_equality_check(
                    example.database_schema, prediction.documented_schema
                ):
                    log.warning("Schema equality check failed")
                    prediction = self.infer(
                        **example.inputs()
                    )  # we should handle retry logic better
                return prediction

            results = executor.execute(process_item, examples)
            # for result in

        except Exception as e:
            raise ValueError(f"An exception occurred while parsing the schema: {e}")
