# system packages
import logging
from typing import Callable, List, Literal, Optional, Union

from graphdoc.evaluate import DocQuality
from graphql import parse, print_ast

# internal packages
from .data import DataHelper


# external packages
import dspy
from dspy import Example, Prediction
from dspy.evaluate import Evaluate

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DocGeneratorEval:
    """
    A helper class for dealing with evaluation of DocGenerator.
    """

    def __init__(self, dh: Optional[DataHelper] = None) -> None:
        if dh:
            self.dh = dh
        else:
            self.dh = DataHelper()

    def validate_schema_format(
        self, entity: Example, prediction: Prediction, trace=None
    ) -> bool:
        """
        A helper function to validate the schema format of the prediction.

        :param entity: The entity to validate against
        :type entity: Example
        :param prediction: The prediction to validate
        :type prediction: Prediction
        :param trace: The trace of the prediction
        :type trace: Optional[str]
        :return: Whether the prediction is correct
        :rtype: bool
        """
        try:
            input_schema = entity.database_schema
            output_schema = prediction.documented_schema

            if isinstance(input_schema, str) and isinstance(output_schema, str):
                input_ast = parse(input_schema)
                output_ast = parse(output_schema)
                return self.dh.par.schema_equality_check(input_ast, output_ast)
            else:
                return False
        except Exception as e:
            log.warning(f"Schema format validation failed due to error: {e}")
            return False

    def preprocess_schema(self, database_schema: str) -> str:
        """
        A helper function to preprocess the schema.

        :param database_schema: The database schema to preprocess
        :type database_schema: str
        :return: The preprocessed database schema
        :rtype: str
        """
        try:
            database_ast = parse(database_schema)
            updated_ast = self.dh.par.fill_empty_descriptions(database_ast)
            return print_ast(updated_ast)
        except Exception as e:
            raise ValueError(
                f"An exception occurred while preprocessing the schema: {e}"
            )

    def evaluate_documentation_quality(
        self, schema: Example, pred: Prediction, trace=None
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
            log.warning(f"evaluate_documentation_quality: An exception occurred while parsing the schema: {e}")
            return 1 
            # raise ValueError(f"An exception occurred while parsing the schema: {e}")
        if not self.dh.par.schema_equality_check(gold_schema, pred_schema):
            log.warning("evaluate_documentation_quality: Schema equality check failed")
            return 1

        evaluation = dspy.Predict(DocQuality)(database_schema=pred.documented_schema)
        log.info(f"evaluate_documentation_quality: Evaluation: {evaluation.rating}")
        return evaluation.rating


#################
# DSPy Modules  #
#################
class DocGenerator(dspy.Signature):
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
