# system packages
import logging
from typing import Callable, List, Literal, Optional, Union

# internal packages
from .data import DataHelper

# external packages
import dspy
from dspy import Example, Prediction
from dspy.evaluate import Evaluate

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DocQualityEval:
    """
    A helper class for dealing with evaluation of DocQuality.
    """

    def __init__(self, dh: Optional[DataHelper] = None) -> None:
        if dh:
            self.dh = dh
        else:
            self.dh = DataHelper()

    def validate_category(
        self, example: Example, prediction: Example, trace=None
    ) -> bool:
        """
        A helper function to validate the category of the prediction.

        :param example: The example to validate against
        :type example: Example
        :param prediction: The prediction to validate
        :type prediction: Example
        :param trace: The trace of the prediction
        :type trace: Optional[str]
        :return: Whether the prediction is correct
        :rtype: bool
        """
        try:
            return prediction.category == example.category
        except Exception as e:
            log.warning(f"Category validation failed due to error: {e}")
            return False

    def validate_rating(
        self, example: Example, prediction: Prediction, trace=None
    ) -> bool:
        """
        A helper function to validate the rating of the prediction.

        :param example: The example to validate against
        :type example: Example
        :param prediction: The prediction to validate
        :type prediction: Example
        :param trace: The trace of the prediction
        :type trace: Optional[str]
        :return: Whether the prediction is correct
        :rtype: bool
        """
        try:
            log.info(f"validate_rating: Prediction Rating: {prediction.rating} {type(prediction.rating)}")
            log.info(f"validate_rating: Example Rating: {example.rating} {type(example.rating)}")
            return prediction.rating == example.rating
        except Exception as e:
            log.warning(f"Rating validation failed due to error: {e}")
            return False

    def create_evaluator(
        self,
        repo_id: str = "semiotic/graphdoc_schemas",
        num_threads: int = 1,
        display_progress: bool = True,
        display_table: bool = True,
        token: Optional[str] = None,
        trainset: Optional[List[Example]] = None,
    ) -> Evaluate:
        """
        A helper function to create an evaluator for the DocQuality module.

        :param repo_id: The repository ID to load the dataset from
        :type repo_id: str
        :param num_threads: The number of threads to use for evaluation
        :type num_threads: int
        :param display_progress: Whether to display progress
        :type display_progress: bool
        :param display_table: The number of rows to display in the table
        :type display_table: int
        :param token: The Hugging Face API token
        :type token: Optional[str]
        :return: The evaluator
        :rtype: Evaluate
        """
        if not trainset:
            trainset = self.dh.create_graph_doc_example_trainset(
                repo_id=repo_id, token=token
            )
        return Evaluate(
            devset=trainset,
            num_threads=num_threads,
            display_progress=display_progress,
            display_table=display_table,
        )

    def run_evaluator(
        self, evaluator: Evaluate, module: dspy.Predict, metric: Callable
    ) -> None:
        """
        A helper function to run the evaluator.
        """
        evaluator(module, metric=metric)


#################
# DSPy Modules  #
#################
class DocQuality(dspy.Signature):
    """
    Given a GraphQL Schema, evaluate the quality of documentation for that schema and provide a category rating.
    The categories are described as:
    - perfect (4): The documentation contains enough information so that the interpretation of the schema and its database content is completely free of ambiguity.
    - almost perfect (3): The documentation is almost perfect and free from ambiguity, but there is room for improvement.
    - somewhat correct (2): The documentation is somewhat correct but has room for improvement due to missing information. The documentation is not incorrect.
    - incorrect (1): The documentation is incorrect and contains inaccurate or misleading information. Any incorrect information automatically leads to an incorrect rating, even if some correct information is present.
    Output a number rating that corresponds to the categories described above.
    """

    database_schema: str = dspy.InputField()
    category: Literal[
        "perfect",
        "almost perfect",
        "somewhat correct",
        "incorrect",
    ] = dspy.OutputField()
    rating: Literal[4, 3, 2, 1] = dspy.OutputField()


#########################
# DSPy Optimizer Module #
#########################
class DocQualityOptimizer(dspy.Signature):
    """
    A helper class for dealing with optimization of DocQuality.
    """

    def __init__(self) -> None:
        pass

    # def
