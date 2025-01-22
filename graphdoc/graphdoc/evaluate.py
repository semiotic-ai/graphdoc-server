# system packages
import logging
from typing import Literal

# internal packages

# external packages
import dspy
from dspy import Signature, InputField, OutputField, Example

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DocQualityEval:
    """
    A helper class for dealing with evaluation of DocQuality.
    """

    def __init__(self) -> None:
        pass

    def validate_category(
        self, example: Example, prediction: Example, trace=None
    ) -> bool:
        try:
            return prediction.category == example.category
        except Exception as e:
            log.warning(f"Category validation failed due to error: {e}")
            return False

    def validate_rating(self, example: Example, prediction: Example, trace=None):
        try:
            return prediction.rating == example.rating
        except Exception as e:
            log.warning(f"Rating validation failed due to error: {e}")
            return False


#################
# DSPy Modules  #
#################
class DocQuality(Signature):
    """
    Given a GraphQL Schema, evaluate the quality of documentation for that schema and provide a category rating.
    The categories are described as:
    - perfect (4): The documentation contains enough information so that the interpretation of the schema and its database content is completely free of ambiguity.
    - almost perfect (3): The documentation is almost perfect and free from ambiguity, but there is room for improvement.
    - somewhat correct (2): The documentation is somewhat correct but has room for improvement due to missing information. The documentation is not incorrect.
    - incorrect (1): The documentation is incorrect and contains inaccurate or misleading information. Any incorrect information automatically leads to an incorrect rating, even if some correct information is present.
    Output a number rating that corresponds to the categories described above.
    """

    database_schema: str = InputField()
    category: Literal[
        "perfect",
        "almost perfect",
        "somewhat correct",
        "incorrect",
    ] = OutputField()
    rating: Literal[4, 3, 2, 1] = OutputField()
