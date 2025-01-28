# system packages
from abc import abstractmethod
from typing import List, Literal

# internal packages
from .prompt import SinglePrompt

# external packages
import dspy


###################
# DSPy Signatures #
###################
class DocQualitySignature(dspy.Signature):
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
    category: Literal["perfect", "almost perfect", "somewhat correct", "incorrect"] = (
        dspy.OutputField()
    )
    rating: Literal[4, 3, 2, 1] = dspy.OutputField()


#######################
# Single Prompt Class #
#######################
class DocQualityPrompt(SinglePrompt):
    def __init__(
        self,
        type: Literal["predict", "chain_of_thought"] = "predict",
        metric_type: Literal["rating", "category"] = "rating",
    ) -> None:
        # TODO: we should type this better
        dcs = DocQualitySignature
        super().__init__(prompt=dcs, type=type, metric_type=metric_type)  # type: ignore

    def _evaluate_rating_metric(
        self, example: dspy.Example, prediction: dspy.Prediction
    ) -> bool:
        return example.rating == prediction.rating

    def _evaluate_category_metric(
        self, example: dspy.Example, prediction: dspy.Prediction
    ) -> bool:
        return example.category == prediction.category

    def evaluate_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> bool:
        if self.metric_type == "rating":
            return self._evaluate_rating_metric(example, prediction)
        elif self.metric_type == "category":
            return self._evaluate_category_metric(example, prediction)
        else:
            raise ValueError(f"Invalid metric type: {self.metric_type}")

    def evaluate(self, example: dspy.Example) -> None:
        """Take in an example, generate the result, and then evaluate the result"""
        # prediction = self.infer(example) # implement this based on the type of prompt
        # return self.evaluate_metric(example, prediction)
        pass

    def _format_metric(
        self,
        examples: List[dspy.Example],
        overal_score: float,
        results: List,
        scores: List,
    ):
        # self.metric_type # ensure that the metric type is used to format the results
        """This takes the results from the evaluate_evalset and does any necessary formatting, taking into account the metric type"""
        pass
