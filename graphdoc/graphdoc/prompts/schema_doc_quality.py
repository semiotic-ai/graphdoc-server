# system packages
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union

# internal packages
from .single_prompt import SinglePrompt

# external packages
import dspy

# logging
log = logging.getLogger(__name__)


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

class DocQualityDemonstrationSignature(dspy.Signature):
    """
    Given a GraphQL Schema, evaluate the quality of documentation for that schema and provide a category rating.
    The categories are described as:
    - perfect (4): The documentation contains enough information so that the interpretation of the schema and its database content is completely free of ambiguity.
        perfect (4) example:
        type Domain @entity {
            " The namehash (id) of the parent name. References the Domain entity that is the parent of the current domain. Type: Domain " 
            parent: Domain
        }
    - almost perfect (3): The documentation is almost perfect and free from ambiguity, but there is room for improvement.
        almost perfect (3) example:
        type Token @entity {
            " Name of the token, mirrored from the smart contract "
            name: String!
        }
    - somewhat correct (2): The documentation is somewhat correct but has room for improvement due to missing information. The documentation is not incorrect.
        somewhat correct (2) example:
        type InterestRate @entity {
            "Description for column: id"
            id: ID!
        }
    - incorrect (1): The documentation is incorrect and contains inaccurate or misleading information. Any incorrect information automatically leads to an incorrect rating, even if some correct information is present.
        incorrect (1) example:
        type BridgeProtocol implements Protocol @entity {
            " Social Security Number of the protocol's main developer "
            id: Bytes!
        }
    Output a number rating that corresponds to the categories described above.
    """

    database_schema: str = dspy.InputField()
    category: Literal["perfect", "almost perfect", "somewhat correct", "incorrect"] = (
        dspy.OutputField()
    )
    rating: Literal[4, 3, 2, 1] = dspy.OutputField()


def doc_quality_factory(key: Union[str, dspy.Signature]):
    if not isinstance(key, str):  # TODO: we could handle this in a much better way
        return key
    factory = {
        "doc_quality": DocQualitySignature,
        "doc_quality_demo": DocQualityDemonstrationSignature,
    }
    return factory[key]


#######################
# Single Prompt Class #
#######################
class DocQualityPrompt(SinglePrompt):
    def __init__(
        self,
        type: Literal["predict", "chain_of_thought"] = "predict",
        metric_type: Literal[
            "rating", "category"
        ] = "rating",  # this could be a factory function instead
        prompt: Union[str, dspy.Signature] = "doc_quality",
        # prompt: Optional[dspy.Signature] = None,
    ) -> None:
        # TODO: we should type this better
        # if prompt is None:
        # prompt = DocQualitySignature  # type: ignore
        prompt_signature = doc_quality_factory(prompt)
        super().__init__(prompt=prompt_signature, type=type, metric_type=metric_type)  # type: ignore

    # def _evaluate_rating_diff() # with a lot of data...

    def _evaluate_rating_metric(
        self, example: dspy.Example, prediction: dspy.Prediction
    ) -> bool:
        return example.rating == prediction.rating

    def _evaluate_category_metric(
        self, example: dspy.Example, prediction: dspy.Prediction
    ) -> bool:
        return example.category == prediction.category

    # abstract methods
    def evaluate_metric(  # this could be a factory function instead that we map to the metric function
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> bool:  # factory function here would remove rtti
        if self.metric_type == "rating":
            return self._evaluate_rating_metric(example, prediction)
        elif self.metric_type == "category":
            return self._evaluate_category_metric(example, prediction)
        else:
            raise ValueError(f"Invalid metric type: {self.metric_type}")

    def _format_metric(  # this should be public
        self,
        examples: List[dspy.Example],
        overall_score: float,
        results: List,
        scores: List,
    ) -> Dict[str, Any]:
        """This takes the results from the evaluate_evalset and does any necessary formatting"""

        formatted_results = {
            "overall_score": overall_score,
            "per_category_scores": {},
            "details": [],
        }
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for result, score in zip(results, scores):
            example, prediction, is_correct = result
            example_data = {key: value for key, value in example.items()}

            category = example_data.get("category", "unknown")
            expected_rating = example_data.get("rating", None)

            predicted_category = getattr(prediction, "category", "unknown")
            predicted_rating = getattr(prediction, "rating", None)

            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1

            detail_entry = {
                "expected_category": category,
                "expected_rating": expected_rating,
                "predicted_category": predicted_category,
                "predicted_rating": predicted_rating,
                "is_correct": is_correct,
            }
            formatted_results["details"].append(detail_entry)

        for category, stats in category_stats.items():
            percent_correct = (
                (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            )
            formatted_results["per_category_scores"][category] = {
                "percent_correct": percent_correct,
                "total": stats["total"],
                "correct": stats["correct"],
            }

        formatted_results["results"] = results

        return formatted_results

    def _compare_metrics(
        self, base_metrics, optimized_metrics, comparison_value: str = "overall_score"
    ) -> bool:
        """Compare the metrics of the base and optimized models

        returns true if the optimized model is better than the base model
        """
        if comparison_value == "overall_score":
            return optimized_metrics["overall_score"] > base_metrics["overall_score"]
        else:
            raise ValueError(f"Invalid comparison value: {comparison_value}")
