# useful tutorial: https://dspy.ai/tutorials/multihop_search/
# system packages
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Tuple, Union, cast

# internal packages

# external packages
import dspy

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# config variables
# prompt_type: str (predict, chain_of_thought, custom)
# metric_type: str (will differ for each prompt)


# this is a template for handling a single prompt
class SinglePrompt(ABC):
    # require that a dspy.Signature gets passed in
    def __init__(
        self,
        prompt: dspy.Signature,
        type: Union[
            Literal["predict", "chain_of_thought"], Callable
        ],  # TODO: we should rename this to prompt_type
        metric_type: str,  # TODO: we should rename this to prompt_metric
    ) -> None:
        self.prompt = prompt
        self.type = type
        self.metric_type = metric_type  # we will use this to determine which metric to use for evaluation

        if self.type == "predict":
            self.infer = self.get_predict()
        elif self.type == "chain_of_thought":
            self.infer = self.get_chain_of_thought()
        elif isinstance(self.type, Callable):
            log.warning(
                f"Using alternative dspy.Module for inference, please know what you are doing"
            )
            self.infer = self.type
        else:
            raise ValueError(f"Invalid type: {self.type}")

    #######################################
    # methods for initializing the prompt #
    #######################################
    def get_predict(self) -> dspy.Predict:
        return dspy.Predict(self.prompt)

    def get_chain_of_thought(self) -> dspy.ChainOfThought:
        return dspy.ChainOfThought(self.prompt)

    #######################################
    # methods for evaluating the prompt   #
    #######################################
    @abstractmethod
    def evaluate_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> Any:
        """This is the metric used to evalaute the prompt"""
        pass

    @abstractmethod
    def _format_metric(
        self,
        examples: List[dspy.Example],
        overall_score: float,
        results: List,
        scores: List,
    ) -> Dict[str, Any]:
        # self.metric_type # ensure that the metric type is used to format the results
        """This takes the results from the evaluate_evalset and does any necessary formatting, taking into account the metric type"""
        pass

    @abstractmethod
    def _compare_metrics(
        self, base_metrics, optimized_metrics, comparison_value: str = "overall_score"
    ) -> bool:
        """Compare the metrics of the base and optimized models

        returns true if the optimized model is better than the base model
        """
        pass

    def evaluate_evalset(
        self,
        examples: List[dspy.Example],
        num_threads: int = 1,
        display_progress: bool = True,
        display_table: bool = True,
    ) -> float:
        """Take in a list of examples and evaluate the results"""
        evaluator = dspy.Evaluate(
            devset=examples,
            num_threads=num_threads,
            display_progress=display_progress,
            display_table=display_table,
            return_all_scores=True,
            return_outputs=True,
        )
        try:
            # TODO: we may want to type this better
            overall_score, results, scores = evaluator(self.infer, self.evaluate_metric)  # type: ignore
            return self._format_metric(examples, overall_score, results, scores)  # type: ignore
        except Exception as e:
            log.error(f"Error evaluating evalset: {e}")
            raise e
