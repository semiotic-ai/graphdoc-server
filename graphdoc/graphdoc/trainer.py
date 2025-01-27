# system packages
from enum import Enum
import logging
from typing import Callable, List, Optional, Union

# internal packages

# external packages
import dspy
from dspy import Example

# configure logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TrainerRunner:
    """
    A class for the optimization of a prompt.
    """

    def __init__(
        self,
        prompt: dspy.Signature,
        metric: Callable,
        module_type: Callable,
        optimizer: str,
        trainset: List[Example],
        evalset: List[Example],
    ) -> None:
        """
        Initialize the TrainerRunner.

        :param prompt: The prompt to optimize.
        :param metric: The metric used to optimize.
        :param module_type: The type of module used for the prompt.
        :param optimizer: The optimizer used to optimize the prompt.
        :param trainset: The training dataset.
        :param evalset: The evaluation dataset.
        """
        self.prompt = prompt
        self.metric = metric
        self.trainset = trainset
        self.evalset = evalset

    # modules
    def _predict(self, prompt: dspy.Signature) -> dspy.Predict:
        return dspy.Predict(prompt)

    def _chain_of_thought(self, prompt: dspy.Signature) -> dspy.ChainOfThought:
        return dspy.ChainOfThought(prompt)

    # optimizers
    def _mipro_v2(self, metric: Callable, optimizer_run: str = "light") -> dspy.MIPROv2:
        return dspy.MIPROv2(metric=metric, auto=optimizer_run)

    def initialize_trainer(
        self,
        optimizer: str = "mipro_v2",
        optimizer_run: str = "light",
        metric: Optional[Callable] = None,
    ) -> dspy.MIPROv2:
        if optimizer == "mipro_v2":
            if metric is None:
                metric = self.metric
            return self._mipro_v2(metric=metric, optimizer_run=optimizer_run)
        else:
            raise ValueError(
                f"Optimizer {optimizer} not supported. Supported optimizers are: mipro_v2"
            )

    def run_trainer(
        self,
        trainer: dspy.MIPROv2,
        module_type: str = "predict",
        trainset: Optional[List[Example]] = None,
    ) -> Callable:
        if module_type == "predict":
            module = self._predict(self.prompt)
        elif module_type == "chain_of_thought":
            module = self._chain_of_thought(self.prompt)
        else:
            raise ValueError(
                f"Module type {module_type} not supported. Supported module types are: predict, chain_of_thought"
            )

        if trainset is None:
            trainset = self.trainset

        optimized_prompt = trainer.compile(
            module, trainset=trainset, max_labeled_demos=0, max_bootstrapped_demos=0
        )

        return optimized_prompt

    # evaluate the result

    # save the model

    # run the trainer


# TrainerRunner
# def __init__(self,
# prompt
# metric:
# dataset: train / eval
# logging: - always log to MLFlow: keep this hardcoded for MLFlow for now

# hard code and assume we are using the right dataset

# initialize the trainer

# run the trainer
# log the results to MLFlow

# evaluate the output on our eval dataset
# log the results to MLFlow

# save the model to MLFlow
