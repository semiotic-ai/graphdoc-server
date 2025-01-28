# system packages

# internal packages
from pathlib import Path
from typing import Union
from .evaluate import DocQuality
from .loader.helper import load_yaml_config
from .train import TrainerFactory
from .prompts import PromptFactory

# external packages
import dspy


class GraphDoc:
    def __init__(
        self,
        model: str,
        api_key: str,
        cache: bool = True,
    ) -> None:

        # initialize base dspy config
        self.lm = dspy.LM(model=model, api_key=api_key, cache=cache)
        dspy.configure(lm=self.lm)

        # initialize modules
        self.doc_eval = dspy.Predict(DocQuality)

    ############
    # TRAINING #
    ############

    # def update_graphdoc_dataset():
    # load the dataset from the repo files
    # optionally, let another location be specified to pull data from
    # drop duplicates
    # require a version, dataset card, and commit message
    # push to the repo
    # return the version number and the commit SHA

    def _get_single_prompt(self, config_path: Union[str, Path]):
        config = load_yaml_config(config_path)
        try:
            prompt_class = config["prompt"]["class"]
            prompt_type = config["prompt"]["type"]
            prompt_metric = config["prompt"]["metric"]
            prompt = PromptFactory.get_single_prompt(prompt_class, prompt_type, prompt_metric)
        except Exception as e:
            raise ValueError(f"Failed to initialize prompt class ({prompt_class}): {e}")
        return prompt

    def _get_single_trainer(self, config_path: Union[str, Path]):
        config = load_yaml_config(config_path)
        trainer_class = config["trainer"]["trainer_class"]
        trainer = TrainerFactory.get_trainer(trainer_class)  # , **config["trainer"]
        return trainer
