# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQualityTrainer, SinglePrompt
from graphdoc import load_yaml_config

# external packages

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestGraphdoc:
    def test_graphdoc(self, gd: GraphDoc):
        assert isinstance(gd, GraphDoc)

    def test__get_single_prompt(self, gd: GraphDoc):
        config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_trainer.yaml"
        )
        prompt = gd._get_single_prompt(config_path)
        assert isinstance(prompt, SinglePrompt)

    def test__get_single_trainer(self, gd: GraphDoc):
        config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_trainer.yaml"
        )
        config = load_yaml_config(config_path)
        assert config["trainer"]["trainer_class"] == "DocQualityTrainer"
        trainer = gd._get_single_trainer(config_path)
        assert issubclass(trainer, DocQualityTrainer)
