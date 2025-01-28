# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQualityTrainer
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

    def test__get_trainer_class(self, gd: GraphDoc):
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
        trainer = gd._get_trainer_class(config_path)
        assert issubclass(trainer, DocQualityTrainer)
