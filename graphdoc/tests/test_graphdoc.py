# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc, DataHelper
from graphdoc import DocQualityTrainer, SinglePrompt
from graphdoc import load_yaml_config

# external packages

# logging
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

    def test__get_single_trainer(self, gd: GraphDoc, dh: DataHelper):
        graphdoc_ds = dh._folder_of_folders_to_dataset()
        trainset = dh._create_graph_doc_example_trainset(graphdoc_ds)
        evalset = dh._create_graph_doc_example_trainset(graphdoc_ds)

        config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_trainer.yaml"
        )
        config = load_yaml_config(config_path)
        trainer = gd._get_single_trainer(config_path, trainset, evalset)
        assert isinstance(trainer, DocQualityTrainer)
