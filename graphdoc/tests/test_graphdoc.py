# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc, DataHelper
from graphdoc import DocQualityTrainer, SinglePrompt, DocQualityPrompt, DocGeneratorPrompt
from graphdoc import load_yaml_config
from graphdoc import DocGeneratorModule

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

    def test_prompt_from_mlflow(self, gd: GraphDoc):
        config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_schema_doc_quality_trainer.yaml"
        )
        prompt = gd.prompt_from_mlflow(config_path)
        assert isinstance(prompt, DocQualityPrompt)

    def test_nested_prompt_from_mlflow(self, gd: GraphDoc):
        config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_schema_doc_generator_trainer.yaml"
        )
        metric_config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_schema_doc_quality_trainer.yaml"
        )
        prompt = gd.nested_prompt_from_mlflow(config_path, metric_config_path)
        assert isinstance(prompt, DocGeneratorPrompt)

    def test_doc_generator_module_from_mlflow(self, gd: GraphDoc):
        config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_schema_doc_generator_trainer.yaml"
        )
        metric_config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_schema_doc_quality_trainer.yaml"
        )
        module = gd.doc_generator_module(config_path, metric_config_path)
        assert isinstance(module, DocGeneratorModule)