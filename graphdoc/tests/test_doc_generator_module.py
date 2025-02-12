# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc.prompts import DocGeneratorPrompt, DocQualityPrompt, SinglePrompt
from graphdoc.modules import DocGeneratorModule
from graphdoc import DataHelper, GraphDoc

# external packages
import dspy

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# global variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestDocGeneratorModule:

    def test_doc_generator_module(self, gd: GraphDoc):
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

        dgp = gd._get_nested_single_prompt(
            config_path=config_path,
            metric_config_path=metric_config_path,
        )

        dgm = DocGeneratorModule(generator_prompt=dgp, retry=True)

        assert isinstance(dgp, DocGeneratorPrompt)
        assert isinstance(dgp.infer, dspy.ChainOfThought)
        assert isinstance(dgm, DocGeneratorModule)
