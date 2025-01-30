# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc.prompts import DocGeneratorPrompt, DocQualityPrompt, SinglePrompt
from graphdoc import DataHelper, GraphDoc

# external packages
import dspy

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# global variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class TestDocGeneratorPrompt:
    def test_doc_generator_prompt(self):
        pass

    def test_doc_generator_prompt_init(self):
        dqp = DocQualityPrompt(type="predict", metric_type="rating")
        dgp = DocGeneratorPrompt(type="chain_of_thought", metric_type=dqp)
        assert isinstance(dgp, SinglePrompt)
        assert isinstance(dgp.infer, dspy.ChainOfThought)
    
    def test_doc_generator_prompt_init_from_config(self, gd: GraphDoc):
        config_path = BASE_DIR / "graphdoc" / "tests" / "assets" / "configs" / "single_prompt_schema_doc_generator_trainer.yaml"
        metric_config_path = BASE_DIR / "graphdoc" / "tests" / "assets" / "configs" / "single_prompt_schema_doc_quality_trainer.yaml"

        dgp = gd._get_nested_single_prompt(
            config_path=config_path,
            metric_config_path=metric_config_path,
        )
        assert isinstance(dgp, SinglePrompt)
        assert isinstance(dgp, DocGeneratorPrompt)
        assert isinstance(dgp.infer, dspy.ChainOfThought)
