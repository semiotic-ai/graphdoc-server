# system packages
import logging

# internal packages
from graphdoc.prompts import DocGeneratorPrompt, DocQualityPrompt, SinglePrompt
from graphdoc import DataHelper, GraphDoc

# external packages
import dspy

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestDocGeneratorPrompt:
    def test_doc_generator_prompt(self):
        pass

    def test_doc_generator_prompt_init(self):
        dqp = DocQualityPrompt(type="predict", metric_type="rating")
        dgp = DocGeneratorPrompt(type="chain_of_thought", metric_type=dqp)
        assert isinstance(dgp, SinglePrompt)
        assert isinstance(dgp.infer, dspy.ChainOfThought)
