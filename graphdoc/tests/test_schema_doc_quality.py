# system packages
import logging

# internal packages
from graphdoc.prompts import DocQualityPrompt, DocQualitySignature, SinglePrompt

# external packages
import dspy

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestSchemaDocQuality:
    def test_schema_doc_quality(self):
        pass

    def test_doc_quality_prompt(self):
        dqp = DocQualityPrompt(type="predict", metric_type="rating")
        assert isinstance(dqp, SinglePrompt)
        assert isinstance(dqp.infer, dspy.Predict)

    # TODO: we should move this to a test for the SinglePrompt class
    def test_get_predict(self):
        dqp = DocQualityPrompt(type="predict", metric_type="rating")
        p = dqp.get_predict()
        assert isinstance(p, dspy.Predict)

    # TODO: we should move this to a test for the SinglePrompt class
    def test_get_chain_of_thought(self):
        dqp = DocQualityPrompt(type="chain_of_thought", metric_type="rating")
        p = dqp.get_chain_of_thought()
        assert isinstance(p, dspy.ChainOfThought)
