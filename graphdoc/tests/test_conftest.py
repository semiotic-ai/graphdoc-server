# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality

# external packages
import pytest
from dspy import LM, Predict

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestConftest:

    def test_gd(self, gd: GraphDoc):
        assert isinstance(gd.lm, LM)
        assert isinstance(gd.doc_eval, Predict)
        # assert isinstance(gd.doc_eval.signature, DocQuality) # TODO: this fails...
