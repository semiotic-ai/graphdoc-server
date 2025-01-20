# system packages
import logging
import os

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser
from graphdoc import DataHelper

# external packages
import pytest
from dspy import LM, Predict
from dotenv import load_dotenv

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TestConftest:

    def test_gd(self, gd: GraphDoc):
        assert isinstance(gd.lm, LM)
        assert isinstance(gd.doc_eval, Predict)
        # assert isinstance(gd.doc_eval.signature, DocQuality) # TODO: this fails...

    def test_par(self, par: Parser):
        assert par.schema_directory_path is not None

    def test_dh(self, dh: DataHelper):
        assert isinstance(dh, DataHelper)
        assert dh.hf_api_key is not None
