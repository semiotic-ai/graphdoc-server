# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser
from graphdoc import DataHelper

# external packages
import pytest
from dspy import LM, Predict
from datasets import Features

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestDataHelper:

    def test__get_graph_doc_columns(self, dh: DataHelper):
        features = dh._get_graph_doc_columns()
        assert isinstance(features, Features)
