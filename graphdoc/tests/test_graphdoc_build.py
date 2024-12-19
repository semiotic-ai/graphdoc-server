# system packages

# internal packages
from graphdoc import GraphDoc, LanguageModel

# external packages
import pytest

class TestGraphDocBuild:
    
    def test_language_model_fixture(self, lm: LanguageModel):
        assert lm.api_key != None

    def test_graphdoc_fixture(self, gd: GraphDoc):
        assert gd.language_model != None
