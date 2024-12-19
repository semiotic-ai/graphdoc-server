# system packages

# internal packages
from graphdoc import GraphDoc, LanguageModel

# external packages
import pytest

class TestGraphDocBuild:
    
    def test_language_model(self):
        language_model = LanguageModel(api_key="test")
        assert language_model.api_key == "test"

    def test_run(self):
        assert 1 == 1
