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

    def test_entity_comparison_assets_fixture(self, entity_comparison_assets):
        assert entity_comparison_assets["gold_entity_comparison"] != None
        assert entity_comparison_assets["four_entity_comparison"] != None
        assert entity_comparison_assets["three_entity_comparison"] != None
        assert entity_comparison_assets["two_entity_comparison"] != None
        assert entity_comparison_assets["one_entity_comparison"] != None