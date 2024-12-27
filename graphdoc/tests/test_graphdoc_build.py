# system packages

# internal packages
from graphdoc import LanguageModel, PromptExecutor, EntityComparisonPromptExecutor

# external packages
import pytest

class TestGraphDocBuild:
    
    def test_language_model_fixture(self, lm: LanguageModel):
        assert lm.api_key != None

    def test_prompt_executor_fixture(self, pe: PromptExecutor):
        assert pe.language_model != None
        assert pe.prompt_templates_dir != None

    def test_entity_comparison_prompt_executor(self, ecpe: EntityComparisonPromptExecutor): 
        assert ecpe.language_model != None
        assert ecpe.prompt_templates_dir != None

    def test_parser_fixture(self, par):
        assert par.schema_directory_path != None

    def test_entity_comparison_assets_fixture(self, entity_comparison_assets):
        assert entity_comparison_assets["gold_entity_comparison"] != None
        assert entity_comparison_assets["four_entity_comparison"] != None
        assert entity_comparison_assets["three_entity_comparison"] != None
        assert entity_comparison_assets["two_entity_comparison"] != None
        assert entity_comparison_assets["one_entity_comparison"] != None