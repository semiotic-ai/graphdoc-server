import pytest

class TestEntityComparison:
    
    def test_run(self):
        assert 1 == 1

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_with_fire(self):
        assert True 

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_prompt_entity_comparison(self, gd, entity_comparison_assets):
        response = gd.prompt_entity_comparison(entity_comparison_assets["gold_entity_comparison"], entity_comparison_assets["gold_entity_comparison"])
        response = gd.language_model.parse_response(response)
        
        assert response["correctness"] in [1, 2, 3, 4]
        assert isinstance(response["reasoning"], str)

    # def test_graphdoc_fixture(self, gd: GraphDoc):
    #     assert gd.language_model != None

    # def test_entity_comparison_assets_fixture(self, entity_comparison_assets):
    #     assert entity_comparison_assets["gold_entity_comparison"] != None
    #     assert entity_comparison_assets["four_entity_comparison"] != None
    #     assert entity_comparison_assets["three_entity_comparison"] != None
    #     assert entity_comparison_assets["two_entity_comparison"] != None
    #     assert entity_comparison_assets["one_entity_comparison"] != None

    