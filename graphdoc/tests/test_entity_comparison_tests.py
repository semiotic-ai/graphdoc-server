# system packages
import asyncio
import logging

# internal packages

# external packages
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

    @pytest.mark.asyncio
    @pytest.mark.skipif("not config.getoption('--fire')")
    async def test_instantiate_entity_comparison_revision_prompt(self, gd, entity_comparison_assets):
        test_assets = [
            entity_comparison_assets["four_entity_comparison"],
            entity_comparison_assets["three_entity_comparison"],
            entity_comparison_assets["two_entity_comparison"],
            entity_comparison_assets["one_entity_comparison"]
        ]

        tasks = [
            asyncio.to_thread(
                gd.prompt_entity_comparison,
                entity_comparison_assets["gold_entity_comparison"],
                test_asset
            )
            for test_asset in test_assets
        ]

        responses = await asyncio.gather(*tasks)
        parsed_responses = [gd.language_model.parse_response(r) for r in responses]

        for response in parsed_responses:
            assert response["correctness"] in [1, 2, 3, 4], f"Unexpected correctness: {response['correctness']}"
            assert isinstance(response["reasoning"], str), "Reasoning should be a string"