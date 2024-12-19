# system packages
import os
import pickle
import asyncio
import logging
from pathlib import Path

# internal packages

# external packages
import pytest

CACHE_DIR = Path(__file__).parent / 'assets/cache/'

class TestEntityComparison:
    
    def test_run(self):
        assert 1 == 1

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_with_fire(self):
        assert True 

    @pytest.mark.skipif("not config.getoption('--dry-fire')")
    def test_with_dry_fire(self):
        assert True

    @pytest.mark.skipif("not (config.getoption('--fire') or config.getoption('--dry-fire'))")
    def test_prompt_entity_comparison(self, gd, entity_comparison_assets, request):
        """
        file_name = "test_prompt_entity_comparison.pkl"
        cache_pick_file = {
            "gold_entity_comparison": "gold_entity_comparison.pkl",
        }
        """
        #################### caching ####################
        # TODO: we should break out this caching to a fixture, but for now we just want to get the test working
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_prompt_entity_comparison.pkl"
    
        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        if os.path.exists(test_cache_file_path) and dry_fire:
            with open(test_cache_file_path, "rb") as f:
                response = pickle.load(f)
                response = response["gold_entity_comparison"]

        #################### testing ####################
        if os.path.exists(CACHE_DIR) and fire: 
            response = gd.prompt_entity_comparison(entity_comparison_assets["gold_entity_comparison"], entity_comparison_assets["gold_entity_comparison"])
            cache = { "gold_entity_comparison" : response }
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)
  
        response = gd.language_model.parse_response(response)
        
        assert response["correctness"] in [1, 2, 3, 4]
        assert isinstance(response["reasoning"], str)

    @pytest.mark.asyncio
    @pytest.mark.skipif("not config.getoption('--fire')")
    async def test_instantiate_entity_comparison_revision_prompt(self, gd, entity_comparison_assets, request):
        """
        file_name = "test_instantiate_entity_comparison_revision_prompt.pkl"
        cache_pick_file = {
            test_asset_comparisons: "test_prompt_entity_comparison.pkl",
        }
        """
        #################### caching ####################
        # TODO: we should break out this caching to a fixture, but for now we just want to get the test working
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_instantiate_entity_comparison_revision_prompt.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        if os.path.exists(test_cache_file_path) and dry_fire:
            with open(test_cache_file_path, "rb") as f:
                response = pickle.load(f)
                test_asset_comparisons = response["test_asset_comparisons"]

        #################### testing ####################
        if os.path.exists(CACHE_DIR) and fire:
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

            test_asset_comparisons = await asyncio.gather(*tasks)
            
            cache = { "test_asset_comparisons" : test_asset_comparisons }
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)
        
        parsed_test_asset_comparisons = [gd.language_model.parse_response(r) for r in test_asset_comparisons]

        for test_asset_comparison in parsed_test_asset_comparisons:
            assert test_asset_comparison["correctness"] in [1, 2, 3, 4], f"Unexpected correctness: {test_asset_comparison['correctness']}"
            assert isinstance(test_asset_comparison["reasoning"], str), "Reasoning should be a string"

        # revised_prompt = gd.prompt_entity_comparison_revision(
        #     original_prompt_template = gd.entity_comparison_prompt_template,
        #     four_comparison = parsed_test_asset_comparisons[0],
        #     three_comparison = parsed_test_asset_comparisons[1],
        #     two_comparison = parsed_test_asset_comparisons[2],
        #     one_comparison = parsed_test_asset_comparisons[3],
        # )

        # assert isinstance(revised_prompt["reasoning"], str)
        # assert isinstance(revised_prompt["modified_prompt"], int)