# system packages
import os
import logging
from pathlib import Path

# internal packages
from graphdoc import LanguageModel, PromptExecutor, EntityComparisonPromptExecutor

# external packages
import pickle
from graphql import print_ast
import pytest
from graphdoc.main import GraphDoc

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# global variables
CACHE_DIR = Path(__file__).parent / "assets/cache/"


class TestGraphDoc:

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_schema_doc_prompt(self, gd: GraphDoc, request):
        """
        file_name = "test_schema_doc_prompt.pkl"
        cache_pick_file = {
            "schema_doc_prompt": response,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_schema_doc_prompt.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        if os.path.exists(test_cache_file_path) and dry_fire:
            with open(test_cache_file_path, "rb") as f:
                response = pickle.load(f)
                response = response["schema_doc_prompt"]

        #################### testing ####################
        if os.path.exists(CACHE_DIR) and fire:
            response = gd.schema_doc_prompt(database_schema="test schema")
            cache = {
                "schema_doc_prompt": response,
            }
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)
        log.debug(f"before parsing: {response}")
        assert response != None

    @pytest.mark.skipif(
        "not (config.getoption('--fire') or config.getoption('--dry-fire'))"
    )
    def test_schema_doc_prompt_from_file(self, gd: GraphDoc, request):
        """
        file_name = "test_schema_doc_prompt_from_file.pkl"
        cache_pick_file = {
            "schema_doc_prompt_from_file": response,
        }
        """
        #################### caching ####################
        fire = request.config.getoption("--fire")
        dry_fire = request.config.getoption("--dry-fire")
        test_cache_file_path = CACHE_DIR / "test_schema_doc_prompt_from_file.pkl"

        if fire and dry_fire:
            raise ValueError("Cannot use --fire and --dry-fire simultaneously")

        if os.path.exists(test_cache_file_path) and dry_fire:
            with open(test_cache_file_path, "rb") as f:
                response = pickle.load(f)
                response = response["schema_doc_prompt_from_file"]

        #################### testing ####################
        if os.path.exists(CACHE_DIR) and fire:
            response = gd.schema_doc_prompt_from_file(
                schema_file="opensea_original_schema.graphql"
            )
            cache = {
                "schema_doc_prompt_from_file": response,
            }
            with open(test_cache_file_path, "wb") as f:
                pickle.dump(cache, f)
        log.debug(f"before parsing: {response}")
        assert response != None
        assert response.prompt_cost != None

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_schema_doc_prompt_with_equality(self, gd: GraphDoc, request):
        """
        this is for us to be able to run the schema_doc_prompt_with_equality easily, but we will not include explicity tests for now.
        """
        database_schema = gd.parser.parse_schema_from_file(
            schema_file="opensea_original_schema.graphql"
        )

        database_schema = gd.parser.fill_empty_descriptions(database_schema)

        response = gd.schema_doc_prompt_with_equality(
            database_schema=print_ast(database_schema),
        )
        log.debug(f"The returned object is: {response}")
