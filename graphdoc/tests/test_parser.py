# system packages
import copy
import logging
from datetime import datetime
from typing import Dict, Any

# internal packages
from graphdoc import Prompt, PromptRevision, RequestObject
from graphdoc import Parser

# external packages
import pytest

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestParser:
    def test_parse_schema_from_file(self, par: Parser):
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema_from_file(schema_file)
        assert schema != None
        assert len(schema.definitions) == 9

    def test_check_schema_token_count(self, par: Parser):
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema_from_file(schema_file)
        token_count = par.check_schema_token_count(schema)
        logging.info(f"Token count: {token_count}")
        assert token_count == 1774

    def test_check_model_data(self, par: Parser):
        assert par.model_data != None
        assert "gpt-4o" in par.model_data
        assert par.model_data["gpt-4o"]["max_input_tokens"] == 128000

    def test_get_model_max_input_tokens(self, par: Parser):
        max_tokens = par.get_model_max_input_tokens()
        assert max_tokens == 128000

    def test_check_prompt_validity(self, par: Parser):
        prompt = "This is a test prompt"
        valid_prompt = par.check_prompt_validity(prompt)
        assert valid_prompt

    def test_update_node_descriptions(self, par: Parser): 
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema_from_file(schema_file)
        log.debug(f"Schema definitions: {schema}")
        updated_schema = par.update_node_descriptions(
            node=schema, 
            new_value="This is a test description"    
        )
        test_node_definition = updated_schema.definitions[4].fields[0].description.value
        assert test_node_definition == "This is a test description"

    def test_fill_empty_descriptions(self, par: Parser):
        schema_file = "opensea_original_schema_sparse.graphql"
        schema = par.parse_schema_from_file(schema_file)
        updated_schema = par.fill_empty_descriptions(schema)

        test_entity_definition_updated = updated_schema.definitions[3].fields[0].description.value
        test_entity_definition_updated_column = updated_schema.definitions[3].fields[0].name.value
        log.debug(f"Test entity node ({test_entity_definition_updated_column}) definition content: {test_entity_definition_updated}")
        assert test_entity_definition_updated == "Description for column: id"

        test_enum_definition_content = updated_schema.definitions[2].values[0].description.value
        test_enum_definition_content_column = updated_schema.definitions[2].values[0].name.value
        log.debug(f"Test enum node ({test_enum_definition_content_column}) definition updated: {test_enum_definition_content}")
        assert test_enum_definition_content == " Strategy that executes an order at a fixed price that can be taken either by a bid or an ask. "

        test_entity_description = updated_schema.definitions[3].description.value
        log.debug(f"Test entity description: {test_entity_description}")
        assert test_entity_description == "Description for table: Marketplace"