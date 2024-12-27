# system packages
import logging
from datetime import datetime
from typing import Dict, Any

# internal packages
from graphdoc import Prompt, PromptRevision, RequestObject 
from graphdoc import Parser

# external packages
import pytest

logging.basicConfig(level=logging.INFO)

class TestParser: 
    def test_parse_schema(self, par: Parser): 
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema(schema_file)
        assert schema != None
        assert len(schema.definitions) == 9
    
    def test_check_schema_token_count(self, par: Parser): 
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema(schema_file)
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