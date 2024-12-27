# system packages
from datetime import datetime
from typing import Dict, Any

# internal packages
from graphdoc import Prompt, PromptRevision, RequestObject 
from graphdoc import Parser

# external packages
import pytest

class TestParser: 
    def test_parse_schema(self, par: Parser): 
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema(schema_file)
        assert schema != None
        assert len(schema.definitions) == 9
            