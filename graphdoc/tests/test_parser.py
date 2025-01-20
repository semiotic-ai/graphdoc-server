# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import Parser

# external packages
from graphql import DocumentNode
import pytest

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestParser:

    def test_parse_schema_from_file(self, par: Parser):
        schema_file = (
            Path(BASE_DIR)
            / "graphdoc"
            / "tests"
            / "assets"
            / "schemas"
            / "opensea_original_schema.graphql"
        )
        schema = par.parse_schema_from_file(schema_file)
        assert isinstance(schema, DocumentNode)

    def test_update_node_descriptions(self, par: Parser):
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema_from_file(schema_file)
        updated_schema = par.update_node_descriptions(
            node=schema, new_value="This is a test description"
        )
        for i in range(3, 6):
            for x in range(3):
                definitions = getattr(updated_schema, "definitions", None)
                if definitions:
                    test_node_definition = definitions[i].fields[x].description.value
                    assert test_node_definition == "This is a test description"
