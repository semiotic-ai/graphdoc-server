# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import Parser

# external packages
from graphql import DocumentNode
import pytest

# logging
# logging.basicConfig(level=logging.DEBUG)
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

    def test_fill_empty_descriptions(self, par: Parser):
        schema_file = "opensea_original_schema_sparse.graphql"
        schema = par.parse_schema_from_file(schema_file)
        updated_schema = par.fill_empty_descriptions(schema)
        definitions = getattr(updated_schema, "definitions", None)

        if definitions:
            test_entity_definition_updated = definitions[3].fields[0].description.value
            assert test_entity_definition_updated == "Description for column: id"

            test_enum_definition_content = definitions[2].values[0].description.value
            assert (
                test_enum_definition_content
                == " Strategy that executes an order at a fixed price that can be taken either by a bid or an ask. "
            )

            test_entity_description = definitions[3].description.value
            assert test_entity_description == "Description for table: Marketplace"

    def test_schema_equality_check(self, par: Parser):
        gold_schema_file = "opensea_original_schema.graphql"
        silver_schema_file = (
            "opensea_original_schema_sparse.graphql"  # only the comments are different
        )
        check_schema_file = "opensea_original_schema_modified.graphql"
        gold_schema = par.parse_schema_from_file(gold_schema_file)
        silver_schema = par.parse_schema_from_file(silver_schema_file)
        check_schema = par.parse_schema_from_file(check_schema_file)

        assert par.schema_equality_check(gold_schema, gold_schema)
        assert par.schema_equality_check(gold_schema, silver_schema)
        assert par.schema_equality_check(gold_schema, check_schema) is False
