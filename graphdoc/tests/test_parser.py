# system packages
import logging
from pathlib import Path

# internal packages

# external packages
from graphql import DocumentNode
import pytest

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestParser:

    def test_parse_schema_from_file(self, par):
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
