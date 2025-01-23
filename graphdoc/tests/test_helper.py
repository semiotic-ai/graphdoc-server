# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import check_directory_path, check_file_path

# external packages
import pytest

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestHelper:

    def test_check_directory_path(self):
        with pytest.raises(ValueError):
            check_directory_path("invalid_path")

        schema_directory_path = BASE_DIR / "graphdoc" / "tests" / "assets" / "schemas"
        assert check_directory_path(str(schema_directory_path)) is None

    def test_check_file_path(self):
        with pytest.raises(ValueError):
            check_file_path("invalid_path")

        schema_file_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "schemas"
            / "opensea_original_schema.graphql"
        )
        assert check_file_path(str(schema_file_path)) is None
