# system packages
import logging
import os
from pathlib import Path

# internal packages
from graphdoc import check_directory_path, check_file_path, load_yaml_config

# external packages
import pytest

# logging
# logging.basicConfig(level=logging.DEBUG)
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

    def test_load_yaml_config(self):
        config_path = (
            BASE_DIR
            / "graphdoc"
            / "tests"
            / "assets"
            / "configs"
            / "single_prompt_trainer.yaml"
        )
        config = load_yaml_config(str(config_path))
        assert config is not None
        assert config["language_model"]["lm_api_key"] is not None
        assert config["language_model"]["lm_api_key"] == os.getenv("OPENAI_API_KEY")
