import pytest
from graphdoc_server.app import create_app
import os
from pathlib import Path


@pytest.fixture
def app():
    """Create application for the tests."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent

    # Set test configs with absolute paths
    os.environ["GRAPHDOC_CONFIG_PATH"] = str(
        project_root
        / "assets"
        / "configs"
        / "single_prompt_schema_doc_generator_module.yaml"
    )
    os.environ["GRAPHDOC_METRIC_CONFIG_PATH"] = str(
        project_root
        / "assets"
        / "configs"
        / "single_prompt_schema_doc_quality_trainer.yaml"
    )

    app = create_app()
    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create a test runner for the app's CLI commands."""
    return app.test_cli_runner()
