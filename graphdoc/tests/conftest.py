# system packages
import os
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser
from graphdoc import DataHelper

# external packages
from pytest import fixture
from dotenv import load_dotenv

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


#############################
# Internal Package Fixtures #
#############################
def pytest_addoption(parser):
    parser.addoption(
        "--fire",
        action="store_true",
        default=False,
        help="Make external API calls and save responses locally",
    )

    parser.addoption(
        "--dry-fire",
        action="store_true",
        default=False,
        help="Load locally saved data instead of making API calls",
    )

    parser.addoption(
        "--write",
        action="store_true",
        default=False,
        help="Make external write call",
    )

    parser.addoption(
        "--run-evaluator",
        action="store_true",
        default=False,
        help="Run the evaluator",
    )


@fixture
def fire(request):
    return request.config.getoption("--fire")


@fixture
def dry_fire(request):
    return request.config.getoption("--dry-fire")


@fixture
def write(request):
    return request.config.getoption("--write")


@fixture
def run_evaluator(request):
    return request.config.getoption("--run-evaluator")


@fixture
def gd() -> GraphDoc:
    if OPENAI_API_KEY:
        return GraphDoc(model="openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    else:
        log.warning("Missing OPENAI_API_KEY. Ensure .env is properly set.")
        return GraphDoc(model="openai/gpt-4o-mini", api_key="filler api key")


@fixture
def par() -> Parser:
    schema_directory_path = BASE_DIR / "graphdoc" / "tests" / "assets" / "schemas"
    return Parser(schema_directory_path=str(schema_directory_path))


@fixture
def dh() -> DataHelper:
    log.debug(f"HF_DATASET_KEY: {HF_DATASET_KEY}")
    return DataHelper(hf_api_key=HF_DATASET_KEY)
