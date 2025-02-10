# system packages
import os
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser
from graphdoc import DataHelper
from graphdoc import FlowLoader

# external packages
from graphdoc.generate import DocGeneratorEval
from pytest import fixture
from dotenv import load_dotenv
from dspy import Example

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")
CACHE = True

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCHEMA_DIR = BASE_DIR / "graphdoc" / "tests" / "assets" / "schemas"
MLFLOW_DIR = Path(BASE_DIR) / "graphdoc" / "tests" / "assets" / "mlruns"


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
    if OPENAI_API_KEY and HF_DATASET_KEY:
        return GraphDoc(
            model="openai/gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            hf_api_key=HF_DATASET_KEY,
            cache=CACHE,
        )
    else:
        log.warning(
            "Missing OPENAI_API_KEY or HF_DATASET_KEY. Ensure .env is properly set."
        )
        return GraphDoc(
            model="openai/gpt-4o-mini",
            api_key="filler api key",
            hf_api_key="filler api key",
            cache=CACHE,
        )


@fixture
def par() -> Parser:
    schema_directory_path = BASE_DIR / "graphdoc" / "tests" / "assets" / "schemas"
    return Parser(schema_directory_path=str(schema_directory_path))


@fixture
def dh() -> DataHelper:
    log.debug(f"HF_DATASET_KEY: {HF_DATASET_KEY}")
    return DataHelper(hf_api_key=HF_DATASET_KEY, schema_directory_path=str(SCHEMA_DIR))


@fixture
def dge() -> DocGeneratorEval:
    dh = DataHelper(hf_api_key=HF_DATASET_KEY)
    return DocGeneratorEval(dh)


@fixture
def trainset(dh: DataHelper) -> list[Example]:
    graphdoc_ds = dh._folder_of_folders_to_dataset(parse_objects=False)
    examples = dh._create_graph_doc_example_trainset(graphdoc_ds)
    return examples

@fixture
def fl() -> FlowLoader:
    return FlowLoader(mlflow_tracking_uri=MLFLOW_DIR)