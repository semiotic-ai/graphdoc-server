# system packages
import os
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser

# external packages
from pytest import fixture
from dotenv import load_dotenv

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

#############################
# Internal Package Fixtures #
#############################
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
