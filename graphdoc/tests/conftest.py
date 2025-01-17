# system packages
import os
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality

# external packages
from pytest import fixture
from dotenv import load_dotenv

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#############################
# Internal Package Fixtures #
#############################
@fixture
def gd() -> GraphDoc:
    if OPENAI_API_KEY:
        return GraphDoc(model="openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    else:
        log.warn("Missing OPENAI_API_KEY. Ensure .env is properly set.")
        return GraphDoc(model="openai/gpt-4o-mini", api_key="filler api key")
