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
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global Variables 
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#############################
# Internal Package Fixtures #
#############################
@fixture
def gd() -> GraphDoc: 
    return GraphDoc(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY
    )