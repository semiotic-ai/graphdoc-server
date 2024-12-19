# internal packages
import os
from typing import List

# external packages
from pytest import fixture
from dotenv import load_dotenv
load_dotenv(".env")

# internal packages
from graphdoc import GraphDoc, LanguageModel

@fixture
def lm() -> LanguageModel:
    return LanguageModel(
        api_key = os.getenv("OPENAI_API_KEY"),
    )

@fixture
def gd(lm: LanguageModel) -> GraphDoc:
    return GraphDoc(
        language_model = lm,
    )