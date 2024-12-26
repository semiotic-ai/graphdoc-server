# internal packages
import os
import json
from typing import Dict
from pathlib import Path
from datetime import datetime

# external packages
from pytest import fixture
from dotenv import load_dotenv
load_dotenv(".env")

# internal packages
from graphdoc import GraphDoc, LanguageModel, OpenAILanguageModel
from graphdoc import Prompt, PromptRevision, RequestObject 
from graphdoc import PromptExecutor, EntityComparisonPromptExecutor

####################
# Config Fixtures
####################
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
        help="Load locally saved data instead of making API calls"
    )

@fixture
def fire(request):
    return request.config.getoption("--fire")

@fixture
def dry_fire(request):
    return request.config.getoption("--dry-fire")

#################### 
# Object Fixtures
####################

@fixture
def lm() -> OpenAILanguageModel:
    return OpenAILanguageModel(
        api_key = os.getenv("OPENAI_API_KEY"),
    )

@fixture
def gd(lm: OpenAILanguageModel) -> GraphDoc:
    return GraphDoc(
        language_model = lm,
    )

@fixture 
def pe(lm: OpenAILanguageModel) -> PromptExecutor: 
    return PromptExecutor(
        language_model = lm,
    )

@fixture
def ecpe(lm: OpenAILanguageModel) -> EntityComparisonPromptExecutor: 
    return EntityComparisonPromptExecutor(
        language_model = lm,
    )

@fixture
def entity_comparison_assets() -> Dict: 
    # Set the absolute or relative path for the assets directory
    assets_dir = Path(__file__).parent/ 'assets/'
    assets_dir = Path(assets_dir)
    if not assets_dir.exists():
        raise FileNotFoundError(f"assets directory not found at: {assets_dir}")
    
    with open(Path(assets_dir / 'entity_comparison_assets.json'), 'r') as f:
        entity_comparison_assets = json.load(f)

    gold_entity_comparison = "".join(entity_comparison_assets["gold"]["prompt"])
    four_entity_comparison = "".join(entity_comparison_assets["four"]["prompt"])
    three_entity_comparison = "".join(entity_comparison_assets["three"]["prompt"])
    two_entity_comparison = "".join(entity_comparison_assets["two"]["prompt"])
    one_entity_comparison = "".join(entity_comparison_assets["one"]["prompt"])
    return {
        "gold_entity_comparison": gold_entity_comparison,
        "four_entity_comparison": four_entity_comparison,
        "three_entity_comparison": three_entity_comparison,
        "two_entity_comparison": two_entity_comparison,
        "one_entity_comparison": one_entity_comparison,
    }

@fixture
def sample_request_object() -> RequestObject:
    return RequestObject(
        prompt="Test prompt",
        response="Test response",
        model="gpt-4",
        prompt_tokens=10,
        response_tokens=20,
        request_time=int(datetime.now().timestamp()),
        request_id="test_123",
        request_object=None
    )

@fixture
def sample_prompt() -> Prompt:
    return Prompt(
        title="Test Prompt",
        base_content="This is a base prompt content",
        metadata={"type": "test"}
    )