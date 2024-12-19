# internal packages
import os
import json
from typing import Dict
from pathlib import Path

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