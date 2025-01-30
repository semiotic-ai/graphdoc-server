# system packages
import os
import argparse

# internal packages
import logging
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt
from graphdoc import GraphDoc, DataHelper, load_yaml_config

# external packages
from dotenv import load_dotenv

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload local data to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-card",
        type=bool,
        required=False,
        help="Whether to create and upload a repo card.",
    )
    args = parser.parse_args()
    repo_card = args.repo_card
    log.info(f"Repo card upload: {repo_card}")

    model_name = "gpt-4o"
    api_key = OPENAI_API_KEY
    hf_api_key = HF_DATASET_KEY
    lm_cache = True

    gd = GraphDoc(
        model=model_name,
        api_key=api_key,
        hf_api_key=hf_api_key,
        cache=lm_cache,
    )

    gd.update_graphdoc_dataset(local_file=True, repo_card=repo_card)
