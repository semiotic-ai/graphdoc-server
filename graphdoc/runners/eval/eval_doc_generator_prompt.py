# system packages
import os
import logging
import argparse
import random

# internal packages
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
    parser = argparse.ArgumentParser(description="Train a document quality model.")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--metric-config-path",
        type=str,
        required=True,
        help="Path to the metric configuration YAML file.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    metric_config = load_yaml_config(args.metric_config_path)

    lm_model_name = config["language_model"]["lm_model_name"]
    lm_api_key = config["language_model"]["lm_api_key"]
    lm_cache = config["language_model"]["cache"]

    gd = GraphDoc(
        model=lm_model_name,
        api_key=lm_api_key,
        hf_api_key=HF_DATASET_KEY,
        cache=lm_cache,
    )

    doc_generator_prompt = gd._get_nested_single_prompt(
        config_path=args.config_path,
        metric_config_path=args.metric_config_path,
    )

    # dataset = gd.dh._load_from_hf()
    # evalset = gd.dh._create_graph_doc_example_trainset(
    #     dataset["train"].train_test_split(0.2)["test"]
    # )
    schema_path = gd.dh._blank_schema_folder()
    schema_objects = gd.dh.schemas_folder(category="blank", rating="0", folder_path=schema_path)
    dataset = gd.dh._schema_objects_to_dataset(schema_objects, parse_objects=True)
    log.info(f"dataset size: {len(dataset)}")
    schema_type = "table schema"
    filtered_dataset = dataset.filter(lambda example: example["schema_type"] == schema_type)
    evalset = gd.dh._create_doc_generator_example_trainset(filtered_dataset)

    log.info(f"evalset size: {len(evalset)}")

    results = doc_generator_prompt.evaluate_evalset(
        examples=evalset[:1],
    )

    log.info(results)
    log.info(results["results"][0][1].documented_schema)
