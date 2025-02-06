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
import mlflow
from dotenv import load_dotenv

# logging
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

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

    # data
    dataset = gd.dh._folder_of_folders_to_dataset()
    log.info(f"dataset size: {len(dataset)}")

    split = dataset.train_test_split(0.1)
    trainset = gd.dh._create_doc_generator_example_trainset(split["train"])
    evalset = gd.dh._create_doc_generator_example_trainset(split["test"])
    random.Random(0).shuffle(trainset)
    random.Random(0).shuffle(evalset)
    trainset = trainset[:2]
    evalset = evalset[:2]

    log.info(f"trainset size: {len(trainset)}")
    log.info(f"evalset size: {len(evalset)}")

    # prompt
    doc_generator_prompt = gd._get_nested_single_prompt(
        config_path=args.config_path,
        metric_config_path=args.metric_config_path,
    )

    # trainer
    doc_generator_trainer = gd._get_single_trainer(
        config_path=args.config_path,
        trainset=trainset,
        evalset=evalset,
        prompt=doc_generator_prompt,
    )   

    doc_generator_trainer.run_training()