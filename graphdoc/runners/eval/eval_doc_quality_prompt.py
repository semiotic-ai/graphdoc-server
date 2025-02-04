# system packages
import os
import random
import argparse

# internal packages
import logging
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt
from graphdoc import GraphDoc, DataHelper, load_yaml_config

# external packages
import mlflow
from dotenv import load_dotenv

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")
LOCAL_SAVE_RESULTS = True

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
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    lm_model_name = config["language_model"]["lm_model_name"]
    lm_api_key = config["language_model"]["lm_api_key"]
    lm_cache = config["language_model"]["cache"]
    mlflow_load_model = config["trainer"][
        "mlflow_load_model"
    ]  # : true # Whether to load the most recent model from MLflow

    gd = GraphDoc(
        model=lm_model_name,
        api_key=lm_api_key,
        hf_api_key=HF_DATASET_KEY,
        cache=lm_cache,
    )
    dh = DataHelper(hf_api_key=HF_DATASET_KEY)
    dataset = dh._folder_of_folders_to_dataset()
    log.info(f"dataset size: {len(dataset)}")

    # split = dataset["train"].train_test_split(0.2)
    split = dataset.train_test_split(0.1, seed=42)
    trainset = dh._create_graph_doc_example_trainset(split["train"])
    evalset = dh._create_graph_doc_example_trainset(split["test"])

    # shuffle
    random.seed(42)
    random.Random(0).shuffle(trainset)
    random.Random(0).shuffle(evalset)

    evalset = evalset[:10]

    log.info(f"trainset size: {len(trainset)}")
    log.info(f"evalset size: {len(evalset)}")

    doc_quality_trainer = gd._get_single_trainer(
        config_path=args.config_path,
        trainset=trainset,
        evalset=evalset,  # prompt: dspy.Signature
    )
    eval = doc_quality_trainer.prompt.evaluate_evalset(examples=evalset, num_threads=16)

    if LOCAL_SAVE_RESULTS:
        dh.par.folders_for_responses(eval["results"])

    # make sure we don't log keys
    config["language_model"]["lm_api_key"] = "REDACTED"
    config["data"]["hf_api_key"] = "REDACTED"
    config["trainer"]["mlflow_tracking_uri"] = "REDACTED"
    # mlflow.log_params(config)
