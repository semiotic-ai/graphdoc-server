# system packages
import copy
import os
import logging
import argparse
import random

# internal packages
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt
from graphdoc import GraphDoc, DataHelper, load_yaml_config
from graphdoc.loader import load_dspy_model

# external packages
import dspy
import mlflow
from dotenv import load_dotenv

# logging
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")


def get_prompt_signature(prompt) -> dspy.Signature:
    if isinstance(prompt, dspy.ChainOfThought):
        return prompt.predict.signature
    elif isinstance(prompt, dspy.Predict):
        return prompt.signature
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")


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
    mlflow_load_model = config["trainer"][
        "mlflow_load_model"
    ]  # : true # Whether to load the most recent model from MLflow

    gd = GraphDoc(
        model=lm_model_name,
        api_key=lm_api_key,
        hf_api_key=HF_DATASET_KEY,
        cache=lm_cache,
    )

    # data
    def filter_by_category(example):
        return example["category"] in ["perfect", "almost perfect"]

    dataset = gd.dh._folder_of_folders_to_dataset()
    dataset = dataset.filter(filter_by_category)
    log.info(f"dataset size: {len(dataset)}")

    split = dataset.train_test_split(0.2)
    trainset = gd.dh._create_doc_generator_example_trainset(split["train"])
    evalset = gd.dh._create_doc_generator_example_trainset(split["test"])
    random.Random(0).shuffle(trainset)
    random.Random(0).shuffle(evalset)
    log.info(f"trainset size: {len(trainset)}")
    log.info(f"evalset size: {len(evalset)}")
    log.info(f"train example: {trainset[0]}")
    log.info(f"eval example: {evalset[0]}")

    # prompt
    doc_generator_prompt = gd._get_nested_single_prompt(
        config_path=args.config_path,
        metric_config_path=args.metric_config_path,
    )

    # load the most recent version of the doc_quality_prompt and set as the metrci
    metric_prompt = load_dspy_model(  # this loads an initialized model (CoT, etc.)
        model_name=metric_config["trainer"]["mlflow_model_name"], latest_version=True
    )

    # initialize the DocQualityPrompt object
    metric_signature = get_prompt_signature(metric_prompt)
    dqp = DocQualityPrompt(
        type=doc_generator_prompt.metric_type.type,
        metric_type=doc_generator_prompt.metric_type.metric_type,  # type: ignore
        prompt=metric_signature,
    )

    # set the metric type
    doc_generator_prompt.metric_type = dqp

    # print out the set metric prompt to check
    test_metric_signature = get_prompt_signature(doc_generator_prompt.metric_type.infer)
    base_prompt = gd.dh.par.format_signature_prompt(
        signature=test_metric_signature, signature_type="doc_generation"
    )
    log.info(f"using metric prompt: {base_prompt}")

    # trainer
    doc_generator_trainer = gd._get_single_trainer(
        config_path=args.config_path,
        trainset=trainset,
        evalset=evalset,
        prompt=doc_generator_prompt,
    )

    # make sure we don't log keys
    report_config = copy.deepcopy(config)
    report_config["language_model"]["lm_api_key"] = "REDACTED"
    report_config["data"]["hf_api_key"] = "REDACTED"
    report_config["trainer"]["mlflow_tracking_uri"] = "REDACTED"
    mlflow.log_params(report_config)

    doc_generator_trainer.run_training(load_model=mlflow_load_model, save_model=True)
