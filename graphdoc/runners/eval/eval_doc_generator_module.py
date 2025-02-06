# system packages
import os
import logging
import argparse
import random

# internal packages
from graphdoc.loader.helper import load_dspy_model
from graphdoc.modules.schema_doc_generator import DocGeneratorModule
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt, DocGeneratorPrompt
from graphdoc import GraphDoc, DataHelper, load_yaml_config

# external packages
from dotenv import load_dotenv
from runners.train.doc_generator_trainer import get_prompt_signature

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
    
    # data
    dataset = gd.dh._folder_of_folders_to_dataset(parse_objects=False) 
    log.info(f"dataset size: {len(dataset)}")

    split = dataset.train_test_split(0.1)
    trainset = gd.dh._create_doc_generator_example_trainset(split["train"])
    evalset = gd.dh._create_doc_generator_example_trainset(split["test"])
    random.Random(0).shuffle(trainset)
    random.Random(0).shuffle(evalset)
    trainset = trainset[:2]
    evalset = evalset[:1]
    log.info(f"trainset size: {len(trainset)}")
    log.info(f"evalset size: {len(evalset)}")

    
    # TODO: all of this will be refactored to enable loading from config for specific mlflow model
    # load the gen prompt
    doc_generator_prompt = gd._get_nested_single_prompt(
        config_path=args.config_path,
        metric_config_path=args.metric_config_path,
    )

    # load the most recent version of the doc_quality_prompt and set as the metrci 
    metric_prompt = load_dspy_model( # this loads an initialized model (CoT, etc.)
        model_name=metric_config["trainer"]["mlflow_model_name"],
        latest_version=True
    )

    # initialize the DocQualityPrompt object
    metric_signature = get_prompt_signature(metric_prompt)
    dqp = DocQualityPrompt(
        type=doc_generator_prompt.metric_type.type,
        metric_type=doc_generator_prompt.metric_type.metric_type,  # type: ignore
        prompt=metric_signature
    )

    # set the metric type
    doc_generator_prompt.metric_type = dqp

    # print out the set metric prompt to check
    test_metric_signature = get_prompt_signature(doc_generator_prompt.metric_type.infer)
    base_prompt = gd.dh.par.format_signature_prompt(
            signature=test_metric_signature, signature_type="doc_generation"
    )
    log.info(f"using metric prompt: {base_prompt}")

    # load in the most recent version of doc_generator_prompt
    generator_prompt = load_dspy_model( # this loads an initialized model (CoT, etc.)
        model_name=config["trainer"]["mlflow_model_name"],
        latest_version=True
    )

    # initialize the DocQualityPrompt object
    generator_signature = get_prompt_signature(generator_prompt)
    dgp = DocGeneratorPrompt(
        metric_type=doc_generator_prompt.metric_type,  # type: ignore
        type=doc_generator_prompt.type,
        prompt=generator_signature
    )

    test_generator_signature = get_prompt_signature(dgp.infer)
    base_gen_prompt = gd.dh.par.format_signature_prompt(
            signature=test_generator_signature, signature_type="doc_generation"
    )
    log.info(f"using generator prompt: {base_gen_prompt}")


    # init the DocGeneratorModule
    dgm = DocGeneratorModule(generator_prompt=doc_generator_prompt, retry=True)
