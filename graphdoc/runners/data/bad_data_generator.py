# system packages
import os
import argparse

# internal packages
import logging
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt
from graphdoc.modules import DocGeneratorModule
from graphdoc import GraphDoc, DataHelper, load_yaml_config

# external packages
from dotenv import load_dotenv

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

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
    lm_temperature = config["language_model"]["temperature"]
    lm_max_tokens = config["language_model"]["max_tokens"]
    log.info(f"caching level: {lm_cache}")

    gd = GraphDoc(
        model=lm_model_name,
        api_key=lm_api_key,
        hf_api_key=HF_DATASET_KEY,
        cache=lm_cache,
        temperature=lm_temperature,
        max_tokens=lm_max_tokens,
    )

    doc_generator_prompt = gd._get_nested_single_prompt(
        config_path=args.config_path,
        metric_config_path=args.metric_config_path,
    )

    dataset = gd.dh._folder_to_dataset(category="almost perfect", parse_objects=False)
    dataset_files = dataset.to_pandas()
    dataset_files = dataset_files["schema_name"].tolist()
    new_file_names = [x.replace("_3", "_1") for x in dataset_files]
    print(dataset_files)
    print(new_file_names)
    trainset = gd.dh._create_doc_generator_example_trainset(dataset)

    dgm = DocGeneratorModule(generator_prompt=doc_generator_prompt)
    print(type(dgm))

    predictions = []
    for i in range(len(trainset)):
        print(f"Documenting schema: {dataset_files[i]}")
        prediction = dgm.document_full_schema(
            database_schema=trainset[i].database_schema
        )
        # predictions.append(prediction.documented_schema)
        # print(prediction.documented_schema)
        with open(f"{new_file_names[i]}.graphql", "w") as f:
            f.write(prediction.documented_schema)

    # for file_name in new_file_names:
    #     with open(f"{file_name}.graphql", "w") as f:
    #         f.write(predictions[0])
