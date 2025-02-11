# system packages 
import logging
import argparse
import os

# internal packages 
from graphdoc import GraphDoc
from graphdoc import load_yaml_config

# external packages 

# logging 
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

    # load in the configs
    args = parser.parse_args()
    config = load_yaml_config(args.config_path)
    metric_config = load_yaml_config(args.metric_config_path)

    # set the variables
    lm_model_name = config["language_model"]["lm_model_name"]
    lm_api_key = config["language_model"]["lm_api_key"]
    lm_cache = config["language_model"]["cache"]
    mlflow_tracking_uri = config["trainer"]["mlflow_tracking_uri"]
    hf_api_key = config["data"]["hf_api_key"]

    # initialize the GraphDoc object
    gd = GraphDoc(
        model=lm_model_name,
        api_key=lm_api_key,
        hf_api_key=hf_api_key,
        cache=lm_cache,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    # load the doc generator module (must be from the mlflow)
    module = gd.doc_generator_module(args.config_path, args.metric_config_path)

    # save module details 
    module_name = config["module"]["module_name"]
    save_module = config["module"]["save_module"]

    if save_module:
        os.makedirs("test", exist_ok=True)
        gd.save("test", True)


    # load the doc generator module
    # module = gd.doc_generator_module_from_mlflow(args.config_path, args.metric_config_path)

    # def test_doc_generator_module_from_mlflow(self, gd: GraphDoc):
    #     config_path = (
    #         BASE_DIR
    #         / "graphdoc"
    #         / "tests"
    #         / "assets"
    #         / "configs"
    #         / "single_prompt_schema_doc_generator_trainer.yaml"
    #     )
    #     metric_config_path = (
    #         BASE_DIR
    #         / "graphdoc"
    #         / "tests"
    #         / "assets"
    #         / "configs"
    #         / "single_prompt_schema_doc_quality_trainer.yaml"
    #     )
    #     module = gd.doc_generator_module(config_path, metric_config_path)
    #     assert isinstance(module, DocGeneratorModule)

    # return GraphDoc(
    #     model="openai/gpt-4o-mini",
    #     api_key=OPENAI_API_KEY,
    #     hf_api_key=HF_DATASET_KEY,
    #     cache=CACHE,
    #     mlflow_tracking_uri=MLFLOW_DIR,
    # )