# system packages 
import logging
import argparse
import os
from pathlib import Path

# internal packages 
from graphdoc import GraphDoc
from graphdoc import load_yaml_config

# external packages 
import dspy
import mlflow

# logging 
logging.basicConfig(level=logging.INFO)
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
    mlflow_experiment_name = config["module"]["experiment_name"]

    # track the mlflow experiment
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # initialize the GraphDoc object
    gd = GraphDoc(
        model=lm_model_name,
        api_key=lm_api_key,
        hf_api_key=hf_api_key,
        cache=lm_cache,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    # load the doc generator module (must be from the mlflow)
    module = gd.doc_generator_module_from_mlflow(args.config_path, args.metric_config_path)

    # save module details 
    module_name = config["module"]["module_name"]
    save_module = config["module"]["save_module"]
    load_module = config["module"]["load_module"]
    mlflow_tracking_uri = config["trainer"]["mlflow_tracking_uri"]
    mlflow_tracking_uri = mlflow_tracking_uri.replace("file://", "")
    # mlflow_tracking_uri = "/Users/denver/Documents/code/graph/graphdoc/mlruns"
    mlflow_module_path = Path(mlflow_tracking_uri) / "modules" / module_name  

    with mlflow.start_run():
        log.info(f"Starting run {mlflow.active_run().info.run_id}")
        log.info(f"save_module: {save_module}")
        log.info(f"load_module: {load_module}")
        trainset = gd.dh.blank_trainset()
        if save_module:
            log.info(f"Saving module to {mlflow_module_path}")
            os.makedirs(mlflow_module_path, exist_ok=True)
            module.save(mlflow_module_path, True)
            mlflow.dspy.log_model(
                module,
                "doc_generator_module",
                input_example=trainset[0].toDict(),
                task=None,
            )
        if load_module:
            log.info(f"Loading module from {mlflow_module_path}")
            module = dspy.load(mlflow_module_path)
            # module = gd.fl.load_model_by_uri("/Users/denver/Documents/code/graph/graphdoc/mlruns/454536219902836119/f05ee299f6f84ee2971693eb155e5bf1/artifacts/doc_generator_module")
            # mlflow models serve -m /Users/denver/Documents/code/graph/graphdoc/mlruns/612951484798863978/233c983acf724d9fa983aacf8073de7d/artifacts/doc_generator_module -p 6000
            
            log.info(f"Module loaded {type(module)}: {module}")

        prediction = module.forward(trainset[0].database_schema)
        log.info("--------------------------------")
        log.info(f"Prediction: {prediction}")
        log.info("--------------------------------")


    # serve the module
    # module.serve(port=5000)


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