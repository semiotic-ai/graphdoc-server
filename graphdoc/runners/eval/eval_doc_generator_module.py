# system packages
import math
import os
import logging
import argparse
import random
import shutil

# internal packages
from graphdoc.loader.helper import load_dspy_model
from graphdoc.modules.schema_doc_generator import DocGeneratorModule
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt, DocGeneratorPrompt
from graphdoc import GraphDoc, DataHelper, load_yaml_config

# external packages
from dotenv import load_dotenv
from graphql import parse, print_ast
import dspy
import mlflow
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
    mlflow_tracking_uri = config["trainer"]["mlflow_tracking_uri"]

    gd = GraphDoc(
        model=lm_model_name,
        api_key=lm_api_key,
        hf_api_key=HF_DATASET_KEY,
        cache=lm_cache,
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("DocGenModule Eval")
    mlflow.dspy.autolog()

    schema_path = gd.dh._blank_schema_folder()
    schema_objects = gd.dh.schemas_folder(category="blank", rating="0", folder_path=schema_path)
    dataset = gd.dh._schema_objects_to_dataset(schema_objects, parse_objects=False)
    log.info(f"dataset size: {len(dataset)}")
    evalset = gd.dh._create_doc_generator_example_trainset(dataset)
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

    # 
    # Create a directory for evaluation results and overwrite it if it exists
    eval_results_dir = "eval_results"
    if os.path.exists(eval_results_dir):
        shutil.rmtree(eval_results_dir)
    os.makedirs(eval_results_dir)

    # Create subdirectories for examples and predictions
    example_dir = os.path.join(eval_results_dir, "example")
    pred_dir = os.path.join(eval_results_dir, "pred")
    os.makedirs(example_dir)
    os.makedirs(pred_dir)

    with mlflow.start_run():
        # TODO: log parameters (when we have configured DocGenModule to be able to load abitrarily from mlflow)
        # TODO: all of this needs to be refactored
        eval = evalset[0]

        avg_component_rating = []
        doc_rating = []
        for i in range(len(evalset)): 
            eval = evalset[i]
            pred = dgm.document_full_schema(database_schema=eval.database_schema)
            pred_ast = parse(pred.documented_schema)
            exs = []
            ratings = []
            for node in pred_ast.definitions: 
                ex = dspy.Example(
                    database_schema=print_ast(node)
                ).with_inputs("database_schema")
                p = dspy.Prediction(
                    database_schema=print_ast(node),
                    documented_schema=print_ast(node)
                ).with_inputs("database_schema")
                rating_pred = dgm.generator_prompt.evaluate_documentation_quality(ex, p)
                rating = math.sqrt(rating_pred) * 25
                if rating == 25:
                    log.info(f"Rating of 1 being returned, setting to 0") 
                    rating = 0 
                ratings.append(rating)
            overal_rating = dgm.generator_prompt.evaluate_documentation_quality(
                dspy.Example(
                    database_schema=eval.database_schema
                ).with_inputs("database_schema"),
                dspy.Prediction(
                    database_schema=eval.database_schema,
                    documented_schema=pred.documented_schema
                ).with_inputs("database_schema")
            )
            average_rating = sum(ratings) / len(ratings)
            log.info(f"Average component score: {average_rating}")
            log.info(f"Overal document rating: {math.sqrt(overal_rating)}")
            # mlflow.log_metric("average_component_rating", average_rating)
            # mlflow.log_metric("overal_document_rating", math.sqrt(overal_rating))
            avg_component_rating.append(average_rating)
            doc_rating.append(math.sqrt(overal_rating))

            with open(example_dir + f"example_{i}.graphql", "w") as f: 
                f.write(eval.database_schema)
            with open(pred_dir + f"pred_{i}.graphql", "w") as f: 
                f.write(pred.documented_schema)
        
        log.info(f"doc component ratings: {avg_component_rating}")
        log.info(f"doc overal ratings: {doc_rating}")
        mlflow.log_dict({"doc_component_ratings": avg_component_rating}, "doc_component_ratings.json")
        mlflow.log_dict({"doc_overal_ratings": doc_rating}, "doc_overal_ratings.json")

        mlflow.log_metric(f"average_component_rating", sum(avg_component_rating) / len(avg_component_rating))
        mlflow.log_metric(f"overal_document_rating", sum(doc_rating) / len(doc_rating))
