# system packages
import os
import yaml
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc, DataHelper, DocQualityEval, DocQuality

# external packages
import dspy
import mlflow
import datasets
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Environment Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# Run Time Variables
EXPERIMENT_NAME = "eval_model_training"
DSPY_MODEL_NAME = "DocQuality_zero_shot"
MODEL = "openai/gpt-4o-mini"
CACHE = True
RUN = False
TRAIN = True
EVALUATE = True

SCRIPT_TEST = True

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# MLFlow 
mlflow.dspy.autolog()
mlflow.set_experiment(EXPERIMENT_NAME)
ml_client = mlflow.MlflowClient()

# set up DSPy and GraphDoc components
dh = DataHelper(hf_api_key=HF_DATASET_KEY)
dqe = DocQualityEval(dh)

lm = dspy.LM(model=MODEL, api_key=OPENAI_API_KEY, cache=CACHE)
dspy.configure(lm=lm)
classify = dspy.Predict(DocQuality)

# load in the dataset
dataset = dh._folder_to_dataset(category="perfect", parse_objects=True)
trainset = dh._create_graph_doc_example_trainset(dataset=dataset)

# sample 50% of the dataset for training
train_split = trainset[:(len(trainset) // 2)]
test_split = trainset[(len(trainset) // 2):]

# get an example to use to infer the signature
example = trainset[0].toDict()
example.pop('category')
example.pop('rating')
print(example)

# infer the signature based on the example
signature = infer_signature(example)
print(signature)

# for now, set the version of the dataset 
# dataset.version = datasets.Version("1.0.0")
print(f"The dataset version is {dataset.version}")

try: 
    latest_version = ml_client.get_latest_versions(DSPY_MODEL_NAME)
    print(f"The latest version of the model is {latest_version}")
    loaded_model = mlflow.dspy.load_model(latest_version[0].source)
    print(f"The loaded model is {loaded_model}")

    # model_metadata_yaml = Path(latest_version[0].source).joinpath("MLmodel")
    # with open(model_metadata_yaml, "r") as file: 
    #     print(file.read())
    #     metadata = yaml.safe_load(file)
    # # print(f"The metadata is {metadata}")

    model = loaded_model
except Exception as e: 
    print(f"No model found in MLFlow, creating a new one")
    # log the model to MLFlow
    model_info = mlflow.dspy.log_model(
        dspy_model=classify, 
        artifact_path="model",
        signature=signature,
        task=None,
        registered_model_name=DSPY_MODEL_NAME,
        metadata={"trainset": "semiotic/graphdoc_schemas", "trainset_version": str(datasets.Version("1.0.0"))} # establish versioning for trainset (commit SHA)
    )
    print(f"The model was logged to MLFlow as {model_info.model_uri}")

    model = classify

if TRAIN: 
    tp = dspy.MIPROv2(metric=dqe.validate_rating, auto="light")
    optimized_model = tp.compile(
        model, trainset=train_split, max_labeled_demos=0, max_bootstrapped_demos=0
    )

if EVALUATE: 
    pred_score = 0
    og_score = 0
    i = 0
    for example in test_split: 
        om_pred = optimized_model(database_schema=example.database_schema)
        og_pred = model(database_schema=example.database_schema)

        if dqe.validate_rating(example, om_pred):
            print(f"The optimized model predicted correctly on example {i}")
            pred_score += 1
        else: 
            print(f"The optimized model predicted incorrectly on example {i}")

        if dqe.validate_rating(example, og_pred):
            print(f"The original model predicted correctly on example {i}")
            og_score += 1
        else: 
            print(f"The original model predicted incorrectly on example {i}")
        print("--------------------------------")
        i += 1

    print(f"The optimized model scored {pred_score} out of {len(test_split)}")
    print(f"The original model scored {og_score} out of {len(test_split)}")

    # set and log expirement details 
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.log_params({"trainset": "semiotic/graphdoc_schemas", "trainset_version": str(datasets.Version("1.0.0")), "model_name": DSPY_MODEL_NAME})
    mlflow.log_metrics({"optimized_model_score": str(round(pred_score / len(test_split), 2)), "original_model_score": str(round(og_score / len(test_split), 2))})

if SCRIPT_TEST: 
    pred_score = og_score + 1

if pred_score > og_score: 
    print("The optimized model performed better than the original model")
    optimized_model_info = mlflow.dspy.log_model(
        dspy_model=optimized_model, 
        artifact_path="model",
        signature=signature,
        task=None,
        registered_model_name=DSPY_MODEL_NAME,
        metadata={"trainset": "semiotic/graphdoc_schemas", "trainset_version": str(datasets.Version("1.0.0"))} # establish versioning for trainset (commit SHA)
    )
    print(f"The model was logged to MLFlow as {optimized_model_info.model_uri}")


# 2. Training / Evaluation pipeline 
#     - [x] load in a given MLFlow model
#     - [ ] assess if it was trained on our most recent dataset
#     - [ ] if not trained, train it on our most recent dataset
#     - [x] assess performance of the trained model on our most recent dataset
#     - [x] store the training and evaluation results to MLFlow 
#     - [x] save the trained model to MLFlow