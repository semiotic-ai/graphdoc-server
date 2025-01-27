# system packages
import os
import logging

# internal packages
from graphdoc import GraphDoc, DataHelper, DocQualityEval, DocQuality

# external packages
import dspy
import mlflow
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Environment Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# Run Time Variables
EXPERIMENT_NAME = "eval_model_saving"
DSPY_MODEL_NAME = "DocQuality_zero_shot"
MODEL = "openai/gpt-4o-mini"
CACHE = True
FIRST_RUN = False

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

# get an example to use to infer the signature
example = trainset[0].toDict()
# drop the keys 'category' and 'rating'
example.pop('category')
example.pop('rating')
print(example)

# infer the signature based on the example
signature = infer_signature(example)
print(signature)

with mlflow.start_run():
    
    if FIRST_RUN:
        # log the model to MLFlow
        model_info = mlflow.dspy.log_model(
            dspy_model=classify, 
            artifact_path="model",
            signature=signature,
            task=None,
            registered_model_name=DSPY_MODEL_NAME,
            metadata={"trainset": "semiotic/graphdoc_schemas"} # establish versioning for trainset
        )
        print(f"The model was logged to MLFlow as {model_info.model_uri}")

        loaded_model = mlflow.dspy.load_model(model_info.model_uri)
        pred = loaded_model(database_schema=trainset[1].database_schema).rating
        print(f"The model predicted a rating of {pred}")
    
    else: 
        # get the latest version of the model
        latest_version = ml_client.get_latest_versions(DSPY_MODEL_NAME)
        print(f"The latest version of the model is {latest_version}")

        # load the latest model 
        loaded_model = mlflow.dspy.load_model(latest_version[0].source)
        pred = loaded_model(database_schema=trainset[1].database_schema).rating
        print(f"The model predicted a rating of {pred}")

# 1. Managing the DSPy / MLFlow relationship
#     - [x] Create a DSPy model
#     - [x] Create a MLFlow model that tracks to that DSPy model 
#     - [x] Save the DSPy model to MLFlow

# Notes 
# - we are running with a local environment setting, and will want to migrate to using a 
# - even with no changes, MLFlow will log the model as a new version
