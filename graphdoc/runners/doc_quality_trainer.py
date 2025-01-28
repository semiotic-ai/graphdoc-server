# system packages
import os

# internal packages
import logging
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt
from graphdoc import GraphDoc, DataHelper

# external packages
from dotenv import load_dotenv

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# Run Time Variables
MLFLOW_MODEL_NAME = "doc_quality_model_zero_shot"
MLFLOW_EXPERIMENT_NAME = "doc_quality_experiment_zero_shot"
MODEL = "openai/gpt-4o"
CACHE = True

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    gd = GraphDoc(model=MODEL, api_key=OPENAI_API_KEY, cache=CACHE)
    dh = DataHelper(hf_api_key=HF_DATASET_KEY)
    dataset = dh._folder_to_dataset(category="perfect", parse_objects=True)
    trainset = dh._create_graph_doc_example_trainset(dataset=dataset)
    
    doc_quality_prompt = DocQualityPrompt(
        type="chain_of_thought",
        metric_type="rating",
    )
    doc_quality_trainer = DocQualityTrainer(
        prompt=doc_quality_prompt,
        optimizer_type="miprov2",
        mlflow_model_name=MLFLOW_MODEL_NAME,
        mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
        trainset=trainset,
        evalset=trainset
    )
    doc_quality_trainer.run_training()
