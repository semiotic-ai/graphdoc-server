# system packages
import logging
from typing import Literal

# internal packages
import os
from graphdoc import GraphDoc, DataHelper, DocQualityEval, DocQuality

# external packages
import dspy
from dspy import Example
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    dh = DataHelper(hf_api_key=HF_DATASET_KEY)
    dqe = DocQualityEval(dh)

    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
    dspy.configure(lm=lm)
    classify = dspy.Predict(DocQuality)

    dataset = dh._folder_of_folders_to_dataset(parse_objects=False)
    trainset = dh._create_graph_doc_example_trainset(dataset=dataset)

    evaluator = dqe.create_evaluator()  # trainset=trainset
    dqe.run_evaluator(evaluator, classify, dqe.validate_rating)
