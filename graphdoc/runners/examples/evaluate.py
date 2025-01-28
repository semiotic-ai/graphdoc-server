# system packages
import logging
from typing import Literal

# internal packages
import os
from graphdoc import GraphDoc, DataHelper, DocQualityEval, DocQuality

# external packages
import dspy
from dspy import ChatAdapter
from dspy import Example
from dspy.evaluate import Evaluate
from dotenv import load_dotenv
import mlflow

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# Run Time Variables
EVALUATE = True
OPTIMIZE = False
MODEL = "openai/gpt-4o-mini"
CACHE = False

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
mlflow.dspy.autolog()
mlflow.set_experiment("DSPy_1")  # failing

if __name__ == "__main__":
    dh = DataHelper(hf_api_key=HF_DATASET_KEY)
    dqe = DocQualityEval(dh)
    ca = ChatAdapter()

    lm = dspy.LM(model=MODEL, api_key=OPENAI_API_KEY, cache=CACHE)
    dspy.configure(lm=lm)
    classify = dspy.Predict(DocQuality)

    # dataset = dh._folder_of_folders_to_dataset(parse_objects=True)
    dataset = dh._folder_to_dataset(category="perfect", parse_objects=True)
    trainset = dh._create_graph_doc_example_trainset(dataset=dataset)
    evaluator = dqe.create_evaluator(trainset=trainset)

    os.makedirs(f"modules", exist_ok=True)
    classify.save(f"modules/baseline_document_classifier.json", save_program=False)

    if EVALUATE:
        dqe.run_evaluator(evaluator, classify, dqe.validate_rating)

    if OPTIMIZE:
        tp = dspy.MIPROv2(metric=dqe.validate_rating, auto="light")
        optimized_evaluator = tp.compile(
            classify, trainset=trainset, max_labeled_demos=0, max_bootstrapped_demos=0
        )
        optimized_evaluator.save(
            f"modules/optimized_document_classifier.json", save_program=False
        )


# TrainerRunner
# def __init__(self,
# prompt
# metric:
# dataset: train / eval
# logging: - always log to MLFlow: keep this hardcoded for MLFlow for now

# hard code and assume we are using the right dataset

# initialize the trainer

# run the trainer
# log the results to MLFlow

# evaluate the output on our eval dataset
# log the results to MLFlow

# save the model to MLFlow


# we will avoid this for now and maybe revisit it when we need another implementation not using dspy
# TrainerDSPy(TrainerRunner)
# def __init__(self,
# module:
# metric: (function)
# dataset: train / eval
# kwargs: this could be for
# not opposed to rewrapping, as we may drop here and want to integrate with something like LangChain
# we want to not be attached to dspy
