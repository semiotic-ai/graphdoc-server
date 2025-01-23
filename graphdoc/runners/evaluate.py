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

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# Run Time Variables
EVALUATE = False
OPTIMIZE = True

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    dh = DataHelper(hf_api_key=HF_DATASET_KEY)
    dqe = DocQualityEval(dh)
    ca = ChatAdapter()

    lm = dspy.LM(model="openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=True)
    dspy.configure(lm=lm)
    classify = dspy.Predict(DocQuality)

    dataset = dh._folder_of_folders_to_dataset(parse_objects=True)
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

        # unoptimized_prompt = dspy.inspect_history(n=-1)
        # optimized_prompt = dspy.inspect_history(n=1)

        # with open(f"modules/unoptimized_prompt.txt", "w") as f:
        #     f.write(unoptimized_prompt)
        # with open(f"modules/optimized_prompt.txt", "w") as f:
        #     f.write(optimized_prompt)
