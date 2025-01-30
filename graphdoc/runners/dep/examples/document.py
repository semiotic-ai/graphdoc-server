# system packages
import os
import logging

# internal packages

# external packages
import dspy
from dspy import Evaluate
from dotenv import load_dotenv
from graphdoc.data import DataHelper
from graphdoc.generate import DocGenerator, DocGeneratorEval

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")

# Run Time Variables
EVALUATE = True
OPTIMIZE = False
MODEL = "openai/gpt-4o"
CACHE = False

# logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    # setup dspy
    lm = dspy.LM(model=MODEL, api_key=OPENAI_API_KEY, cache=CACHE)
    dspy.configure(lm=lm)

    # setup graphdoc components
    dh = DataHelper(hf_api_key=HF_DATASET_KEY)
    dge = DocGeneratorEval(dh)
    dg = dspy.ChainOfThought(DocGenerator)

    categories = dh._categories()

    if EVALUATE:
        for category in categories:
            print(f"Evaluating {category}...")
            dataset = dh._folder_to_dataset(category=category, parse_objects=True)
            trainset = dh._create_graph_doc_example_trainset(dataset=dataset)

            evaluator = Evaluate(
                devset=trainset,
                num_threads=1,
                display_progress=True,
                display_table=5,
                return_all_scores=True,
                return_outputs=True,
            )
            overall_score, results, scores = evaluator(
                dg, dge.evaluate_documentation_quality
            )

            c = 0
            for result in results:
                print("-------------------")
                print(result[1])
                path_category = category.replace(" ", "_")
                os.makedirs(f"results/{path_category}", exist_ok=True)
                with open(f"results/{path_category}/result_{c}.graphql", "w") as f:
                    f.write(result[1].documented_schema)
                c += 1
            print(f"Scores: {scores}")
            print(f"Overall Score: {overall_score}")
            with open(f"results/{path_category}/scores.txt", "w") as f:
                f.write(f"Scores: {scores}")
                f.write(f"Overall Score: {overall_score}")

    if OPTIMIZE:
        log.info("Optimizing Document Generator...")
        dg.save(f"modules/baseline_document_generator.json", save_program=False)
        dataset = dh._folder_of_folders_to_dataset(parse_objects=True)
        trainset = dh._create_graph_doc_example_trainset(dataset=dataset)

        evaluator = Evaluate(
            devset=trainset,
            num_threads=1,
            display_progress=True,
            display_table=5,
            return_all_scores=True,
            return_outputs=True,
        )

        tp = dspy.MIPROv2(metric=dge.evaluate_documentation_quality, auto="light")
        optimized_generator = tp.compile(
            dg, trainset=trainset, max_labeled_demos=0, max_bootstrapped_demos=0
        )
        optimized_generator.save(
            f"modules/optimized_document_generator.json", save_program=False
        )
