# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality, DocQualityEval

# external packages
from graphdoc.data import DataHelper
import pytest
from dspy import Example
from dspy import LM, Predict, Evaluate

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestDocQualityEval:

    def test_validate_category(self):
        ex = Example()
        ex.category = "test"

        dqe = DocQualityEval()
        assert dqe.validate_category(ex, ex)

    def test_validate_rating(self):
        ex = Example()
        ex.rating = "test"

        dqe = DocQualityEval()
        assert dqe.validate_rating(ex, ex)

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_doc_quality(self, gd: GraphDoc):
        dq = DocQuality
        classify = Predict(dq)
        classification = classify(
            database_schema="this is a test, you should reply with a rating of 4 and a category of perfect"
        )
        assert classification.category == "perfect"
        assert classification.rating == 4

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_create_evaluator(self, dh: DataHelper):
        dqe = DocQualityEval(dh)
        training_ds = dh.create_graph_doc_example_trainset()
        assert isinstance(dqe.create_evaluator(), Evaluate)
        assert isinstance(dqe.create_evaluator(trainset=training_ds), Evaluate)

    @pytest.mark.asyncio
    @pytest.mark.skipif("not config.getoption('--run-evaluator')")
    def test_run_evaluator(self, gd: GraphDoc, dh: DataHelper):
        dqe = DocQualityEval(gd)
        trainset = dh.create_graph_doc_example_trainset()
        log.debug(f"Training Example: {trainset[0]}")
        # trainset = [
        #     Example(
        #         database_schema="this is a test, you should reply with a rating of 4 and a category of perfect",
        #         category="perfect",
        #         rating=4,
        #     ).with_inputs("database_schema")
        # ]
        # log.debug(f"Training Example: {trainset[0]}") 
        evaluator = dqe.create_evaluator(trainset=trainset)
        dqe.run_evaluator(evaluator, DocQuality, dqe.validate_rating)
        # assert False
