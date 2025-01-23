# system packages
import logging

# internal packages
from graphdoc import Parser, DocGeneratorEval

# external packages
import dspy
from dspy import Example, Prediction
from graphdoc.generate import DocGenerator
from graphdoc.main import GraphDoc
from graphql import print_ast
import pytest

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestDocGeneratorEval:

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_doc_generator(self, gd: GraphDoc, trainset: list[Example]):
        dg = dspy.ChainOfThought(DocGenerator)
        pred = dg(database_schema=trainset[0].database_schema)
        assert isinstance(pred, Prediction)
        assert isinstance(pred.documented_schema, str)
        log.info(f"The documented schema is: {pred.documented_schema}")

    # TODO:
    # def test_validate_schema_format(self, dge: DocGeneratorEval, par: Parser):
    #     gold_schema_file = "opensea_original_schema.graphql"
    #     check_schema_file = "opensea_original_schema_modified.graphql"

    #     gold_schema_str = print_ast(par.parse_schema_from_file(gold_schema_file))
    #     check_schema_str = print_ast(par.parse_schema_from_file(check_schema_file))

    #     gold_example = Example(database_schema=gold_schema_str, documented_schema=gold_schema_str).with_inputs("database_schema")
    #     check_example = Example(database_schema=check_schema_str, documented_schema=check_schema_str).with_inputs("database_schema")

    #     # TODO: figure out how to properly instantiate a prediction object
    #     gold_prediction = Prediction(gold_example)
    #     check_prediction = Prediction(check_example)

    def test_preprocess_schema(self, dge: DocGeneratorEval, trainset: list[Example]):
        for example in trainset:
            assert isinstance(dge.preprocess_schema(example.database_schema), str)

    # TODO:
    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_evaluate_documentation_quality(
        self, dge: DocGeneratorEval, trainset: list[Example]
    ):
        dg = dspy.ChainOfThought(DocGenerator)
        trainset = [
            Example(database_schema="type Marketplace { confusing: ID! }", documented_schema="type Marketplace { confusing: ID! }")
        ]
        pred = dg(database_schema=trainset[0].database_schema)
        eval = dge.evaluate_documentation_quality(trainset[0], pred)
        assert isinstance(eval, int)
        assert eval in [1, 2, 3, 4, 5]
        log.info(f"The quality of the documentation is: {eval}")
