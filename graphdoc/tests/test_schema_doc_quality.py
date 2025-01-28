# system packages
import logging

# internal packages
from graphdoc.prompts import DocQualityPrompt, DocQualitySignature, SinglePrompt
from graphdoc import DataHelper, GraphDoc

# external packages
import dspy

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestSchemaDocQuality:
    def test_schema_doc_quality(self):
        pass

    def test_doc_quality_prompt(self):
        dqp = DocQualityPrompt(type="predict", metric_type="rating")
        assert isinstance(dqp, SinglePrompt)
        assert isinstance(dqp.infer, dspy.Predict)

    # TODO: we should move this to a test for the SinglePrompt class
    def test_get_predict(self):
        dqp = DocQualityPrompt(type="predict", metric_type="rating")
        p = dqp.get_predict()
        assert isinstance(p, dspy.Predict)

    # TODO: we should move this to a test for the SinglePrompt class
    def test_get_chain_of_thought(self):
        dqp = DocQualityPrompt(type="chain_of_thought", metric_type="rating")
        p = dqp.get_chain_of_thought()
        assert isinstance(p, dspy.ChainOfThought)

    def test_evaluate_metric(self, gd: GraphDoc):
        example = dspy.Example(
            database_schema="this is a test, you should reply with a rating of 4 and a category of perfect",
            category="perfect",
            rating=4,
        )
        prediction = dspy.Prediction(
            category="perfect",
            rating=4,
        )
        failing_prediction = dspy.Prediction(
            category="fail",
            rating=3,
        )
        dqp = DocQualityPrompt(type="predict", metric_type="rating")
        assert dqp.evaluate_metric(example, prediction)
        assert not dqp.evaluate_metric(example, failing_prediction)

        dqp = DocQualityPrompt(type="predict", metric_type="category")
        assert dqp.evaluate_metric(example, prediction)
        assert not dqp.evaluate_metric(example, failing_prediction)

    def test_evaluate_evalset(self, gd: GraphDoc):
        dqp = DocQualityPrompt(type="predict", metric_type="rating")

        example_pass_four = dspy.Example(
            database_schema="this is a test, you should reply with a rating of 4 and a category of perfect",
            category="perfect",
            rating=4,
        ).with_inputs("database_schema")
        example_pass_three = dspy.Example(
            database_schema="this is a test, you should reply with a rating of 3 and a category of good",
            category="good",
            rating=3,
        ).with_inputs("database_schema")
        example_pass_two = dspy.Example(
            database_schema="this is a test, you should reply with a rating of 2 and a category of bad",
            category="bad",
            rating=2,
        ).with_inputs("database_schema")
        example_pass_one = dspy.Example(
            database_schema="this is a test, you should reply with a rating of 1 and a category of terrible",
            category="terrible",
            rating=1,
        ).with_inputs("database_schema")
        example_fail = dspy.Example(
            database_schema="this is a test, you should reply with a rating of 4 and a category of perfect",
            category="good",
            rating=3,
        ).with_inputs("database_schema")
        evalset = [example_pass_four, example_pass_three, example_pass_two, example_pass_one, example_fail]
        formatted_results = dqp.evaluate_evalset(evalset)
        assert isinstance(formatted_results, dict)
        assert isinstance(formatted_results['overall_score'], float)
        assert isinstance(formatted_results['per_category_scores'], dict)
        assert isinstance(formatted_results['details'], list)
        assert formatted_results['per_category_scores']['good']['percent_correct'] == 50.0

