# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser
from graphdoc import DataHelper

# external packages
import pytest
from dspy import LM, Predict
from datasets import Features, Dataset

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class TestDataHelper:

    def test__get_graph_doc_columns(self, dh: DataHelper):
        features = dh._get_graph_doc_columns()
        assert isinstance(features, Features)

    def test___check_graph_doc_data_dict(self, dh: DataHelper):
        passing_dict = {
            "category": ["test"],
            "rating": ["test"],
            "schema_name": ["test"],
            "schema_type": ["test"],
            "schema_str": ["test"],
        }
        missing_key_dict = {
            "category": ["test"],
            "rating": ["test"],
            "schema_name": ["test"],
            "schema_type": ["test"],
        }
        extra_key_dict = {
            "category": ["test"],
            "rating": ["test"],
            "schema_name": ["test"],
            "schema_type": ["test"],
            "schema_str": ["test"],
            "extra_key": ["test"],
        }

        assert dh._check_graph_doc_data_dict(passing_dict)

        with pytest.raises(ValueError):
            dh._check_graph_doc_data_dict(missing_key_dict)
        with pytest.raises(ValueError):
            dh._check_graph_doc_data_dict(extra_key_dict)

    def test__create_graph_doc_dataset(self, dh: DataHelper):
        passing_dict = {
            "category": ["test"],
            "rating": ["test"],
            "schema_name": ["test"],
            "schema_type": ["test"],
            "schema_str": ["test"],
        }
        wrong_type_dict = {
            "category": [1],
            "rating": ["test"],
            "schema_name": ["test"],
            "schema_type": ["test"],
            "schema_str": ["test"],
        }
        empty_ds = dh._create_graph_doc_dataset()
        passing_ds = dh._create_graph_doc_dataset(passing_dict)
        type_ds = dh._create_graph_doc_dataset(wrong_type_dict)

        assert isinstance(empty_ds, Dataset)
        assert isinstance(passing_ds, Dataset)
        assert isinstance(type_ds, Dataset)
        assert isinstance(type_ds.to_pandas().iloc[0]["category"], str)
