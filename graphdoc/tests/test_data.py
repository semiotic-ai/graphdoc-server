# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser
from graphdoc import DataHelper

# external packages
import pandas as pd
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
        type_df = type_ds.to_pandas()

        assert isinstance(empty_ds, Dataset)
        assert isinstance(passing_ds, Dataset)
        assert isinstance(type_ds, Dataset)

        # add for type checking as to_pandas() can yield an iterable
        if isinstance(type_df, pd.DataFrame):
            assert isinstance(type_df.at[0, "category"], str)

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test__upload_to_hf(self, dh: DataHelper):
        graphdoc_ds = dh._create_graph_doc_dataset()
        assert dh._upload_to_hf(graphdoc_ds)

    # @pytest.mark.skipif("not config.getoption('--fire')")
    # def test__load_from_hf(self, dh: DataHelper, request):
    #     assert dh._load_from_hf()
