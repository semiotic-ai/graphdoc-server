# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocQuality
from graphdoc import Parser
from graphdoc import DataHelper

# external packages
from graphdoc.data import SchemaObject
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

    # @pytest.mark.skipif("not config.getoption('--fire')")
    # def test__load_from_hf(self, dh: DataHelper, request):
    #     assert dh._load_from_hf()

    def test__load_folder_schemas(self, dh: DataHelper):
        schemas = dh._load_folder_schemas(category="perfect")
        assert isinstance(schemas, dict)
        for schema in schemas.values():
            assert isinstance(schema, SchemaObject)
        # TODO: add in test for an alternative schema directory location

    def test__load_folder_of_folders(self, dh: DataHelper):
        schemas = dh._load_folder_of_folders()
        assert isinstance(schemas, dict)
        counter = 0
        for schema in schemas.values():
            assert isinstance(schema, SchemaObject)
            counter += 1
        # TODO: make this a static value by deriving from a knowmn schema directory
        assert counter == 4

    def test__parse_objects_from_full_schema_object(self, dh: DataHelper):
        schemas = dh._load_folder_of_folders()
        if schemas:
            keys = list(schemas.keys())
            schema = schemas[keys[0]]
            objects = dh._parse_objects_from_full_schema_object(schema)
            assert isinstance(objects, dict)
            counter = 0
            for obj in objects.values():
                assert isinstance(obj, SchemaObject)
                counter += 1
            # TODO: make this a static value by deriving from a knowmn schema directory
            assert counter == 6
        else:
            log.warning("No schemas found in the schema directory")
            assert False

    def test__schema_objects_to_dict(self, dh: DataHelper):
        schemas = dh._load_folder_of_folders()
        if schemas:
            schema_dict = dh._schema_objects_to_dict(schemas)
            assert isinstance(schema_dict, dict)
            # TODO: make this a static value by deriving from a knowmn schema directory
            assert len(schema_dict["category"]) == 4

    def test__schema_objects_to_dataset(self, dh: DataHelper):
        schemas = dh._load_folder_of_folders()
        if schemas:
            dataset = dh._schema_objects_to_dataset(schemas)
            assert isinstance(dataset, Dataset)
            # TODO: make this a static value by deriving from a knowmn schema directory
            assert len(dataset) == 28

    def test__folder_to_dataset(self, dh: DataHelper):
        dataset = dh._folder_to_dataset(category="perfect")
        assert isinstance(dataset, Dataset)
        # TODO: make this a static value by deriving from a knowmn schema directory
        assert len(dataset) == 7

    def test__folder_of_folders_to_dataset(self, dh: DataHelper):
        dataset = dh._folder_of_folders_to_dataset()
        assert isinstance(dataset, Dataset)
        # TODO: make this a static value by deriving from a knowmn schema directory
        assert len(dataset) == 28

    @pytest.mark.skipif("not config.getoption('--write')")
    def test__upload_to_hf(self, dh: DataHelper):
        # TODO: later update this to not overwrite the dataset, but to append or not push if it makes no changes
        graphdoc_ds = dh._folder_of_folders_to_dataset()
        if graphdoc_ds:
            assert dh._upload_to_hf(graphdoc_ds)
        else:
            log.warning("Failed to create a dataset from the schema directory")
            assert False

    @pytest.mark.skipif("not config.getoption('--write')")
    def test__create_and_upload_repo_card(self, dh: DataHelper):
        assert dh._create_and_upload_repo_card()

    def test__check_graph_doc_dataset_format(self, dh: DataHelper):
        graphdoc_ds = dh._folder_of_folders_to_dataset()
        failing_ds = Dataset.from_dict(
            {
                "failing": [1],
                "rating": [1],
                "schema_name": [1],
                "schema_type": [1],
                "schema_str": [1],
            }
        )
        assert dh._check_graph_doc_dataset_format(graphdoc_ds)
        assert not dh._check_graph_doc_dataset_format(failing_ds)

    def test__add_to_graph_doc_dataset(self, dh: DataHelper):
        graphdoc_ds = dh._folder_of_folders_to_dataset()
        graphdoc_ds_2 = dh._folder_of_folders_to_dataset()
        len_before = len(graphdoc_ds)
        len_after = len(dh._add_to_graph_doc_dataset(graphdoc_ds, graphdoc_ds_2))
        assert len_after == len_before + len(graphdoc_ds_2)

    def test__drop_dataset_duplicates(self, dh: DataHelper):
        graphdoc_ds = dh._folder_of_folders_to_dataset()
        graphdoc_ds_2 = dh._folder_of_folders_to_dataset()
        len_original = len(graphdoc_ds)
        grouped_ds = dh._add_to_graph_doc_dataset(graphdoc_ds, graphdoc_ds_2)
        de_duplicated_ds = dh._drop_dataset_duplicates(grouped_ds)
        assert len(de_duplicated_ds) == len_original
        assert not len(de_duplicated_ds) == len(grouped_ds)

    @pytest.mark.skipif("not config.getoption('--fire')")
    def test_create_graph_doc_example_trainset(self, dh: DataHelper):
        examples = dh.create_graph_doc_example_trainset()
        assert isinstance(examples, list)
