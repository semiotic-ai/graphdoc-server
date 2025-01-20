# system packages

# internal packages

# external packages
from typing import Optional
from datasets import Features, Value, Dataset


class DataHelper:
    """
    A helper class for interacting with external and internal data sets and sources.
    """

    def __init__(
        self,
    ) -> None:
        pass

    ######################
    # loading local data
    ######################

    # load schemas from a folder, keep the difficulty tag

    # load folder of schemas from a folder, keep the difficulty tag

    # parse out tables from a schema, keep the difficulty tag

    # convert parsed schemas to a dataset

    # convert a folder of schemas to a dataset

    # convert a folder of folders to a dataset

    ######################
    # hf functions
    ######################

    # return the graph_doc dataset columns
    def _get_graph_doc_columns(self) -> Features:
        """
        Return the columns for the graph_doc dataset.

        :return: The columns for the graph_doc dataset
        :rtype: Features
        """
        return Features(
            {
                "category": Value("string"),
                "rating": Value("string"),
                "schema_name": Value("string"),
                "schema_type": Value("string"),
                "schema_str": Value("string"),
            }
        )

    def _get_empty_graphdoc_data(self) -> dict:
        """
        Return an empty dictionary for the graph_doc dataset.

        :return: An empty dictionary for the graph_doc dataset
        :rtype: dict
        """
        return {
            "category": [],
            "rating": [],
            "schema_name": [],
            "schema_type": [],
            "schema_str": [],
        }

    def _check_graph_doc_data_dict(self, data: dict) -> bool:
        """
        Check that the data dictionary has the correct format.

        :param data: The data dictionary to check
        :type data: dict
        :return: True if the data dictionary has the correct format
        :rtype: bool
        """
        features = self._get_graph_doc_columns()
        required_keys = set(features.keys())
        data_keys = set(data.keys())

        missing_keys = required_keys - data_keys
        extra_keys = data_keys - required_keys

        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        if extra_keys:
            raise ValueError(f"Found unexpected extra keys: {extra_keys}")

        return True

    # create a dataset
    def _create_graph_doc_dataset(self, data: Optional[dict] = None) -> Dataset:
        features = self._get_graph_doc_columns()
        if data is None:
            data = self._get_empty_graphdoc_data()
        else:
            self._check_graph_doc_data_dict(data)
        return Dataset.from_dict(data, features=features)

    # pull down a dataset from huggingface

    # check that the dataset has the correct format

    # add new data to a dataset

    # check that there are no duplicates in a dataset

    # deduplicate a dataset

    # upload a dataset to huggingface
