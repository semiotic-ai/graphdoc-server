# system packages

# internal packages

# external packages
from datasets import Features, Value


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
        return Features(
            {
                "category": Value("string"),
                "rating": Value("string"),
                "schema_name": Value("string"),
                "schema_type": Value("string"),
                "schema_str": Value("string"),
            }
        )

    # create a dataset

    # pull down a dataset from huggingface

    # check that the dataset has the correct format

    # add new data to a dataset

    # check that there are no duplicates in a dataset

    # deduplicate a dataset

    # upload a dataset to huggingface
