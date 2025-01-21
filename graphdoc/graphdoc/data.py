# system packages

# internal packages
from dataclasses import dataclass
import logging
from pathlib import Path

from graphql import (
    DocumentNode,
    EnumValueDefinitionNode,
    Node,
    ObjectTypeDefinitionNode,
    print_ast,
)
from .helper import check_directory_path, check_file_path
from .parser import Parser

# external packages
from typing import Literal, Optional, Union
from datasets import (
    Features,
    Value,
    Dataset,
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
)

# configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


@dataclass
class SchemaObject:
    key: str
    category: Optional[
        Literal["perfect", "almost perfect", "somewhat correct", "incorrect"]
    ] = None
    rating: Optional[Literal["4", "3", "2", "1"]] = None
    schema_name: Optional[str] = None
    schema_type: Optional[Literal["full schema", "table schema", "enum schema"]] = (
        None  # , "column schema"
    )
    schema_str: Optional[str] = None
    schema_ast: Optional[Node] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaObject":
        """Create SchemaObject from dictionary with validation."""

        # Check required key
        if "key" not in data:
            raise ValueError("Missing required field: key")

        # Validate category if present
        if "category" in data and data["category"] is not None:
            valid_categories = [
                "perfect",
                "almost perfect",
                "somewhat correct",
                "incorrect",
            ]
            if data["category"] not in valid_categories:
                raise ValueError(
                    f"Invalid category. Must be one of: {valid_categories}"
                )

        # Validate rating if present
        if "rating" in data and data["rating"] is not None:
            valid_ratings = ["4", "3", "2", "1"]
            if data["rating"] not in valid_ratings:
                raise ValueError(f"Invalid rating. Must be one of: {valid_ratings}")

        # Validate schema_type if present
        if "schema_type" in data and data["schema_type"] is not None:
            valid_types = ["full schema", "table schema", "enum schema"]
            if data["schema_type"] not in valid_types:
                raise ValueError(f"Invalid schema_type. Must be one of: {valid_types}")

        # Create instance with validated data
        return cls(
            key=data["key"],
            category=data.get("category"),
            rating=data.get("rating"),
            schema_name=data.get("schema_name"),
            schema_type=data.get("schema_type"),
            schema_str=data.get("schema_str"),
            schema_ast=data.get("schema_ast"),
        )


class DataHelper:
    """
    A helper class for interacting with external and internal data sets and sources.

    :param hf_api_key: The Hugging Face API key
    :type hf_api_key: str
    :param schema_directory_path: A path to a directory containing sub-directories of schemas. This is mainly for accessing internal package data.
    :type schema_directory_path: str
    """

    def __init__(
        self,
        hf_api_key: Optional[str] = None,
        schema_directory_path: Optional[str] = None,
    ) -> None:
        self.hf_api_key = hf_api_key
        if schema_directory_path:
            check_directory_path(schema_directory_path)
            self.schema_directory_path = schema_directory_path
        else:
            self.schema_directory_path = Path(__file__).parent / "assets" / "schemas"
            check_directory_path(self.schema_directory_path)

        # instantiate the parser for handling graphql
        self.par = Parser()

    ######################
    # loading local data
    ######################

    def _categories(self) -> list:
        """
        Return the valid categories for the schema directory.

        :return: The categories for the schema directory
        :rtype: list
        """
        return ["perfect", "almost perfect", "somewhat correct", "incorrect"]

    def _folder_paths(self) -> dict:
        """
        Return the folder paths for the schema directory.

        :return: The folder paths for the schema directory
        :rtype: dict
        """
        return {
            "perfect": Path(self.schema_directory_path) / "perfect",
            "almost perfect": Path(self.schema_directory_path) / "almost_perfect",
            "somewhat correct": Path(self.schema_directory_path) / "somewhat_correct",
            "incorrect": Path(self.schema_directory_path) / "incorrect",
        }

    def _category_ratings(self) -> dict:
        """
        Return the category ratings for the schema directory.

        :return: The category ratings for the schema directory
        :rtype: dict
        """
        return {
            "perfect": "4",
            "almost perfect": "3",
            "somewhat correct": "2",
            "incorrect": "1",
        }

    def _check_category_validity(self, category: str) -> bool:
        """
        Check the validity of a category.

        :param category: The category to check
        :type category: str
        :return: True if the category is valid
        :rtype: bool
        """
        if category not in self._categories():
            raise ValueError(f"Invalid category: {category}")
        return True

    # TODO: uppdate field types for enums and entities
    def _check_node_type(self, node: Node) -> str:
        # Union[DocumentNode, ObjectTypeDefinitionNode, EnumValueDefinitionNode]
        if isinstance(node, DocumentNode):
            return "full schema"
        elif isinstance(node, ObjectTypeDefinitionNode):
            return "table schema"
        elif isinstance(node, EnumValueDefinitionNode):
            return "enum schema"
        else:
            return "unknown schema"

    # load schemas from a folder, keep the difficulty tag
    def _load_folder_schemas(
        self, category: str, folder_path: Optional[Union[str, Path]] = None
    ) -> dict[str, SchemaObject]:
        """
        Load schemas from a folder, keeping the difficulty tag.

        :param folder_path: The path to the folder containing the schemas
        :type folder_path: Union[str, Path]
        :param category: The category of the schemas
        :type category: str
        :return: The loaded schemas
        :rtype: dict
        """
        self._check_category_validity(category)

        if folder_path is None:
            folder_path = self._folder_paths().get(category)
            if folder_path is None:
                raise ValueError(
                    f"Invalid category: {category} or folder path: {folder_path}"
                )
        else:
            check_directory_path(folder_path)

        schemas = {}
        for schema_file in Path(folder_path).iterdir():
            check_file_path(schema_file)

            try:
                schema_ast = self.par.parse_schema_from_file(schema_file)
            except Exception as e:
                log.warning(f"Error parsing schema {schema_file}: {e}")
                schema_ast = None

            schema = SchemaObject.from_dict(
                {
                    "key": str(schema_file),
                    "category": category,
                    "rating": self._category_ratings().get(category),
                    "schema_name": schema_file.stem,
                    "schema_type": (
                        self._check_node_type(schema_ast) if schema_ast else None
                    ),
                    "schema_str": print_ast(schema_ast) if schema_ast else None,
                    "schema_ast": schema_ast,
                }
            )
            schemas[schema_file] = schema
        return schemas

    # load folder of schemas from a folder, keep the difficulty tag
    def _load_folder_of_folders(
        self, folder_path: Optional[dict] = None
    ) -> Union[dict[str, SchemaObject], None]:
        """
        Load a folder of folders containing schemas, keeping the difficulty tag.

        :param folder_path: The dictionary that maps the category to the folder. {category: folder_path}
        :type folder_path: dict
        :return: The loaded schemas
        :rtype: dict
        """
        if folder_path:
            for key in folder_path.keys():
                self._check_category_validity(key)
        else:
            folder_path = self._folder_paths()

        schemas = {}
        for category, path in folder_path.items():
            schemas.update(self._load_folder_schemas(category, path))
        return schemas

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
    def _load_from_hf(
        self, repo_id: str = "semiotic/graphdoc_schemas", token: Optional[str] = None
    ) -> Union[(DatasetDict | Dataset | IterableDatasetDict | IterableDataset), None]:
        """
        A method to load a dataset from the Hugging Face Hub.

        :param repo_id: The repository ID to load the dataset from
        :type repo_id: str
        :param token: The Hugging Face API token
        :type token: str
        :return: The loaded dataset
        :rtype: Union[Dataset, None]
        """
        try:
            if token:
                return load_dataset(path=repo_id, token=token)
            else:
                return load_dataset(path=repo_id, token=self.hf_api_key)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    # check that the dataset has the correct format

    # add new data to a dataset

    # check that there are no duplicates in a dataset

    # deduplicate a dataset

    # upload a dataset to huggingface
    def _upload_to_hf(
        self,
        dataset: Dataset,
        repo_id: str = "semiotic/graphdoc_schemas",
        token: Optional[str] = None,
    ) -> bool:
        """
        A method to upload a dataset to the Hugging Face Hub.

        :param dataset: The dataset to upload
        :type dataset: Dataset
        :param repo_id: The repository ID to upload the dataset to
        :type repo_id: str
        :param token: The Hugging Face API token
        :type token: str
        :return: Whether the upload was successful
        :rtype: bool
        """
        try:
            if token:
                dataset.push_to_hub(repo_id=repo_id, token=token)
            else:
                dataset.push_to_hub(repo_id=repo_id, token=self.hf_api_key)
            return True
        except Exception as e:
            print(f"Error uploading dataset: {e}")
            return False
