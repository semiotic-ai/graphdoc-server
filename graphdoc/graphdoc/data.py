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
import pandas as pd
from .helper import check_directory_path, check_file_path
from .parser import Parser

# external packages
from typing import List, Literal, Optional, Union
from datasets import (
    Features,
    Value,
    Dataset,
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
)
from huggingface_hub.repocard import RepoCard
from huggingface_hub import HfApi
from datasets import concatenate_datasets
from datasets_sql import query
from dspy import Example

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
    :param repo_card_path: A path to a repo card file. This is mainly for accessing internal package data.
    :type repo_card_path: str
    """

    def __init__(
        self,
        hf_api_key: Optional[str] = None,
        schema_directory_path: Optional[str] = None,
        repo_card_path: Optional[str] = None,
    ) -> None:
        self.hf_api_key = hf_api_key
        self.hf_api = HfApi(token=self.hf_api_key)

        if schema_directory_path:
            check_directory_path(schema_directory_path)
            self.schema_directory_path = schema_directory_path
        else:
            self.schema_directory_path = Path(__file__).parent / "assets" / "schemas"
            check_directory_path(self.schema_directory_path)

        if repo_card_path:
            check_file_path(repo_card_path)
            self.repo_card_path = repo_card_path
        else:
            self.repo_card_path = (
                Path(__file__).parent / "assets" / "cards" / "GRAPHDOC_SCHEMAS.MD"
            )
            check_file_path(self.repo_card_path)

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
        """
        Check the type of a schema node.

        :param node: The schema node to check
        :type node: Node
        :return: The type of the schema node
        :rtype: str
        """
        if isinstance(node, DocumentNode):
            return "full schema"
        elif isinstance(node, ObjectTypeDefinitionNode):
            return "table schema"
        elif isinstance(node, EnumValueDefinitionNode):
            return "enum schema"
        else:
            return "unknown schema"

    def _parse_objects_from_full_schema_object(
        self, schema: SchemaObject
    ) -> Union[dict[str, SchemaObject], None]:
        """
        Parse out all available tables from a full schema object.

        :param schema: The full schema object to parse
        :type schema: SchemaObject
        :return: The parsed objects (tables and enums)
        :rtype: Union[dict, None]
        """
        if schema.schema_ast is None:
            log.info(f"Schema object has no schema_ast: {schema.schema_name}")
            return None
        elif not isinstance(schema.schema_ast, DocumentNode):
            log.info(
                f"Schema object cannot be further decomposed: {schema.schema_name}"
            )
            return None

        tables = {}
        for definition in schema.schema_ast.definitions:
            if isinstance(definition, ObjectTypeDefinitionNode):
                key = f"{schema.key}_{definition.name.value}"
                schema_type = self._check_node_type(definition)
            elif isinstance(definition, EnumValueDefinitionNode):
                key = f"{schema.key}_{definition.name.value}"
                schema_type = self._check_node_type(definition)
            else:
                continue
            object_schema = SchemaObject.from_dict(
                {
                    "key": key,
                    "category": schema.category,
                    "rating": schema.rating,
                    "schema_name": definition.name.value,
                    "schema_type": schema_type,
                    "schema_str": print_ast(definition),
                    "schema_ast": definition,
                }
            )
            tables[object_schema.key] = object_schema
        return tables

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

    def _schema_objects_to_dict(self, schemas: dict[str, SchemaObject]) -> dict:
        """
        Convert parsed schemas to a dictionary object.

        :param schemas: The parsed schemas
        :type schemas: dict
        :return: The converted schemas
        :rtype: dict
        """
        schema_dict = {
            "category": [],
            "rating": [],
            "schema_name": [],
            "schema_type": [],
            "schema_str": [],
        }
        for schema in schemas.values():
            schema_dict["category"].append(schema.category)
            schema_dict["rating"].append(schema.rating)
            schema_dict["schema_name"].append(schema.schema_name)
            schema_dict["schema_type"].append(schema.schema_type)
            schema_dict["schema_str"].append(schema.schema_str)
        return schema_dict

    def _schema_objects_to_dataset(
        self, schemas: dict[str, SchemaObject], parse_objects: bool = True
    ) -> Dataset:
        """
        Convert parsed schemas to a dataset.

        :param schemas: The parsed schemas
        :type schemas: dict
        :param parse_objects: Whether to parse objects from the schemas
        :type parse_objects: bool
        :return: The converted dataset
        :rtype: Dataset
        """
        if parse_objects:
            original_schemas = schemas.copy()
            for schema in original_schemas.values():
                objects = self._parse_objects_from_full_schema_object(schema)
                if objects:
                    schemas.update(objects)
        schema_dict = self._schema_objects_to_dict(schemas)
        return self._create_graph_doc_dataset(schema_dict)

    def _folder_to_dataset(
        self,
        category: str,
        folder_path: Optional[Union[str, Path]] = None,
        parse_objects: bool = True,
    ) -> Dataset:
        """
        Convert a folder of schemas to a dataset.

        :param category: The category of the schemas
        :type category: str
        :param folder_path: The path to the folder containing the schemas
        :type folder_path: Union[str, Path]
        :param parse_objects: Whether to parse objects from the schemas
        :type parse_objects: bool
        :return: The converted dataset
        :rtype: Dataset
        """
        schemas = self._load_folder_schemas(category, folder_path)
        return self._schema_objects_to_dataset(schemas, parse_objects)

    def _folder_of_folders_to_dataset(
        self, folder_path: Optional[dict] = None, parse_objects: bool = True
    ) -> Dataset:
        """
        Convert a folder of folders containing schemas to a dataset.

        :param folder_path: The dictionary that maps the category to the folder. {category: folder_path}
        :type folder_path: dict
        :param parse_objects: Whether to parse objects from the schemas
        :type parse_objects: bool
        :return: The converted dataset
        :rtype: Dataset
        """
        schemas = self._load_folder_of_folders(folder_path)
        if schemas is None:
            raise ValueError("No schemas found")
        return self._schema_objects_to_dataset(schemas, parse_objects)

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

    def _create_graph_doc_dataset(self, data: Optional[dict] = None) -> Dataset:
        """
        Create a graph_doc dataset from a data dictionary.

        :param data: The data dictionary to create the dataset from
        :type data: dict
        :return: The created dataset
        :rtype: Dataset
        """
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

    def _check_graph_doc_dataset_format(self, dataset: Dataset) -> bool:
        """
        Check that the dataset has the correct format.

        :param dataset: The dataset to check
        :type dataset: Dataset
        :return: True if the dataset has the correct format
        :rtype: bool
        """
        features = self._get_graph_doc_columns()
        return dataset.features == features

    def _add_to_graph_doc_dataset(
        self, dataset: Dataset, data: Union[Dataset, dict]
    ) -> Dataset:
        """
        Add new data to a dataset.

        :param dataset: The dataset to add the data to
        :type dataset: Dataset
        :param data: The data to add to the dataset (either a Dataset or dictionary)
        :type data: Union[Dataset, dict]
        :return: The updated dataset
        :rtype: Dataset
        """
        if self._check_graph_doc_dataset_format(dataset):
            if isinstance(data, Dataset):
                if self._check_graph_doc_dataset_format(data):
                    return concatenate_datasets([dataset, data])
                else:
                    raise ValueError("Input Dataset does not have the correct format")
            elif isinstance(data, dict):
                if self._check_graph_doc_data_dict(data):
                    data_ds = self._create_graph_doc_dataset(data)
                    return concatenate_datasets([dataset, data_ds])
                else:
                    raise ValueError("Data dictionary does not have the correct format")
        else:
            raise ValueError("Base dataset does not have the correct format")

    def _drop_dataset_duplicates(self, dataset: Dataset) -> Dataset:
        """
        Drop duplicates from a dataset.

        :param dataset: The dataset to check
        :type dataset: Dataset
        :return: True if there are no duplicates
        :rtype: bool
        """
        de_duplicated_dataset = query("SELECT DISTINCT * FROM dataset")
        return de_duplicated_dataset

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

    def _upload_repo_card_to_hf(
        self,
        repo_id: str,
        repo_card: RepoCard,
        repo_type: Literal["dataset", "model"] = "dataset",
        token: Optional[str] = None,
    ) -> bool:
        """
        A method to upload a repo card to the Hugging Face Hub.

        :param repo_id: The repository ID to upload the repo card to
        :type repo_id: str
        :param repo_card: The repo card to upload
        :type repo_card: RepoCard
        :param token: The Hugging Face API token
        :type token: str
        :return: Whether the upload was successful
        :rtype: bool
        """
        try:
            repo_url = self.hf_api.create_repo(
                repo_id=repo_id, repo_type=repo_type, exist_ok=True
            )
            log.debug(f"Repo URL: {repo_url}")
            log.debug(f"Repo Card Type: {type(repo_card)}")
            if token:
                repo_card.push_to_hub(repo_id=repo_id, token=token, repo_type=repo_type)
            else:
                repo_card.push_to_hub(
                    repo_id=repo_id, token=self.hf_api_key, repo_type=repo_type
                )
            return True
        except Exception as e:
            print(f"Error uploading repo card: {e}")
            return False

    def _create_repo_card(self, repo_card_path: str) -> RepoCard:
        """
        A method to create a repo card.

        :param repo_card_path: The path to the repo card
        :type repo_card_path: str
        :return: The created repo card
        :rtype: RepoCard
        """
        check_file_path(repo_card_path)
        return RepoCard.load(repo_card_path)

    def _create_and_upload_repo_card(
        self,
        repo_id: str = "semiotic/graphdoc_schemas",
        repo_card_path: Optional[str] = None,
    ) -> bool:
        """
        A method to create and upload a repo card to the Hugging Face Hub.

        :param repo_id: The repository ID to upload the repo card to
        :type repo_id: str
        :param repo_card_path: The path to the repo card
        :type repo_card_path: str
        :return: Whether the upload was successful
        :rtype: bool
        """
        if repo_card_path is None:
            repo_card_path = str(self.repo_card_path)
        else:
            check_file_path(repo_card_path)

        repo_card = self._create_repo_card(repo_card_path)
        try:
            return self._upload_repo_card_to_hf(repo_id=repo_id, repo_card=repo_card)
        except Exception as e:
            print(f"Error uploading repo card: {e}")
            return False

    ######################
    # DSPy Data Helper
    ######################
    def _create_graph_doc_example_trainset(self, dataset: Dataset) -> List[Example]:
        """
        Create a trainset for the graph_doc dataset.

        :param dataset: The dataset to create the trainset from
        :type dataset: Dataset
        :return: The created trainset
        :rtype: List[Example]
        """
        # TODO: refactor this to use the dataset directly
        records = dataset.to_pandas()
        if isinstance(records, pd.DataFrame):
            records = records.to_dict("records")
        else:
            raise ValueError(
                f"Dataset is not a valid type, must be a DataFrame. Is: {type(records)}"
            )

        return [
            Example(
                database_schema=record["schema_str"],
                category=record["category"],
                rating=int(
                    record["rating"]
                ),  # TODO: we should handle this at the evaluation signature and agree upon a type
            ).with_inputs("database_schema")
            for record in records
        ]

    def _create_doc_generator_example_trainset(self, dataset: Dataset) -> List[Example]:
        """
        Create a trainset for the DocGenerator module.
        """
        # TODO: refactor this to use the dataset directly
        records = dataset.to_pandas()
        if isinstance(records, pd.DataFrame):
            records = records.to_dict("records")
        else:
            raise ValueError(
                f"Dataset is not a valid type, must be a DataFrame. Is: {type(records)}"
            )

        return [
            Example(
                database_schema=record["schema_str"],
                documented_schema=record[
                    "schema_str"
                ],  # TODO: we must refactor this to use the gold
            ).with_inputs("database_schema")
            for record in records
        ]

    def create_graph_doc_example_trainset(
        self, repo_id: str = "semiotic/graphdoc_schemas", token: Optional[str] = None
    ) -> List[Example]:
        """
        Create a trainset for the graph_doc dataset.

        :param repo_id: The repository ID to load the dataset from
        :type repo_id: str
        :param token: The Hugging Face API token
        :type token: str
        :return: The created trainset
        :rtype: List[Example]
        """
        dataset = self._load_from_hf(repo_id=repo_id, token=token)
        if dataset:
            if isinstance(dataset, DatasetDict):
                dataset = dataset.get("train")
            if isinstance(dataset, Dataset):
                return self._create_graph_doc_example_trainset(dataset)
            else:
                raise ValueError(
                    f"Dataset is not a valid type, must be a Dataset. Is: {type(dataset)}"
                )
        else:
            raise ValueError("No dataset found")


# TODO: we could make this a subclass in the future if we start to add more datasets
# class DocQualityDataHelper(DataHelper):
#     """
#     A helper class specifically for the DocQuality dataset.
#     """

#     def __init__(self, hf_api_key: Optional[str] = None, schema_directory_path: Optional[str] = None, repo_card_path: Optional[str] = None) -> None:
#         super().__init__(hf_api_key, schema_directory_path, repo_card_path)
