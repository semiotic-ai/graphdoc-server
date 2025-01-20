# system packages

# internal packages
from pathlib import Path
from typing import Optional, Union

from graphql import Node, StringValueNode, parse
from .helper import check_directory_path, check_file_path

# external packages
from graphql.language.ast import DocumentNode


class Parser:
    """
    A class for the parsing and handling of GraphQL objects.

    :param schema_directory_path: A path to a directory containing schemas
    :type schema_directory_path: str
    """

    def __init__(self, schema_directory_path: Optional[str] = None) -> None:
        if schema_directory_path:
            check_directory_path(schema_directory_path)
        self.schema_directory_path = schema_directory_path

    ###################
    # Parsing Methods #
    ###################
    # def parse_schema_from_str

    def parse_schema_from_file(
        self, schema_file: Union[str, Path], schema_directory_path: Optional[str] = None
    ) -> DocumentNode:
        """
        Parse a schema from a file.

        :param schema_file: The name of the schema file
        :type schema_file: str
        :param schema_directory_path: A path to a directory containing schemas
        :type schema_directory_path: str
        :return: The parsed schema
        :rtype: DocumentNode
        """
        if schema_directory_path:
            check_directory_path(schema_directory_path)
            schema_path = Path(schema_directory_path) / schema_file
        elif self.schema_directory_path:
            schema_path = Path(self.schema_directory_path) / schema_file
        else:
            check_file_path(schema_file)
            schema_path = Path(schema_file)

        schema = schema_path.read_text()
        return parse(schema)

    def update_node_descriptions(
        self, node: Node, new_value: Optional[str] = None
    ) -> Node:
        """
        Given a GraphQL node, recursively traverse the node and its children, updating all descriptions with the new value.

        :param node: The GraphQL node to update
        :type node: Node
        :param new_value: The new description value
        :type new_value: Optional[str]
        :return: The updated node
        :rtype: Node
        """
        if hasattr(node, "description"):
            description = getattr(node, "description", None)
            if isinstance(description, StringValueNode):
                if new_value:
                    description.value = new_value
                else:
                    setattr(node, "description", None)

        for attr in dir(node):
            if attr.startswith("__") or attr == "description":
                continue
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, Node):
                        self.update_node_descriptions(item, new_value)
            elif isinstance(child, Node):
                self.update_node_descriptions(child, new_value)
        return node

    ###################
    # File Methods    #
    ###################
    # def format_text_schema_to_json

    # def format_schema_ast_to_json

    # def format_schema_file_to_json

    ###################
    # Token Methods   #
    ###################
    # def check_schema_token_count

    # def get_model_max_input_tokens

    # def check_prompt_validity
