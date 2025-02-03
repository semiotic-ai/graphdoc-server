# system packages

# internal packages
import copy
import logging
from pathlib import Path
from typing import Optional, Union

from graphql import (
    EnumValueDefinitionNode,
    EnumTypeDefinitionNode,
    FieldDefinitionNode,
    Node,
    ObjectTypeDefinitionNode,
    StringValueNode,
    parse,
    print_ast,
)
from .loader.helper import check_directory_path, check_file_path

# external packages
import dspy
from graphql.language.ast import DocumentNode

# configure logging
# logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


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
    # GraphQL Methods #
    ###################
    # def parse_schema_from_str

    # def build_entity_select_all_query

    # def get_all_select_queries

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

    def fill_empty_descriptions(
        self,
        node: Node,
        new_column_value: str = "Description for column: {}",
        new_table_value: str = "Description for table: {}",
        use_value_name: bool = True,
        value_name: Optional[str] = None,
    ):
        """
        Recursively traverse the node and its children, filling in empty descriptions with the new column or table value. Do not update descriptions that already have a value.

        :param node: The GraphQL node to update
        :type node: Node
        :param new_column_value: The new column description value
        :type new_column_value: str
        :param new_table_value: The new table description value
        :type new_table_value: str
        :param use_value_name: Whether to use the value name in the description
        :type use_value_name: bool
        :param value_name: The name of the value
        :type value_name: Optional[str]
        :return: The updated node
        :rtype: Node
        """
        if hasattr(node, "description"):  # and node.description == None:
            description = getattr(node, "description", None)
            if description == None:
                # if the node is a table, use the table value
                if isinstance(node, ObjectTypeDefinitionNode):
                    new_value = new_table_value
                elif isinstance(node, EnumTypeDefinitionNode): # this is an enum type 
                    new_value = f"Description for enum type: {value_name}" # TODO: we should add this back to the fill_empty_descriptions parameter list
                # else the node is a column, use the column value
                else:
                    new_value = new_column_value
                # format with the value name if needed (table/column name)
                if use_value_name:
                    update_value = new_value.format(value_name)
                else:
                    update_value = new_value

                node.description = StringValueNode(value=update_value)

        for attr in dir(node):
            if attr.startswith("__") or attr == "description":
                continue
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, Node):
                        if (
                            isinstance(item, FieldDefinitionNode)
                            or isinstance(item, EnumValueDefinitionNode)
                            or isinstance(item, ObjectTypeDefinitionNode)
                            or isinstance(item, EnumTypeDefinitionNode) # EnumTypeDefinitionNode: check
                        ):
                            if isinstance(child, ObjectTypeDefinitionNode):
                                log.debug(
                                    f"found an instance of a ObjectTypeDefinitionNode: {item.name.value}"
                                )
                            value_name = item.name.value
                        self.fill_empty_descriptions(
                            item,
                            new_column_value,
                            new_table_value,
                            use_value_name,
                            value_name,
                        )
            elif isinstance(child, Node):
                if (
                    isinstance(child, FieldDefinitionNode)
                    or isinstance(child, EnumValueDefinitionNode)
                    or isinstance(child, ObjectTypeDefinitionNode)
                    or isinstance(item, EnumTypeDefinitionNode) # EnumTypeDefinitionNode: check
                ):
                    if isinstance(child, ObjectTypeDefinitionNode):
                        log.debug(
                            f"found an instance of a ObjectTypeDefinitionNode: {child.name.value}"
                        )
                    value_name = child.name.value
                self.fill_empty_descriptions(
                    child,
                    new_column_value,
                    new_table_value,
                    use_value_name,
                    value_name,
                )
        return node

    def schema_equality_check(self, gold_node: Node, check_node: Node) -> bool:
        """
        A method to check if two schema nodes are equal. Only checks that the schemas structures are equal, not the descriptions.

        :param gold_node: The gold standard schema node
        :type gold_node: Node
        :param check_node: The schema node to check
        :type check_node: Node
        :return: Whether the schemas are equal
        :rtype: bool
        """
        gold_node_copy = copy.deepcopy(gold_node)
        check_node_copy = copy.deepcopy(check_node)
        gold_node = self.update_node_descriptions(gold_node_copy)
        check_node = self.update_node_descriptions(check_node_copy)

        if print_ast(gold_node) != print_ast(check_node):
            return False
        else:
            return True

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

    ###################
    # DSPy Methods    #
    ###################
    # TODO: it would be better to move this elsewhere 
    def _signature_example_factory(self, signature_type: str) -> dspy.Example: 
        example_factory = {
            "doc_quality": dspy.Example(
                database_schema="filler schema",
                category="filler category",
                rating=1,
            ).with_inputs("database_schema"),
            "doc_generation": dspy.Example( 
                database_schema="filler schema",
                documented_schema="filler documented schema",
            ).with_inputs("database_schema"),
        }
        return example_factory[signature_type]

    def format_signature_prompt(self, signature: dspy.Signature, example: Optional[dspy.Example] = None, signature_type: Optional[str] = None) -> str:
        adapter = dspy.ChatAdapter()
        if not example:
            if signature_type:
                try:
                    example = self._signature_example_factory(signature_type)
                except KeyError:
                    raise ValueError(f"Invalid signature type: {signature_type}. Use one of (doc_quality, doc_generation)")
            else: 
                raise ValueError("No example provided and no signature type provided")
            
        try: 
            prompt = adapter.format(
                signature=signature,
                demos=[example],
                inputs=example,
            )
            prompt_str = f"------\nSystem\n------\n {prompt[0]["content"]} \n------\nUser\n------\n {prompt[1]['content']}"
            return prompt_str
        except Exception as e:
            raise ValueError(f"Failed to format signature prompt: {e}")