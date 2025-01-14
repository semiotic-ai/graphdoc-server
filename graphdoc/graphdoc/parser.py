# system packages
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional
import importlib.resources as pkg_resources

# internal packages

# external packages
from tokencost import count_string_tokens
from graphql import EnumValueDefinitionNode, FieldNode, build_schema, parse, build_ast_schema, validate_schema, print_ast
from graphql import Node, StringValueNode
from graphql import parse
from graphql.language.ast import (
    DocumentNode,
    ObjectTypeDefinitionNode,
    FieldDefinitionNode,
)

from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

# configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class Parser:
    def __init__(self, schema_directory_path: Optional[str] = None):
        if schema_directory_path:
            schema_directory_path = Path(schema_directory_path).resolve()
            if not schema_directory_path.is_dir():
                raise ValueError(
                    f"The provided schema directory path '{schema_directory_path}' is not a valid directory."
                )
        self.schema_directory_path = schema_directory_path

        # load the model token details from tokencost
        package_name = "tokencost"
        resource_name = "model_prices.json"

        try:
            with pkg_resources.files(package_name).joinpath(resource_name).open(
                "r"
            ) as file:
                self.model_data = json.load(file)
        except FileNotFoundError:
            raise ValueError(
                f"{resource_name} not found in the package {package_name}."
            )

    def parse_schema_from_text(self, schema_text: str):
        schema_ast = parse(schema_text)
        return schema_ast

    def parse_schema_from_file(
        self, schema_file: str, schema_directory_path: Optional[str] = None
    ):
        if schema_directory_path:
            schema_directory_path = Path(schema_directory_path).resolve()
            if not schema_directory_path.is_dir():
                raise ValueError(
                    f"The provided schema directory path '{schema_directory_path}' is not a valid directory."
                )
            else:
                self.schema_directory_path = schema_directory_path

        schema_path = Path(self.schema_directory_path) / schema_file
        schema = schema_path.read_text()
        return self.parse_schema_from_text(schema)

    def check_schema_token_count(self, schema_ast, model: str = "gpt-4o"):
        schema_str = print_ast(schema_ast)
        token_count = count_string_tokens(schema_str, model)
        return token_count

    def get_model_max_input_tokens(self, model: str = "gpt-4o"):
        if model in self.model_data:
            return self.model_data[model]["max_input_tokens"]
        else:
            raise ValueError(f"Model {model} not found in the model data.")

    def check_prompt_validity(self, prompt: str, model: str = "gpt-4o"):
        """Check to see if the prompt for a model is too long."""
        prompt_token_count = count_string_tokens(prompt, model)
        max_input_tokens = self.get_model_max_input_tokens(model)
        if prompt_token_count > max_input_tokens:
            logging.warning(
                f"Prompt token count {prompt_token_count} exceeds the maximum input tokens {max_input_tokens} for model {model}."
            )
            return False
        return True

    def format_text_schema_to_json(self, schema_text, label="gold", version="1.0.0"):
        lines = schema_text.strip().split("\n")
        formatted_lines = []

        for line in lines:
            cleaned_line = line.replace('"', '"').rstrip()

            if not cleaned_line:
                formatted_lines.append("\n")
                continue

        formatted_lines.append(cleaned_line + " \n")
        json_output = {f"{label}": {"version": f"{version}", "prompt": formatted_lines}}
        return json_output

    def format_schema_ast_to_json(self, schema_ast, label="gold", version="1.0.0"):
        schema_str = print_ast(schema_ast)
        return self.format_text_schema_to_json(schema_str, label, version)

    def format_schema_file_to_json(
        self, schema_file, schema_directory_path=None, label="gold", version="1.0.0"
    ):
        schema_ast = self.parse_schema_from_file(schema_file, schema_directory_path)
        return self.format_schema_ast_to_json(schema_ast, label, version)

    def update_node_descriptions(self, node, new_value=None):
        """Update the descriptions of the nodes in the schema. If new_value is None, the description nodes will be removed."""
        if hasattr(node, "description") and isinstance(
            node.description, StringValueNode
        ):
            if new_value:
                node.description.value = new_value
            else:
                node.description = None

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
    
    def fill_empty_descriptions(self, node, new_value="Description for column: {}", use_column_name=True, column_name=None):
        """Fill empty descriptions in the schema."""
        log.debug(f"Node: {node}")  
        if hasattr(node, "description") and node.description == None:
            log.debug(f"The column name is: {column_name}")
            log.debug(f"We are attempting to update a missing description: {node}")
            if new_value: 
                if use_column_name: 
                    new_value = new_value.format(column_name)
                    log.debug(f"The new value to update is: {new_value} (column name: {column_name})")
                    node.description = StringValueNode(value=new_value)

        for attr in dir(node):
            if attr.startswith("__") or attr == "description":
                continue
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, Node):
                        if isinstance(item, FieldDefinitionNode) or isinstance(item, EnumValueDefinitionNode): 
                            column_name = item.name.value
                            log.debug(f"Column name: {column_name}")
                        self.fill_empty_descriptions(item, new_value, use_column_name, column_name)
            elif isinstance(child, Node):
                if isinstance(child, FieldDefinitionNode) or isinstance(child, EnumValueDefinitionNode):
                    column_name = child.name.value
                    log.debug(f"Column name: {column_name}")
                self.fill_empty_descriptions(child, new_value, use_column_name, column_name)
        return node

    def build_entity_select_all_query(self, ast: DocumentNode, type_name: str) -> str:
        """
        Builds a GraphQL query string from an AST for a given type, including all fields.
        Works directly with the AST to avoid schema validation issues.

        Args:
            ast: The parsed GraphQL AST
            type_name: Name of the type to query (e.g., "CollectionDailySnapshot")

        Returns:
            A GraphQL query string
        """
        query_name = f"{type_name}s"
        query_name = query_name[0].lower() + query_name[1:]

        # Find the type definition in the AST
        type_def = None
        for definition in ast.definitions:
            if (
                isinstance(definition, ObjectTypeDefinitionNode)
                and definition.name.value == type_name
            ):
                type_def = definition
                break

        if not type_def:
            raise ValueError(f"Type {type_name} not found in AST")

        # build query parts for each field
        field_queries = []

        for field in type_def.fields:
            field_name = field.name.value
            field_type = field.type

            # iterate through NonNull or List wrappers
            while hasattr(field_type, "type"):
                field_type = field_type.type

            # if it's a named type (potentially an entity), just get its id
            if hasattr(field_type, "name"):
                type_name = field_type.name.value
                if type_name not in [
                    "ID",
                    "String",
                    "Int",
                    "Float",
                    "Boolean",
                    "BigInt",
                    "BigDecimal",
                ]:
                    field_queries.append(f"{field_name} {{ id }}")
                else:
                    field_queries.append(field_name)
            else:
                field_queries.append(field_name)

        # combine all field queries=
        fields_str = "\n    ".join(field_queries)

        # build the query with the entity name directly
        query = f"""{{{query_name}(first: 5) {{{fields_str}}}}}"""
        return query

    def get_all_select_queries(self, schema_ast: DocumentNode):
        select_queries = {}

        for definition in schema_ast.definitions:
            if isinstance(definition, ObjectTypeDefinitionNode):
                type_name = definition.name.value
                select_query = self.build_entity_select_all_query(schema_ast, type_name)
                select_queries[type_name] = select_query

        return select_queries
