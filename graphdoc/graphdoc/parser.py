# system packages 
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional

# internal packages 

# external packages
from tokencost import count_string_tokens
from graphql import build_schema, parse, build_ast_schema, validate_schema, print_ast


from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

# configure logging
logging.basicConfig(level=logging.INFO)

class Parser: 
    def __init__(self, schema_directory_path: Optional[str] = None): 
        if schema_directory_path:
            schema_directory_path = Path(schema_directory_path).resolve()
            if not schema_directory_path.is_dir():
                raise ValueError(f"The provided schema directory path '{schema_directory_path}' is not a valid directory.")
        self.schema_directory_path = schema_directory_path

    def parse_schema(self, schema_file: str, schema_directory_path: Optional[str] = None):
        if schema_directory_path:
            schema_directory_path = Path(schema_directory_path).resolve()
            if not schema_directory_path.is_dir():
                raise ValueError(f"The provided schema directory path '{schema_directory_path}' is not a valid directory.")
            else:
                self.schema_directory_path = schema_directory_path

        schema_path = Path(self.schema_directory_path) / schema_file
        schema = schema_path.read_text()
        schema_ast = parse(schema)
        return schema_ast