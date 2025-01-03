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

        # load the model token details from tokencost
        package_name = "tokencost"
        resource_name = "model_prices.json"

        try:
            with pkg_resources.files(package_name).joinpath(resource_name).open('r') as file:
                self.model_data = json.load(file)
        except FileNotFoundError:
            raise ValueError(f"{resource_name} not found in the package {package_name}.")

    def parse_schema_from_text(self, schema_text: str):
        schema_ast = parse(schema_text)
        return schema_ast

    def parse_schema_from_file(self, schema_file: str, schema_directory_path: Optional[str] = None):
        if schema_directory_path:
            schema_directory_path = Path(schema_directory_path).resolve()
            if not schema_directory_path.is_dir():
                raise ValueError(f"The provided schema directory path '{schema_directory_path}' is not a valid directory.")
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
            logging.warning(f"Prompt token count {prompt_token_count} exceeds the maximum input tokens {max_input_tokens} for model {model}.")
            return False
        return True
    
    def format_text_schema_to_json(self, schema_text, label="gold", version="1.0.0"): 
        lines = schema_text.strip().split('\n')
        formatted_lines = []
    
        for line in lines:
            cleaned_line = line.replace('"', '\"').rstrip()
        
            if not cleaned_line:
                formatted_lines.append("\n")
                continue
            
        formatted_lines.append(cleaned_line + " \n")
        json_output = {
            f"{label}": {
                "version": f"{version}",
                "prompt": formatted_lines
            }
        }
        return json_output
    
    def format_schema_ast_to_json(self, schema_ast, label="gold", version="1.0.0"):
        schema_str = print_ast(schema_ast)
        return self.format_text_schema_to_json(schema_str, label, version)
    
    def format_schema_file_to_json(self, schema_file, schema_directory_path=None, label="gold", version="1.0.0"):
        schema_ast = self.parse_schema_from_file(schema_file, schema_directory_path)
        return self.format_schema_ast_to_json(schema_ast, label, version)
