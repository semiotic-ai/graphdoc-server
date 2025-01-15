# system packages
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# internal packages
from .executor import (
    LanguageModel,
    OpenAILanguageModel,
    EntityComparisonPromptExecutor,
    PromptExecutor,
    PromptCost,
)
from .prompt import Prompt, PromptRevision, RequestObject
from .parser import Parser

# external packages
from openai import OpenAI
from dotenv import load_dotenv
from graphql import build_schema, parse, print_ast
from jinja2 import Environment, FileSystemLoader

# configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class GraphDoc:
    def __init__(
        self, openai_api_key: str, schema_directory_path: Optional[str] = None
    ):
        """
        Initialize the GraphDoc class. This is the main class for the GraphDoc package.

        :param openai_api_key: The OpenAI API key to be used for the language model.
        :type openai_api_key: str
        :param schema_directory_path: The path to the directory containing the database schema files.
        :type schema_directory_path: Optional[str]
        """

        # set up the openai language model
        self.openai_api_key = openai_api_key
        self.openai_lm = OpenAILanguageModel(api_key=openai_api_key)
        self.openai_ex = PromptExecutor(
            language_model=self.openai_lm
        )  # prompt_templates_dir defaults to prompts/

        # set up the parser
        self.parser = Parser(schema_directory_path=schema_directory_path)

    ####################
    # LLM Methods      #
    ####################
    def schema_doc_prompt(
        self,
        database_schema: str,
        template_name: str = "schema_generation_prompt.txt",
        temperature: float = 0.7,
        with_cost: bool = True,
    ):
        """Given a database schema, request that the llm generate a schema documentation.

        :param database_schema: The database schema to be documented.
        :type database_schema: str
        :param template_name: The name of the template to be used for the prompt.
        :type template_name: str
        :param temperature: The temperature to be used for the prompt.
        :type temperature: float
        """
        response = self.openai_ex.execute_prompt(
            template_name=template_name,
            template_variables={
                "database_schema": database_schema,
            },
            temperature=temperature,
        )
        if with_cost:
            cost = self.openai_lm.return_prompt_cost(
                prompt=self.openai_ex.instantiate_prompt(
                    template_name=template_name,
                    template_variables={
                        "database_schema": database_schema,
                    },
                ),
                response=response,
            )
            response.prompt_cost = cost
        return response

    def schema_doc_prompt_from_file(
        self,
        schema_file: str,
        template_name: str = "schema_generation_prompt.txt",
        temperature: float = 0.7,
        with_cost: bool = True,
        schema_directory_path: Optional[str] = None,
    ):
        """Given a database schema file, request that the llm generate a schema documentation.

        :param database_schema_file: The path to the database schema file to be documented.
        :type database_schema_file: str
        :param template_name: The name of the template to be used for the prompt.
        :type template_name: str
        :param temperature: The temperature to be used for the prompt.
        :type temperature: float
        """
        schema = self.parser.parse_schema_from_file(
            schema_file=schema_file, schema_directory_path=schema_directory_path
        )
        schema_str = print_ast(schema)
        return self.schema_doc_prompt(
            database_schema=schema_str,
            template_name=template_name,
            temperature=temperature,
            with_cost=with_cost,
        )

    def schema_doc_prompt_with_equality(
        self,
        database_schema: str,
        template_name: str = "schema_generation_prompt.txt",
        retries: int = 3,
        temperature: float = 0.7,
        with_cost: bool = True,
    ):
        """
        Given a database schema string, request that the llm generate a schema documentation. Check for schema equality. Retry if not equal (until retry limit is hit).

        :param database_schema: The database schema to be documented.
        :type database_schema: str
        :param template_name: The name of the template to be used for the prompt.
        :type template_name: str
        :param retries: The number of times to retry if the schema is not equal.
        :type retries: int
        :param temperature: The temperature to be used for the prompt.
        :type temperature: float
        :param with_cost: Whether to return the cost of the prompt.
        :type with_cost: bool
        """
        gold_schema = parse(database_schema)

        for i in range(retries):
            log.debug(f"Retrying. Iteration: {i}")
            response = self.schema_doc_prompt(
                database_schema=database_schema,
                template_name=template_name,
                temperature=temperature,
                with_cost=with_cost,
            )
            response_str = self.openai_lm.parse_response(response)

            try:
                log.debug("Attempting to parse response schema.")
                response_schema = parse(response_str)
            except:
                log.debug(
                    f"Failed to parse response schema. Retries: {i}. Response: {response_str}"
                )
                continue

            if self.parser.schema_equality_check(gold_schema, response_schema):
                log.debug(f"Schema equal. Retries: {i}")
                return response
            else:
                log.debug(f"Schema not equal. Retries: {i}")
                log.debug(f"Gold schema: {print_ast(gold_schema)}")
                log.debug(f"Response schema: {response_str}")
                continue

        return "Retries exceeded. Schema not equal."
