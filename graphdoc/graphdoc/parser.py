# system packages

# internal packages
from pathlib import Path
from typing import Optional

from graphql import parse
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
    # Methods         #
    ###################
    def parse_schema_from_file(self, schema_file: str, schema_directory_path: Optional[str] = None) -> DocumentNode: 
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
        else:
            schema_directory_path = self.schema_directory_path
        
        schema_path = Path(schema_directory_path) / schema_file
        check_file_path(str(schema_path))
        
        schema = schema_path.read_text()
        return parse(schema)