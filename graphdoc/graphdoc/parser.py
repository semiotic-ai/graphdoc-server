# system packages

# internal packages
from typing import Optional
from .helper import check_directory_path

# external packages


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
    # def
