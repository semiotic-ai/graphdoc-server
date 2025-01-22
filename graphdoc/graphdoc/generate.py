# system packages
import logging
from typing import Callable, List, Literal, Optional, Union

# internal packages
from .data import DataHelper

# external packages
import dspy
from dspy import Example
from dspy.evaluate import Evaluate

# logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

#################
# DSPy Modules  #
#################
class DocGenerator(dspy.Signature):
    """
    ### TASK:
    Given a GraphQL Schema, generate a precise description for the columns of the tables in the database.

    ### Requirements:
    - Focus solely on confirmed details from the provided schema.
    - Keep the description concise and factual.
    - Exclude any speculative or additional commentary.
    - DO NOT return the phrase "in the { table } table" in your description.

    ### Formatting 
    - Ensure that the schema maintains proper documentation formatting, as is provided. 
    """
    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField()