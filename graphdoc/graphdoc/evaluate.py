# system packages 
from typing import Literal

# internal packages 

# external packages 
import dspy
from dspy import Signature, InputField, OutputField

class DocQuality(Signature):
    """
    Given a GraphQL Schema, evaluate the quality of documentation for that schema and provide a category rating. 
    The categories are described as: 
    - Perfect (4): The documentation contains enough information so that the interpretation of the schema and its database content is completely free of ambiguity.
    - Almost Perfect (3): The documentation is almost perfect and free from ambiguity, but there is room for improvement.
    - Somewhat Correct (2): The documentation is somewhat correct but has room for improvement due to missing information. The documentation is not incorrect.
    - Incorrect (1): The documentation is incorrect and contains inaccurate or misleading information. Any incorrect information automatically leads to an incorrect rating, even if some correct information is present.
    Output a number rating that corresponds to the categories described above.
    """

    schema: str = InputField()
    category: Literal[
        "Perfect", 
        "Almost Perfect", 
        "Somewhat Correct", 
        "Incorrect",
    ] = OutputField()
    rating: int = OutputField()