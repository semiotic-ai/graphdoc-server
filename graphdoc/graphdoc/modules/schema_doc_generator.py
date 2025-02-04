# system packages
import logging
from typing import Union

# internal packages
from ..prompts import SinglePrompt
from ..parser import Parser

# external packages
import dspy
from graphql import parse, print_ast

# configure logging
log = logging.getLogger(__name__)


class DocGeneratorModule(dspy.Module):
    def __init__(
        self,
        generator_prompt: SinglePrompt,
        fill_empty_descriptions: bool = True,
        retry_limit: int = 3,
    ) -> None:
        self.generator_prompt = generator_prompt
        self.fill_empty_descriptions = fill_empty_descriptions

        self.par = Parser()
        # signature fields are:
        # database_schema: str = dspy.InputField()
        # documented_schema: str = dspy.OutputField()

    def forward(
        self, database_schema: str
    ) -> Union[
        dspy.Prediction, None
    ]:  # TODO: we should probably replace what is here with document_full_schema
        # check that the graphql is valid
        try:
            database_ast = parse(database_schema)
        except Exception as e:
            log.warning(f"Invalid GraphQL schema provided at onset: {e}")
            return dspy.Prediction(documented_schema=database_schema)

        # fill the empty descriptions
        if self.fill_empty_descriptions:
            updated_ast = self.par.fill_empty_descriptions(database_ast)
            database_schema = print_ast(updated_ast)

        # try to generate the schema
        try:
            prediction = self.generator_prompt.infer(database_schema=database_schema)
        except Exception as e:
            log.warning(f"Error generating schema: {e}")
            return dspy.Prediction(documented_schema=database_schema)

        # check that the generated schema is valid
        try:
            prediction_ast = parse(prediction.documented_schema)
        except Exception as e:
            log.warning(f"Invalid GraphQL schema generated: {e}")
            return dspy.Prediction(documented_schema=database_schema)

        # check that the generated schema matches the original schema
        if self.par.schema_equality_check(database_ast, prediction_ast):
            return dspy.Prediction(documented_schema=prediction.documented_schema)
        else:
            log.warning(f"Generated schema does not match the original schema")
            return dspy.Prediction(
                documented_schema=database_schema
            )  # we should handle retry logic here

    def document_full_schema(
        self, database_schema: str
    ) -> Union[dspy.Prediction, None]:
        try:
            document_ast = parse(database_schema)
        except Exception as e:
            raise ValueError(f"Invalid GraphQL schema provided: {e}")

        examples = []
        for node in document_ast.definitions:
            example = dspy.Example(
                database_schema=print_ast(node), documented_schema=""
            ).with_inputs("database_schema")
            examples.append(example)

        documented_examples = self.batch(examples, num_threads=32)
        document_ast.definitions = tuple(
            parse(ex.documented_schema) for ex in documented_examples  # type: ignore # TODO: we should have better type handling, but we know this works
        )

        if self.par.schema_equality_check(parse(database_schema), document_ast):
            log.info("Schema equality check passed, returning documented schema")
            # log.info(f"Documented schema: {print_ast(document_ast)}")
            return dspy.Prediction(documented_schema=print_ast(document_ast))
        else:
            log.warning(f"Generated schema does not match the original schema")
            if self.fill_empty_descriptions:
                updated_ast = self.par.fill_empty_descriptions(document_ast)
                return dspy.Prediction(documented_schema=print_ast(updated_ast))
            return dspy.Prediction(documented_schema=database_schema)

    # def document_batch(self, examples: List[dspy.Example]):
