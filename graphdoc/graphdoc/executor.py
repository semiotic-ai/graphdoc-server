# system packages
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

# internal packages

# external packages
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from jinja2 import Environment, FileSystemLoader
from tokencost import (
    calculate_prompt_cost,
    calculate_completion_cost,
    count_string_tokens,
)

# configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


#######################################
### Data Structures                 ###
#######################################
@dataclass
class PromptCost:
    model: str
    prompt_tokens: int
    prompt_cost: float
    response_tokens: int
    response_cost: float

    def total_cost(self) -> float:
        return self.prompt_cost + self.response_cost


#######################################
### Language Models                 ###
#######################################


class LanguageModel(ABC):
    @abstractmethod
    def prompt(self, prompt, model, temperature):
        pass

    @abstractmethod
    def parse_response(self, response):
        pass

    @abstractmethod
    def parse_json_format_response(self, response):
        pass

    @abstractmethod
    def return_prompt_cost(self, response):
        pass


class OpenAILanguageModel(LanguageModel):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        logging.info(f"Initialized LanguageModel")

    def prompt(self, prompt, model="gpt-4o", temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                temperature=temperature,
            )
            logging.info(f"Prompt successful")
            return response
        except Exception as e:
            logging.error(f"Prompt failed: {e}")

    def parse_response(self, response):
        return response.choices[0].message.content

    def parse_json_format_response(self, response):
        response = response.choices[0].message.content
        response = response.strip("```json").strip("```").strip()
        response = json.loads(response)
        return response

    def return_prompt_cost(self, prompt: str, response: ChatCompletion) -> PromptCost:
        """
        Attempts to calculate the cost of the prompt and response.

        :param prompt: The prompt to be calculated.
        :type prompt: str
        :param response: The response to be calculated.
        :type response: ChatCompletion
        :return: The cost of the prompt and response (in USD).
        :rtype: PromptCost
        """
        try:
            model = response.model
            prompt_tokens = count_string_tokens(prompt, model)
            prompt_cost = calculate_prompt_cost(prompt, model)
            response_tokens = count_string_tokens(
                response.choices[0].message.content, model
            )
            response_cost = calculate_completion_cost(
                response.choices[0].message.content, model
            )
            return PromptCost(
                model, prompt_tokens, prompt_cost, response_tokens, response_cost
            )
        except Exception as e:
            logging.error(f"Cost calculation failed: {e}")


#######################################
### Prompt Executors                ###
#######################################


class PromptExecutor(ABC):
    def __init__(
        self,
        language_model: LanguageModel,
        prompt_templates_dir=None,  # "prompts/",
    ):
        """
        :param language_model: the language model you choose to use
        :param prompt_templates_dir: the directory for the prompts you are using
        """

        # set the language model
        self.language_model = language_model

        # Set the absolute or relative path for the prompts directory
        if prompt_templates_dir is None:
            prompt_templates_dir = Path(__file__).parent / "prompts/"

        # prompt_dir = Path(prompt_templates_dir)
        if not prompt_templates_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found at: {prompt_templates_dir}"
            )

        # load in our jinja based prompt templates
        self.prompt_templates_dir_path = prompt_templates_dir
        self.prompt_templates_dir = Environment(
            loader=FileSystemLoader(prompt_templates_dir)
        )

    def get_prompt_template(self, template_name: str):
        try:
            return self.prompt_templates_dir.get_template(template_name)
        except:
            raise FileNotFoundError(
                f"Prompt template not found for: {Path(__file__).parent / 'prompts/' / {template_name}}"
            )

    def instantiate_prompt(self, template_name: str, template_variables: dict):
        prompt_template = self.get_prompt_template(template_name)
        return prompt_template.render(template_variables)

    def execute_prompt(
        self, template_name: str, template_variables: dict, temperature=0.7
    ):
        instantiated_prompt = self.instantiate_prompt(template_name, template_variables)
        log.debug(f"Instantiated prompt: {instantiated_prompt}")
        return self.language_model.prompt(instantiated_prompt, temperature=temperature)


class EntityComparisonPromptExecutor(PromptExecutor):

    def format_entity_comparison_revision_prompt(self, response):
        response = self.language_model.parse_json_format_response(response)
        revised_prompt = response["modified_prompt"]

        # Replace the placeholder with Jinja syntax
        revised_prompt = revised_prompt.replace(
            r"{ entity_pred }", r"{{ entity_pred }}"
        )
        revised_prompt = revised_prompt.replace(r"{entity_pred}", r"{{ entity_pred }}")
        revised_prompt = revised_prompt.replace(
            r"{ entity_gold }", r"{{ entity_gold }}"
        )
        revised_prompt = revised_prompt.replace(r"{entity_gold}", r"{{ entity_gold }}")
        return revised_prompt

    def template_variablest_from_four_comparisons(
        self,
        original_prompt_template,
        four_comparison,
        three_comparison,
        two_comparison,
        one_comparison,
    ):
        """
        This is function is intended to help with rendering the template in scenarios where four comparisons were made (intended for help with prompt optimization).
        _comparison = {
            "reasoning": "the reasoning behind the score of the comparison between the gold and predicted entity",
            "correctness": "a score from 1-4, where 4 is the best and 1 is the worst",
        }
        """
        return {
            "original_prompt": {original_prompt_template},
            "four_result_correct": {
                "True" if four_comparison["correctness"] == 4 else "False"
            },
            "four_result": {four_comparison["reasoning"]},
            "three_result_correct": {
                "True" if three_comparison["correctness"] == 3 else "False"
            },
            "three_result": {three_comparison["reasoning"]},
            "two_result_correct": {
                "True" if two_comparison["correctness"] == 2 else "False"
            },
            "two_result": {two_comparison["reasoning"]},
            "one_result_correct": {
                "True" if one_comparison["correctness"] == 1 else "False"
            },
            "one_result": {one_comparison["reasoning"]},
        }

    def execute_four_comparison_prompt(
        self,
        original_prompt_template,
        four_comparison,
        three_comparison,
        two_comparison,
        one_comparison,
        template_name="entity_comparison_revision.txt",
        temperature=0.7,
    ):
        template_variables = self.template_variablest_from_four_comparisons(
            original_prompt_template,
            four_comparison,
            three_comparison,
            two_comparison,
            one_comparison,
        )
        return self.execute_prompt(
            template_name,
            template_variables=template_variables,
            temperature=temperature,
        )
