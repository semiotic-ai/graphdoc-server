# system packages 
import os
import json
import logging
from pathlib import Path

# internal packages 

# external packages
from openai import OpenAI
from dotenv import load_dotenv
from graphql import build_schema, parse
from jinja2 import Environment, FileSystemLoader

# configure logging
logging.basicConfig(level=logging.INFO)

class LanguageModel: 
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        logging.info(f"Initialized LanguageModel")

    def prompt(self, prompt, model="gpt-4o"):
        try: 
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
            )
            logging.info(f"Prompt successful")
            return response
        except Exception as e:
            logging.error(f"Prompt failed: {e}")
    
    @staticmethod
    def parse_response(response): 
        response = response.choices[0].message.content
        response = response.strip('```json').strip('```').strip()
        response = json.loads(response)
        return response

class GraphDoc:
    def __init__(
            self, 
            language_model,
            prompt_templates_dir = None, # "prompts/",
        ):
        
        # set the language model
        self.language_model = language_model

        # Set the absolute or relative path for the prompts directory
        if prompt_templates_dir is None:
            prompt_templates_dir = Path(__file__).parent / 'prompts/'
        
        prompt_dir = Path(prompt_templates_dir)
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found at: {prompt_dir}")
        
        # load in our jinja based prompt templates
        self.entity_comparison_prompt_env = Environment(loader=FileSystemLoader(prompt_templates_dir))
        self.entity_comparison_prompt_template = self.entity_comparison_prompt_env.get_template("entity_comparison_prompt.txt") # TODO: we can move this out to a config file
        self.entity_comparison_prompt_revision_template = self.entity_comparison_prompt_env.get_template("entity_comparison_revision.txt") # TODO: we can move this out to a config file
        
        logging.info(f"Initialized GraphDoc")

    def instantiate_entity_comparison_prompt(self, entity_gold, entity_pred):
        return self.entity_comparison_prompt_template.render({"entity_gold": {entity_gold}, "entity_pred": {entity_pred}})
    
    def prompt_entity_comparison(self, entity_gold, entity_pred):
        prompt = self.instantiate_entity_comparison_prompt(entity_gold, entity_pred)
        response = self.language_model.prompt(prompt)
        return response

    def format_entity_comparison_revision_prompt(self, response): 
        response = self.language_model.parse_response(response)
        revised_prompt = response["modified_prompt"]

        # Replace the placeholder with Jinja syntax
        revised_prompt = revised_prompt.replace(r"{entity_pred}", r"{{ entity_pred }}")
        revised_prompt = revised_prompt.replace(r"{entity_gold}", r"{{ entity_gold }}")
        return revised_prompt

    # TODO: we could refactor this to either be more generic or take a more specific configuration
    def instantiate_entity_comparison_revision_prompt( 
            self, 
            original_prompt_template, 
            four_comparison,
            three_comparison,
            two_comparison,
            one_comparison,
    ): 
        """
        _comparison = {
            "reasoning": "the reasoning behind the score of the comparison between the gold and predicted entity",
            "correctness": "a score from 1-4, where 4 is the best and 1 is the worst",
        }
        """
        return self.entity_comparison_prompt_revision_template.render(
            {
                "original_prompt": {original_prompt_template}, 

                "four_result_correct": {"True" if four_comparison["correctness"] == 4 else "False"},
                "four_result": {four_comparison["reasoning"]},
                
                "three_result_correct": {"True" if three_comparison["correctness"] == 3 else "False"},
                "three_result": {three_comparison["reasoning"]},
                
                "two_result_correct": {"True" if two_comparison["correctness"] == 2 else "False"},
                "two_result": {two_comparison["reasoning"]},
                
                "one_result_correct": {"True" if one_comparison["correctness"] == 1 else "False"},
                "one_result": {one_comparison["reasoning"]},
            }
        )
    
    def prompt_entity_comparison_revision(self, original_prompt_template, four_comparison, three_comparison, two_comparison, one_comparison):
        prompt = self.instantiate_entity_comparison_revision_prompt(original_prompt_template, four_comparison, three_comparison, two_comparison, one_comparison)
        response = self.language_model.prompt(prompt)
        return response