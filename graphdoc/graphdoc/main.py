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
        logging.info(f"Initialized GraphDoc")

    def generate_entity_comparison_prompt(self, entity1, entity2):
        return self.entity_comparison_prompt_template.render(entity1=entity1, entity2=entity2)
    
    def prompt_entity_comparison(self, entity1, entity2):
        prompt = self.generate_entity_comparison_prompt(entity1, entity2)
        response = self.language_model.prompt(prompt)
        extracted_response = response.choices[0].message.content
        return extracted_response