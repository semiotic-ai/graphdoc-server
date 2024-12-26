# system packages 
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod

# internal packages 

# external packages
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

# configure logging
logging.basicConfig(level=logging.INFO)

#######################################
### Language Models                 ###
#######################################

class LanguageModel(ABC): 
    @abstractmethod
    def prompt(self, prompt, model): 
        pass

    @abstractmethod 
    def parse_response(self, response): 
        pass

class OpenAILanguageModel(LanguageModel): 
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
    
    # @staticmethod
    def parse_response(self, response): 
        response = response.choices[0].message.content
        response = response.strip('```json').strip('```').strip()
        response = json.loads(response)
        return response
    
#######################################
### Prompt Executors                ###
#######################################

class PromptExecutor(ABC):
    def __init__(self, 
                 language_model: LanguageModel,
                 prompt_templates_dir = None, # "prompts/",
        ):
        """
        :param language_model: the language model you choose to use 
        :param prompt_templates_dir: the directory for the prompts you are using
        """

        # set the language model
        self.language_model = language_model

        # Set the absolute or relative path for the prompts directory
        if prompt_templates_dir is None:
            prompt_templates_dir = Path(__file__).parent / 'prompts/'
        
        # prompt_dir = Path(prompt_templates_dir)
        if not prompt_templates_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found at: {prompt_templates_dir}")
        
        # load in our jinja based prompt templates
        self.prompt_templates_dir = Environment(loader=FileSystemLoader(prompt_templates_dir))

    def get_prompt_template(self, template_name: str): 
        try: 
            return self.prompt_templates_dir.get_template(template_name)
        except:
            raise FileNotFoundError(f"Prompt template not found for: {Path(__file__).parent / 'prompts/' / {template_name}}")

    def instantiate_prompt(self, template_name: str, **template_variables):
        prompt_template = self.get_prompt_template(template_name)
        return prompt_template.render(template_variables)
    
    def execute_prompt(self, template_name: str, **template_variables):
        instantiated_prompt = self.instantiate_prompt(template_name, **template_variables)
        return self.language_model.prompt(instantiated_prompt)

class EntityComparisonPromptExecutor(PromptExecutor): 

    def format_entity_comparison_revision_prompt(self, response): 
        response = self.language_model.parse_response(response)
        revised_prompt = response["modified_prompt"]

        # Replace the placeholder with Jinja syntax
        revised_prompt = revised_prompt.replace(r"{entity_pred}", r"{{ entity_pred }}")
        revised_prompt = revised_prompt.replace(r"{entity_gold}", r"{{ entity_gold }}")
        return revised_prompt
    
    def template_variablest_from_four_comparisons(self,
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

                "four_result_correct": {"True" if four_comparison["correctness"] == 4 else "False"},
                "four_result": {four_comparison["reasoning"]},
                
                "three_result_correct": {"True" if three_comparison["correctness"] == 3 else "False"},
                "three_result": {three_comparison["reasoning"]},
                
                "two_result_correct": {"True" if two_comparison["correctness"] == 2 else "False"},
                "two_result": {two_comparison["reasoning"]},
                
                "one_result_correct": {"True" if one_comparison["correctness"] == 1 else "False"},
                "one_result": {one_comparison["reasoning"]},
            }
    
    def execute_four_comparison_prompt(self,
                                        original_prompt_template, 
                                        four_comparison,
                                        three_comparison,
                                        two_comparison,
                                        one_comparison,
                                        template_name = "entity_comparison_revision.txt"
    ): 
        template_variables = self.template_variablest_from_four_comparisons(
                                        original_prompt_template, 
                                        four_comparison,
                                        three_comparison,
                                        two_comparison,
                                        one_comparison,
                                )
        return self.execute_prompt(template_name, **template_variables)