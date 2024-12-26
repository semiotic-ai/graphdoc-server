# system packages 
import json
import logging
from abc import ABC, abstractmethod

# internal packages 

# external packages
from openai import OpenAI

# configure logging
logging.basicConfig(level=logging.INFO)

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