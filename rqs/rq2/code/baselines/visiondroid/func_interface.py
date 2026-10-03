#Generate a function introduction for each step.
from llm_interface import LLMInterface
from string import Template
import configparser
import os
from bug_detect import api_key, model_name, url

config = configparser.ConfigParser()
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
config.read(os.path.join(project_root, 'config.ini'))

# api_key = config.get('S', 'api_key')
# model_name = config.get('S', 'model_name')
# url = config.get('S', 'url')


enable_thinking = config.getboolean('S', 'enable_thinking', fallback=False)

prompt = """
## Task
Please make a judgment based on the following test sequence.

Test sequence:
$actual_path

What is the function currently being tested?

Return ONLY a JSON object in this exact format:
{
  "function_name": "...",
  "function_description": "...",
  "function_goal": "..."
}

Do not include any extra text, explanations, or formatting outside the JSON.
"""

class FuncInterface:
    def __init__(self):
        self.prompt = Template(prompt)
        self.LLM_interface = LLMInterface(
            api_key,
            model_name,
            url,
            enable_thinking=enable_thinking,
        )

    def build_prompt(self, actual_path):
        return self.prompt.substitute(actual_path=actual_path)

    def get_output(self,actual_path):
        prompt = self.build_prompt(actual_path)
        output, token_usage, prompt_tokens, completion_tokens = self.LLM_interface.get_response_from_lm([], prompt, return_json=False)
        return output, token_usage, prompt_tokens, completion_tokens
