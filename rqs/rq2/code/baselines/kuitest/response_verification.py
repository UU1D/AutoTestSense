from llm_interface import LLMInterface
from string import Template
import configparser
import os

config = configparser.ConfigParser()
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
config_path = os.path.join(project_root, 'config.ini')
if not config.read(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
api_key = config.get('S', 'api_key', fallback='')
model_name = config.get('S', 'model_name')
url = config.get('S', 'url')

enable_thinking = config.getboolean('S', 'enable_thinking', fallback=False)

prompt = """
Here are the interacted component's function description, alone with the screenshot of the page after interaction:
$component_function

Determine whether the UI response after interaction aligns with the expectations?
Return ONLY a JSON object in the following format:
{
    "judgement": true/false,
    "reason": "Provide concise step-by-step reasoning",
    "confidence": 0.0 
}
"""

prompt_with_page_descriptions = """
Here are the interacted component's function description, alone with the screenshot of the page after interaction:
$component_function

The textual descriptions of the pages before and after the interaction are provided as additional context:
Pre-interaction page description: $before_page_description
Post-interaction page description: $after_page_description

Determine whether the UI response after interaction aligns with the expectations?
Return ONLY a JSON object in the following format:
{
    "judgement": true/false,
    "reason": "Provide concise step-by-step reasoning",
    "confidence": 0.0 
}
"""

class ResponseVerification:
    def __init__(self):
        self.llm_interface = LLMInterface(
            api_key,
            model_name,
            url,
            temperature=0,
            enable_thinking=enable_thinking,
        )
        self.prompt = Template(prompt)
        self.prompt_with_page_descriptions = Template(prompt_with_page_descriptions)

    def build_prompt(
        self,
        component_function,
        before_page_description=None,
        after_page_description=None,
    ):
        if before_page_description is None and after_page_description is None:
            return self.prompt.substitute(component_function=component_function)
        if before_page_description is None or after_page_description is None:
            raise ValueError('Both before and after page descriptions are required')
        return self.prompt_with_page_descriptions.substitute(
            component_function=component_function,
            before_page_description=before_page_description,
            after_page_description=after_page_description,
        )

    def response_verification(
        self,
        image,
        component_function,
        before_page_description=None,
        after_page_description=None,
    ):
        prompt = self.build_prompt(
            component_function,
            before_page_description,
            after_page_description,
        )
        output, token_usage, prompt_tokens, completion_tokens = self.llm_interface.get_response_from_lm([image], prompt, return_json=False)
        return output, token_usage, prompt_tokens, completion_tokens
