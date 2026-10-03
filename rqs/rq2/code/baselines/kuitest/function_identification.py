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
Here is a UI screenshot:

The component I selected is marked with a red bounding box in the image. The action description is: $action_description
Infer the function of the selected UI component based on the screenshot in one concise sentence.
Please ensure that your function predictions reflect high confidence.
"""

prompt_with_page_description = """
Here is a UI screenshot:

The component I selected is marked with a red bounding box in the image. The action description is: $action_description
The textual description of the current pre-interaction page is provided as additional context:
Current page description: $page_description

Infer the function of the selected UI component based on the screenshot, action description, and current page description in one concise sentence.
Please ensure that your function predictions reflect high confidence.
"""

class FunctionIdentification:
    def __init__(self):
        self.llm_interface = LLMInterface(
            api_key,
            model_name,
            url,
            temperature=0,
            enable_thinking=enable_thinking,
        )
        self.prompt = Template(prompt)
        self.prompt_with_page_description = Template(prompt_with_page_description)

    def build_prompt(self, action_description, page_description=None):
        if page_description is None:
            return self.prompt.substitute(action_description=action_description)
        return self.prompt_with_page_description.substitute(
            action_description=action_description,
            page_description=page_description,
        )

    def func_identification(
        self,
        anno_image_path,
        action_description,
        page_description=None,
    ):
        prompt = self.build_prompt(action_description, page_description)
        output, token_usage, prompt_tokens, completion_tokens = self.llm_interface.get_response_from_lm([anno_image_path], prompt)
        return output, token_usage, prompt_tokens, completion_tokens
