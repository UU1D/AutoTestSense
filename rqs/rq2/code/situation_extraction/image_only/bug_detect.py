from llm_interface import LLMInterface
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
Detector Prompt Template

Below are instructions for the no-crash bug detection task.

[Task Description] You are given:
1. A concatenated screenshot strip arranged from left to right and top to bottom.
2. Annotated colored bounding boxes indicating user actions.

Your goal is to determine whether any non-crash functional bug exists in this tested sequence.

------------------------------------------------------------

OUTPUT FORMAT (STRICT JSON)

Return ONLY a JSON object in the following format:

{ “function_name”: “…”, “expected_path_summary”: “…”, “step_analysis”: [
{ “step_id”: 1, “expected_behavior”: “…”, “actual_behavior”: “…”,
“meets_expectation”: true, “reason”: “…” } ], “bug_found”: true,
“bug_page_step”: 4, “bug_type”: “functional_mismatch | missing_effect |
wrong_navigation | state_inconsistency | input_validation | other |
none”, “bug_description”: “…”, “possible_bug_paths”: [ { “page_step”: 3,
“action_widget”: “…”, “risk_reason”: “…” } ], “exception_paths”: [ {
“page_step”: 2, “action_widget”: “…”, “abnormal_reason”: “…” } ],
“confidence”: 0.0 }

Rules: - Do not hallucinate widgets not shown in the legend or steps. -
If no bug is found, set “bug_found”: false and “bug_type”: “none”. -
Keep reasoning concise but logically grounded in provided evidence.
"""


class BugDetectInterface:
    def __init__(self):
        self.prompt = prompt
        self.LLM_interface = LLMInterface(
            api_key, model_name, url, temperature=0, enable_thinking=enable_thinking
        )

    def build_prompt(self):
        return self.prompt

    def get_output(self, image):
        output, token_usage, prompt_tokens, completion_tokens = self.LLM_interface.get_response_from_lm(
            [image],
            self.build_prompt(),
            return_json=False,
        )
        return output, token_usage, prompt_tokens, completion_tokens
