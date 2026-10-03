"""Image-only situation extraction for the VanillaMLLM baseline."""

import configparser
import os

from llm_interface import LLMInterface


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.ini')

config = configparser.ConfigParser()
if not config.read(CONFIG_PATH):
    raise FileNotFoundError(f'Config file not found: {CONFIG_PATH}')

api_key = config.get('S', 'api_key')
model_name = config.get('S', 'model_name')
url = config.get('S', 'url')
enable_thinking = config.getboolean('S', 'enable_thinking', fallback=False)


prompt = """
You are a GUI test-situation extractor.
Your only case-specific input is one image containing a time-ordered sequence of GUI screenshots arranged from left to right and then from top to bottom.
Red bounding boxes indicate the UI elements operated on by the user. The final screenshot may show only the result of the preceding operation and may contain no new operation.

Base every case-specific claim only on visible evidence in the image.

Complete the task in two phases.

## Phase 1: visually reconstruct the operations

1. Identify the ordered screenshot panels using reading order from left to right and then from top to bottom.
2. For every visible red-box operation, assign a consecutive operation `step_id` starting from 1.
3. Infer a concise operation description from the highlighted widget, visible labels or icons, and the change in the following screenshot.
4. Treat the screenshot containing the red box as the source frame and the following screenshot as the response or target frame.
5. Do not invent exact labels, values, gestures, or targets that are not visually supported. Use a cautious generic description when details are unreadable.
6. Do not count a final result-only screenshot as a separate operation.

## Phase 2: derive test situations

A `situation` is the abstract recurring situation in which an expectation applies.
A test situation is a group of related operations performed within an operational relationship or state context.
It should capture relationships such as sequencing, repetition, continuing after a state change, returning, switching, interruption, re-execution, nested navigation, or follow-up actions on an earlier result.

A single isolated function or action is not normally a test situation.
For example, "searching for content" is only a function, while "searching again after modifying the input" is a situation.

## Examples

Concrete operations: Open a menu, select Exit, and launch the application again.
Situation: Relaunching an application after exiting it.

Concrete operations: Perform a search, modify the input, and perform the search again.
Situation: Re-executing a search after modifying the input.

Concrete operations: Enter one page, enter two deeper nested pages, and then go back.
Situation: Navigating into multiple nested pages and then returning.

Concrete operations: Switch the list layout and continue browsing content.
Situation: Continuing to browse after switching the display mode.

The following descriptions are functions rather than qualified situations: "Searching for content", "Opening a file", "Switching views", "Browsing a directory", "Modifying settings", and "Launching an application".

Partition and abstract the visually reconstructed sequence using these rules:

1. Group consecutive operations that jointly express one relationship or state context.
2. Start a new situation only when the main relationship or context changes.
3. Merge auxiliary actions such as opening a menu or confirming an operation with the core operation.
4. Situation step ranges must refer to Phase 1 `step_id` values and must not overlap.
5. A result-only screenshot does not form a situation by itself.
6. Remove application names, project names, specific pages, files, directories, menus, buttons, input values, coordinates, positions, and colors from situation descriptions.
7. Do not describe expected results, actual results, defects, or abnormal causes.
8. Preserve the essential relationship between operations, using wording such as "then", "again", "consecutively", "after returning", or "after switching".
9. Do not infer a user intention that is not visually supported.
10. Keep each situation description short and objective, normally no more than about 20 words.
11. Prefer an abstraction that can recur in other applications or projects.

Before returning, verify that each situation describes the abstract recurring context in which an expectation applies.
Before returning, verify that each situation answers "what behavior occurred under what operational relationship or state context?", rather than merely "what function was used?"

## Output

Return only valid JSON in this format:

{
  "recognized_operations": [
    {
      "step_id": 1,
      "source_frame": 1,
      "target_frame": 2,
      "operation_description": "concise visually grounded operation",
      "visual_evidence": "brief evidence from the highlighted widget and transition"
    }
  ],
  "test_situations": [
    {
      "start_step": 1,
      "end_step": 3,
      "situation_description": "abstract test situation description"
    }
  ]
}
"""


class SituationExtractInterface:
    def __init__(self):
        self.prompt = prompt
        self.llm_interface = LLMInterface(
            api_key,
            model_name,
            url,
            temperature=0,
            enable_thinking=enable_thinking,
        )

    def build_prompt(self):
        return self.prompt

    def get_output(self, image):
        return self.llm_interface.get_response_from_lm(
            [image],
            self.build_prompt(),
            return_json=False,
        )
