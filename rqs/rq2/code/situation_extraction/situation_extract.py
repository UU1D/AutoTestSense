from llm_interface import LLMInterface
from string import Template
import configparser
import os


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'))
api_key = config.get('S', 'api_key')
model_name = config.get('S', 'model_name')
url = config.get('S', 'url')
enable_thinking = config.getboolean('S', 'enable_thinking')

# Placeholder note: the original prompt slot `{见图片}` becomes the Template
# placeholder `$image`. Since an image cannot be inlined as text, `get_output`
# replaces `$image` with a short "see image" note, while the real screenshot is
# passed through `get_response_from_lm([image], prompt)`'s image list (the same
# way bug_detect.py / state_inference.py pass images).
#
# `$pagepath` (the per-step page + action text description) is optional. When
# provided it is inlined into the prompt to align steps to screenshots and to
# help understand the operational relationships among steps.
prompt = """
You are a GUI test-situation extractor.

The input is a time-ordered sequence of GUI operations together with the
corresponding page screenshots.

Your task is not to summarize application functionality, nor to recount the
specific operation steps, but to identify one or more reproducible **test
situations** from the operation sequence.

## What is a test situation

A test situation describes:

**A group of related operations that a user performs within some operational
relationship or state context.**

It typically reflects at least one of the following relationships:

- One operation is followed by another operation;
- The same kind of operation is executed consecutively or repeatedly;
- Operations continue after a state has changed;
- Operations are executed again after returning, exiting, switching, or being
  interrupted;
- Consecutive operations across multiple nested pages or multiple interfaces;
- Follow-up operations performed on the results of previous operations.

A test situation focuses on the **relationships and context among operations**,
rather than any single function itself.

For example:

Concrete operations:
Open menu -> click Exit -> launch the app again

Test situation:
Relaunching the app after exiting it

Concrete operations:
Perform a search -> modify the input -> search again

Test situation:
Re-executing a search after modifying the input

Concrete operations:
Enter page A -> page B -> page C -> go back

Test situation:
Navigating into multiple nested pages and then returning

Concrete operations:
Switch the list layout -> continue browsing content

Test situation:
Continuing to browse after switching the display mode

## What is not a test situation

Do not treat a single function or a single action as a test situation.

The following descriptions are too close to being a "function" and should not
be output:

- Searching for content
- Opening a file
- Switching views
- Browsing a directory
- Modifying settings
- Launching an app

Unless these operations form a clear operational relationship or state context
with the operations before and after them.

For example:

"Switching views" is not an ideal test situation.

"Continuing to browse after switching the display mode" is a test situation.

"Searching for content" is not an ideal test situation.

"Re-executing a search after modifying the input" is a test situation.

## Situation partitioning rules

1. A sequence of operations can contain one or more test situations.
2. Group consecutive operations that jointly form the same operational
   relationship or state context into one situation.
3. When the operational relationship, primary context, or state changes
   noticeably, a new situation may begin.
4. Do not partition mechanically by every single click.
5. Auxiliary actions such as opening menus, expanding options, and confirming
   should be merged with their core operation.
6. A situation can contain only a few operations, or many steps.
7. The same test situation may appear repeatedly within a sequence, described
   with the same wording.
8. The step ranges of different test situations must not overlap.
9. Screenshots that contain no actual user operation and only present a result
   do not constitute a situation on their own.

## Abstraction rules

When generating the situation description:

1. Remove the specific application name.
2. Remove the specific page, file, directory, menu, and button names.
3. Remove the specific input content, file names, paths, and data values.
4. Remove implementation details such as coordinates, positions, and colors.
5. Do not describe expected results, actual results, abnormal causes, or defects.
6. Preserve the core operation types and the relationships of order, repetition,
   return, switching, re-execution, or state change among operations.
7. Do not infer user intent that is not clearly reflected in the input.
8. Do not abstract into overly broad descriptions such as "using a function",
   "operating the app", or "browsing pages".
9. Prefer expressions that convey the operational relationship, such as
   "... then ...", "... again ...", "consecutively ...", "after returning ...",
   "after switching ...".
10. Keep the description short and objective, usually no more than about 20
    words.

## Judging criteria

After generating each situation, check:

**If this description merely answers "What function did the user use?", the
abstraction is not qualified.**

A qualified description should be closer to answering:

**"What behavior did the user perform under what operational relationship or
state context?"**

## Input

Operation step texts (use to align steps with screenshots):
$pagepath

Screenshots:
$image

## Output

Output only valid JSON:

{
  "test_situations": [
    {
      "start_step": 1,
      "end_step": 3,
      "situation_description": "test situation description"
    }
  ]
}
"""



class SituationExtractInterface:
    def __init__(self):
        self.prompt = Template(prompt)
        self.LLM_interface = LLMInterface(api_key, model_name, url, temperature=0, enable_thinking=enable_thinking)

    def get_output(self, image, pagepath=None):
        prompt = self.prompt.substitute(
            pagepath=pagepath or "[No text description provided]",
            image="see image",
        )
        output, token_usage, prompt_tokens, completion_tokens = self.LLM_interface.get_response_from_lm(
            [image],
            prompt,
            return_json=False,
        )
        return output, token_usage, prompt_tokens, completion_tokens