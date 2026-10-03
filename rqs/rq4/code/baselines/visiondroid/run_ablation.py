"""Run VisionDroid with flat instance-level commonsense retrieval."""

from __future__ import annotations

import os

import run_commonsense as shared
from bug_detect import BugDetectInterface
from rq4_instance_level_context import build_instance_level_context


CONTEXT_MARKER = "\nOUTPUT FORMAT (STRICT JSON)"
CONTEXT_TEMPLATE = """
  ------------------------------------------------------------
  ### [Retrieved Instance-Level Common-Sense Context]

  The test operations are grouped into situations. Each retrieved item is an
  independently extracted common-sense expectation, not a generalized rule.
  First decide whether its Situation applies to the observed interaction. If
  it applies, check whether the GUI contradicts its Expected behavior. Ignore
  retrieved items that are not applicable.

{commonsense_context}
  ------------------------------------------------------------
"""


class InstanceLevelCommonsenseInterface(BugDetectInterface):
    def build_prompt(
        self, funcname, funcdescription, funcgoal, pagepath, commonsense_context=None
    ):
        prompt = super().build_prompt(
            funcname, funcdescription, funcgoal, pagepath
        )
        if CONTEXT_MARKER not in prompt:
            raise ValueError("Could not locate OUTPUT FORMAT in baseline prompt.")
        injected = CONTEXT_TEMPLATE.format(
            commonsense_context=commonsense_context or "[None provided]"
        )
        return prompt.replace(CONTEXT_MARKER, injected + CONTEXT_MARKER, 1)

    def get_output(
        self,
        image,
        funcname,
        funcdescription,
        funcgoal,
        pagepath,
        commonsense_context=None,
    ):
        prompt = self.build_prompt(
            funcname,
            funcdescription,
            funcgoal,
            pagepath,
            commonsense_context,
        )
        return self.LLM_interface.get_response_from_lm(
            [image], prompt, return_json=False
        )


def build_context(document):
    top_k = int(os.environ.get("RQ4_INJECTION_TOP_K", "10"))
    return build_instance_level_context(document, top_k=top_k)


def main():
    shared.build_commonsense_context = build_context
    shared.BugDetectCommonsenseInterface = InstanceLevelCommonsenseInterface
    shared.main()


if __name__ == "__main__":
    main()
