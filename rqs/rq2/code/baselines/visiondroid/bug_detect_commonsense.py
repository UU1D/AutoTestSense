"""VisionDroid bug detector with retrieved common-sense context injection."""

from bug_detect import BugDetectInterface


CONTEXT_MARKER = "\nOUTPUT FORMAT (STRICT JSON)"
CONTEXT_TEMPLATE = """
  ------------------------------------------------------------
  ### [Retrieved Common-Sense Context]

  The operation steps shown in the screenshot strip are roughly grouped into
  the following test situations. Each situation covers a specific step range
  and lists several retrieved common-sense rule families that may describe the
  expected behavior for those steps.

  In each retrieved family, "Family base" states the shared common-sense
  expectation. When "Matched variant" is present, it states a more specific
  form of that expectation whose situation was the closest match to the test
  situation. Interpret the matched variant together with its family base.

  For each test situation, compare the corresponding textual steps and
  screenshots with the listed expectations. First determine whether a rule's
  situation is applicable to the observed interaction. If it is applicable,
  check whether the actual GUI behavior violates its expected behavior. An
  applicable rule violation is evidence of a bug. Retrieved rules that are not
  applicable to the observed interaction must be ignored.

{commonsense_context}
  ------------------------------------------------------------
"""


class BugDetectCommonsenseInterface(BugDetectInterface):
    """Add auxiliary common-sense text without changing baseline behavior."""

    def build_prompt(
        self,
        funcname,
        funcdescription,
        funcgoal,
        pagepath,
        commonsense_context=None,
    ):
        baseline_prompt = super().build_prompt(
            funcname,
            funcdescription,
            funcgoal,
            pagepath,
        )
        if CONTEXT_MARKER not in baseline_prompt:
            raise ValueError("Could not locate OUTPUT FORMAT in baseline prompt.")
        context = commonsense_context or "[None provided]"
        injected = CONTEXT_TEMPLATE.format(commonsense_context=context)
        return baseline_prompt.replace(CONTEXT_MARKER, injected + CONTEXT_MARKER, 1)

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
            [image],
            prompt,
            return_json=False,
        )
