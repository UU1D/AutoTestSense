"""KuiTest response verification with page context and retrieved commonsense."""

from string import Template

from response_verification import ResponseVerification


prompt_with_commonsense = """
Here are the interacted component's function description and the screenshot of the page after interaction:
$component_function

Action performed: $action_description

The textual descriptions of the pages before and after the interaction are provided as additional context:
Pre-interaction Page $before_page_id: $before_page_description
Post-interaction Page $observed_page_id: $after_page_description

The following auxiliary context was produced by mapping page-indexed test situations to the current KuiTest transition:
$commonsense_context

The retrieved common-sense rules are candidate references, not mandatory requirements. First determine whether each rule is applicable to the current action, component, UI state, and current position in the situation. Ignore rules that are not applicable. Do not report a bug merely because a rule is present. A violation requires concrete contradictory evidence in the current screenshot or in the observed history above. Do not assume any future page or action that is not included in the observed history.

Determine whether the UI response after interaction aligns with the component function and every applicable common-sense expectation.
Return ONLY a JSON object in the following format:
{
    "judgement": true/false,
    "reason": "Provide concise evidence-grounded reasoning",
    "confidence": 0.0,
    "applicable_rule_family_ids": ["family id"]
}
"""


class ResponseVerificationCommonsense(ResponseVerification):
    def __init__(self):
        super().__init__()
        self.prompt_with_commonsense = Template(prompt_with_commonsense)

    def build_prompt(
        self,
        component_function,
        action_description,
        before_page_id,
        observed_page_id,
        before_page_description,
        after_page_description,
        commonsense_context,
    ):
        return self.prompt_with_commonsense.substitute(
            component_function=component_function,
            action_description=action_description,
            before_page_id=before_page_id,
            observed_page_id=observed_page_id,
            before_page_description=before_page_description,
            after_page_description=after_page_description,
            commonsense_context=commonsense_context,
        )

    def response_verification(
        self,
        image,
        component_function,
        action_description,
        before_page_id,
        observed_page_id,
        before_page_description,
        after_page_description,
        commonsense_context,
    ):
        prompt = self.build_prompt(
            component_function,
            action_description,
            before_page_id,
            observed_page_id,
            before_page_description,
            after_page_description,
            commonsense_context,
        )
        return self.llm_interface.get_response_from_lm(
            [image], prompt, return_json=False
        )
