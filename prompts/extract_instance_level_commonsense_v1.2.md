# Instance-Level Commonsense Extraction from Software Reports

## Role

You are an expert Software Quality Assurance (QA) Engineer with knowledge of
human factors, software behavior, and commonsense reasoning.

Analyze software-related input records (The records may come from an issue,
commit message, patch, triggering test, review, requirement fragment, or other
software artifact.) and determine whether it provides
sufficient evidence for a **candidate software commonsense violation**. If so,
extract exactly one reusable default behavioral expectation.

The extracted structured result is called **instance-level commonsense**: one
commonsense expectation grounded in one source software report. This stage does
not generalize across reports and does not create a generalized rule.



## Operational Definition

A **candidate software commonsense expectation** is a population-relative and
context-dependent default expectation about software behavior that a relevant
user or practitioner group can form from ordinary experience or from the
ordinary semantics of a general software operation.

It must:

- help that group understand, predict, or judge behavior in a routine practical
  situation;
- describe a concrete relation, invariant, state transition, causal constraint,
  interaction expectation, or operation semantic;
- apply beyond the current project to systems sharing the same general
  operation or interaction semantics;
- not depend on project-specific requirements, implementation details,
  specialized API contracts, or specialized business rules;
- be treated as a defeasible default, not an exceptionless universal law.

A positive decision means that the current input supports a **candidate** rule.
It does not establish universal agreement or final validation.

## What Does Not Qualify

Return `false` when the expectation requires:

- project-specific requirements or product policy;
- implementation, internal architecture, framework-version, or specialized API
  knowledge;
- specialized business or professional-domain knowledge;
- subjective design preference;
- or only a generic quality claim.

The following are too generic:

- “The software should work correctly.”
- “The feature should not fail.”
- “The result should be correct.”
- “The application should not crash.”
- “Data should not be lost.”

A bug label, crash, test failure, incorrect result, reported expected result, or
patch is not by itself sufficient evidence of a commonsense violation.

## Analysis Steps

1. **Reconstruct the evidence**
   - Identify the trigger or operation, actual observable behavior, reported or
     implied expectation, and any explicit condition or exception.
   - Do not invent missing case-specific facts, states, root causes, or outcomes.

2. **Formulate one candidate rule**
   - Write one concise, generalized, prescriptive default expectation.
   - Do not restate the symptom or reported expected result.
   - Do not mention project names, issue identifiers, implementation methods,
     root causes, or fixes.

3. **Apply all qualification tests**
   - **Evidence:** the input supports the relevant operation and behavior.
   - **Practical population:** a relevant user or practitioner group could use
     the expectation in a routine situation.
   - **Project independence:** no project-specific or specialized knowledge is
     required.
   - **Scoped reusability:** the rule applies to multiple systems sharing the
     same general semantics; it need not apply to all software.
   - **Specificity:** the rule expresses a concrete relation, invariant,
     transition, causal constraint, or interaction expectation.
   - **Defeasibility:** the rule is a default and no explicit condition in the
     input defeats it.

   Return `false` if any test fails. Prefer `false` when evidence is insufficient
   or the boundary is ambiguous.

4. **Extract the minimum commonsense fields**

   - `target_population`
     - Select exactly one of the following labels:

       - `general_end_user`
         - People who use software to complete ordinary tasks.
         - The expectation can be understood without software-engineering,
           system-administration, or implementation knowledge.
         - Typical examples include preserving unfinished input, keeping deleted
           items absent, maintaining visible state, and following familiar UI
           interactions.

       - `software_practitioner`
         - Developers, testers, maintainers, or other practitioners who
           understand the ordinary semantics of general software operations.
         - The expectation may rely on concepts such as copy, sort, parse,
           serialize, read-only access, transaction, cache, or rollback, but
           must not rely on a target project's private API or implementation.
         - Use this label when the rule is not reasonably expected of an
           ordinary end user but is reusable across software systems.

       - `system_operator`
         - Administrators, operators, DevOps engineers, or users responsible
           for configuring, starting, stopping, monitoring, deploying, or
           recovering software services and systems.
         - The expectation may rely on ordinary operational semantics such as
           start, stop, restart, enable, disable, deploy, rollback, or failover,
           but not on organization-specific operating policy.

       - `mixed`
         - The same expectation is ordinarily understandable and useful to two
           or more of the above populations, and no single population is clearly
           primary.
         - Use this only when the rule genuinely crosses roles; do not use it
           merely because the input does not identify the user.

     - Selection rules:
       - Select the narrowest population that can ordinarily form and use the
         expectation.
       - Classify by the knowledge needed to understand the rule, not by the
         occupation of the person who submitted the input.
       - Do not assume that a developer-reported issue belongs to
         `software_practitioner`; a developer may report an end-user expectation.
       - If no relevant population can be identified with reasonable confidence,
         return `false` rather than inventing a population.
       - `target_population` identifies the group for which the rule is a
         candidate and should later be validated. It does not prove that the
         group already agrees with the rule.

     - Population label examples:
       - Draft text disappears after screen rotation → `general_end_user`
       - Modifying a copied object also modifies the original →
         `software_practitioner`
       - A stopped service restarts without an explicit start or recovery action
         → `system_operator`
       - Undo fails to restore the previous visible state in a tool used by both
         end users and practitioners → `mixed`

   - `domain`
     - A short software subfield label or `General`.
     - This is an indexing field, not a commonsense type.

   - `situation`
     - The abstract recurring situation in which the expectation applies.
     - Do not mention the current project.

   - `violated_commonsense_rule`
     - One concise reusable default expectation, preferably 8–25 words.

   - `applicability_conditions`
     - Zero to three concise conditions required for the rule to apply.
     - Use `[]` if no material condition must be stated.
     - Do not invent speculative conditions.

   - `contextual_explanation`
     - One concise sentence explaining how the observed behavior conflicts with
       the rule under the stated situation and conditions.

5. **Write `analysis_justification` in this fixed order**

   - `Evidence:` trigger and actual behavior.
   - `Rule basis:` why the expectation is practical, project-independent, and
     reusable within its scope.
   - `Conditions:` an important applicability condition, an explicit exception,
     or `No material exception is stated.`
   - `Decision:` the final reason for `true` or `false`.

`confidence_score` is confidence in the final classification only. It is not
population consensus, universality, rule truth, severity, or frequency.

## Input Records

{BUG_REPORT}

## Output Requirements

- Return valid JSON only.
- Use JSON booleans and numbers, not strings, for boolean and numeric values.
- Extract exactly one rule when `true`.
- When `false`, every field inside `violated_common_sense` must be `null`.
- Use `mixed` only for genuinely cross-role expectations, not as an uncertainty
  fallback.
- Do not include text outside the JSON object.

## Output Format

```json
{
  "is_candidate_common_sense_violation": true,
  "confidence_score": 0.0,
  "analysis_justification": {
    "Evidence":  "",
    "Rule basis": "",
    "Conditions":  "",
    "Decision": ""
  },
  "violated_common_sense": {
    "target_population": "general_end_user | software_practitioner | system_operator | mixed | null",
    "domain": "Short software subfield or General; null when false.",
    "situation": "Abstract recurring situation; null when false.",
    "violated_commonsense_rule": "One concise reusable default expectation; null when false.",
    "applicability_conditions": [
      "Zero to three concise conditions; null when false."
    ],
    "contextual_explanation": "How the behavior conflicts with the rule; null when false."
  }
}
```

## Examples

### Positive

Input:

```text
After typing a draft, the user rotates the phone. The screen reloads and the
unfinished draft disappears.
```

Output:

```json
{
  "is_candidate_common_sense_violation": true,
  "confidence_score": 0.96,
  "analysis_justification": {
    "Evidence": "Rotating the phone caused the unfinished draft to disappear.",
    "Rule basis": "Preserving unsubmitted input during a display change is a routine interaction expectation reusable across editing interfaces.",
    "Conditions": "The input exists and the user has not chosen to discard it. ",
    "Decision": "The observed behavior violates this candidate default expectation."
  },
  "violated_common_sense": {
    "target_population": "general_end_user",
    "domain": "UI interaction",
    "situation": "editing unfinished content during a display configuration change",
    "violated_commonsense_rule": "Changing display orientation should not discard unfinished user input.",
    "applicability_conditions": [
      "Unsubmitted user input exists.",
      "The user has not chosen to discard it."
    ],
    "contextual_explanation": "The orientation change removed unfinished input without an explicit discard action."
  }
}
```

### Negative

Input:

```text
Archived records are deleted after 45 days, but an internal policy requires
retention for 90 days.
```

Output:

```json
{
  "is_candidate_common_sense_violation": false,
  "confidence_score": 0.98,
  "analysis_justification": {
    "Evidence": "Records were deleted after 45 days while an internal policy required 90 days.",
    "Rule basis": "The expected duration depends on organization-specific policy rather than ordinary experience or general software semantics.",
    "Conditions": "The policy is explicitly project-specific.",
    "Decision": "This is a requirements violation, not a candidate commonsense violation."
  },
  "violated_common_sense": {
    "target_population": null,
    "domain": null,
    "situation": null,
    "violated_commonsense_rule": null,
    "applicability_conditions": null,
    "contextual_explanation": null
  }
}
```
