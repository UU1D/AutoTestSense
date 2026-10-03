# Judge Consolidation Membership for Unassigned Instance-Level Commonsense

## Terminology and Schema Compatibility

The new item is **unassigned instance-level commonsense**. Each candidate is a
**commonsense generalization record**, and its `base_unit` is the current
**generalized commonsense rule**. Legacy JSON keys and decision enum values are
retained for compatibility and must be emitted exactly as specified.

## Your Task

You will receive:

1. one unassigned instance-level commonsense item;
2. several candidate commonsense generalization records that may be related to it.

Decide whether the new item may be consolidated into exactly one candidate record. If it
does, classify it as:

- `EQUIVALENT_MEMBER`; or
- `VARIANT_MEMBER`.

If none of the candidates is suitable, return `NO_MATCH`.

This is a preliminary consolidation decision. A later consolidation step will verify
any `VARIANT_MEMBER`, construct its delta, and decide whether the family base
must be revised. Therefore, accept a strong commonsense generalization record relationship even when
the current base is slightly too specific, but do not force together units that
only share a topic.

Only make this classification. Do not:

- rewrite a family base;
- generate or edit a delta;
- choose an existing variant group;
- create a new family;
- describe what the later generalization stage should do.

The candidates were found by embedding retrieval and reranking. Similarity is
only a reason to compare them, not evidence that they belong together.

## Input Meaning

The new unit has two fields:

- `Situation`: the recurring situation in which the expectation applies,
  especially the relevant operations, states, triggers, and their relationships.
- `Common-sense rule`: the expected behavior or result, or the failure that the
  software is expected to prevent.

Each candidate family has a `Base unit` with the same fields. The base states
the meaning currently shared by its members.

A candidate may include a `Matched variant`: the existing variant whose
embedding was closest to the new unit. It contains its original situation and
rule together with its delta from the base. Use it as supporting evidence, but
judge whether the new unit fits the family as a whole.

Candidate IDs such as `C0` and `C1` are temporary reference labels. Copy the
selected ID exactly. Do not infer meaning from an ID or invent another ID.

## Commonsense Generalization Record Principle

Units may belong to one family when they preserve the same specific and
testable software property, even if they express that property through
different situations, restrictions, mechanisms, or failure outcomes.

A useful family base must state substantive shared meaning. It must not be a
vague statement such as "software should work correctly", "the interface should
be usable", or "unexpected behavior should not occur".

Do not group units merely because they mention the same component, feature,
workflow, object, platform, or keyword.

## Equivalent Member

Use `EQUIVALENT_MEMBER` only when the current family base can stand in place of
the new unit without losing important meaning or adding unsupported meaning.

The base must already preserve:

- the material operation or state relationship;
- the applicability scope;
- the expected behavior or result;
- the prohibited failure or outcome.

Differences in wording, terminology, project, platform, and incidental examples
may be ignored when they do not change these meanings. If a meaningful delta or
base revision would be needed, do not use `EQUIVALENT_MEMBER`; evaluate
`VARIANT_MEMBER` instead.

## Variant Member

Use `VARIANT_MEMBER` in either of the following two cases.

### Case A: Direct Variant of the Current Base

The current base remains valid in the new unit's situation, and the new unit can
be represented as:

> current base + limited situation or rule differences

The differences may include:

- a narrower situation or applicability condition;
- a specific trigger, operation, interaction sequence, or state relationship;
- a concrete mechanism for preserving the same property;
- an additional required behavior while the base behavior remains required;
- a particular failure outcome that violates the same property.

The differences must not deny, contradict, replace, or correct the base.

### Case B: Compatible Family Expansion

The current base is too specific to directly represent the new unit, but the new
unit and the existing family are sibling or broader/narrower expressions of the
same specific and testable property.

Use `VARIANT_MEMBER` under this case only when all of the following are true:

1. A concise shared base can be stated from meanings explicitly supported by
   both the new unit and the candidate family.
2. The shared base identifies a concrete behavior or correctness property, not
   merely a common topic or goal.
3. The current family and the new unit could each be reconstructed as that
   shared base plus limited, clear differences.
4. The differences concern specific situations, operations, mechanisms,
   restrictions, or failure manifestations; they do not introduce unrelated
   primary obligations.
5. Forming the shared base would require limited generalization, not replacement
   of nearly the entire family meaning.
6. Neither side conflicts with the other under overlapping conditions.

This classification does not authorize a base rewrite. It only sends the case
to the later generalization stage for full validation.

For example:

```text
Candidate base:
Shell scripts should use line endings compatible with their execution
environment.

New unit:
Source files should use line endings that do not break a build in the target
environment.
```

These may be `VARIANT_MEMBER` because both preserve the concrete property that
files consumed in a target environment must use line endings compatible with
the operation performed there. Script execution and source compilation are
limited differences that the later stage can retain.

Another example:

```text
Candidate base:
Rapid view switching should not make the interface freeze or become
unresponsive.

New unit:
Rapid view switching should not permanently lock application state or cause
data loss.
```

These may be `VARIANT_MEMBER` because both protect application stability during
the same operation, while identifying different concrete failure outcomes.

## When to Return NO_MATCH

Return `NO_MATCH` for a candidate when:

- the connection is only a shared topic, object, component, or workflow;
- the units require materially different primary behaviors;
- they protect different correctness properties;
- their situations are unrelated except for broad domain terminology;
- one permits an outcome that the other prohibits under the same conditions;
- a shared base would be vague or would omit nearly all specific meaning;
- reconstructing either side would require a delta that repeats most of its
  situation and rule;
- combining them would introduce unsupported assumptions;
- the relationship is uncertain.

For example, "a virtual keyboard should use the correct language layout" and
"a virtual keyboard should not cover selectable suggestions" are `NO_MATCH`.
They concern the same component but protect different properties.

Likewise, preventing duplicate page instances and ensuring that navigation can
be repeated after returning to a prior screen are `NO_MATCH`. Both concern
navigation, but they require different behavior and prevent different failures.

## How to Use a Matched Variant

A matched variant explains why the family was retrieved. Its similarity to the
new unit is not sufficient by itself.

- A new unit may be equivalent to the base even when retrieval matched a
  variant.
- A new unit may be a different valid variant without being equivalent to the
  matched variant.
- Similarity to a variant must not override a conflict with the family.
- Do not assume anything about variants that are not shown.

## Decision Steps

Evaluate every candidate separately:

1. Identify the new unit's situation, expected behavior, protected property,
   and prohibited failure.
2. Compare them with the current family base and any shown matched variant.
3. If the current base alone preserves the complete meaning, classify it as
   `EQUIVALENT_MEMBER`.
4. Otherwise, test whether it is a direct variant of the current base under
   Case A.
5. If not, test whether a limited and substantive shared base could represent
   both sides under Case B.
6. Choose the single strongest valid candidate. If no candidate passes these
   tests, return `NO_MATCH`.

Be conservative about the existence of a shared property, but do not reject a
candidate solely because the current base is narrower than the new unit or
because the two units describe different concrete manifestations of the same
property.

## Output

Return exactly one JSON object:

```json
{
  "decision": "ADD_TO_FAMILY | NO_MATCH",
  "selected_candidate_id": "C0 or null",
  "membership_type": "EQUIVALENT_MEMBER | VARIANT_MEMBER | null",
  "rationale": "..."
}
```

## Output Requirements

- For `ADD_TO_FAMILY`, select exactly one supplied candidate and use either
  `EQUIVALENT_MEMBER` or `VARIANT_MEMBER`.
- For `NO_MATCH`, both `selected_candidate_id` and `membership_type` must be
  `null`.
- In `rationale`, identify the shared or incompatible situation and testable
  property. For Case B, briefly identify the substantive common property and
  the main differences that must be retained.
- Do not mention retrieval ranks or similarity scores.
- Do not output the new unit's ID, a base rewrite, a delta, or a proposed later
  operation.
- Return JSON only, without Markdown or additional text.
