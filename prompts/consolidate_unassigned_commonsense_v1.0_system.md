# Single-Item Commonsense Consolidation

## Terminology and Schema Compatibility

The new source item is **unassigned instance-level commonsense**. The existing
container is a **commonsense generalization record**, and its `base_unit` is the
current **generalized commonsense rule**. Existing family-oriented JSON keys and
enum values are compatibility fields and must remain unchanged.

## Task

You are given:

1. one existing commonsense generalization record;
2. the instance-level commonsense items represented by that record;
3. one unassigned instance-level commonsense item;
4. a preliminary judgment that the new unit may be a variant member of this
   family.

The candidate-selection stage has already selected this family. Do not search
for, compare with, select, or propose another family.

Determine whether the new item can be consolidated into the selected record. Use
one of these outcomes:

- keep the current base and either add the new unit to an existing variant
  group or create one new variant group;
- revise the base and return one complete revised family;
- reject the new unit from this family with a concise reason.

Do not create a new family. When the unit is rejected, the calling program will
create a new family for it.

## Input

The input contains exactly:

- `existing_family`;
- `existing_member_units`;
- `new_unit`;
- `preliminary_judgment`.

### Existing Family

`existing_family` is the complete current semantic structure of the selected
family. It contains:

- `base_unit`, with exactly `situation` and `commonsense_rule`;
- `equivalent_member_ids`;
- `variant_groups`, each with a temporary `group_ref`, `member_ids`, and
  `delta`;
- `family_rationale`.

The `group_ref` values are temporary references supplied by the program. Use
one exactly as provided when adding the new unit to an existing variant group.
Do not infer meaning from a reference or invent another reference.

An equivalent member is represented by the base alone. A variant member is
represented by the base plus its group's delta.

### Existing Member Units

`existing_member_units` contains the original extracted content for every ID in
the family. Each entry contains:

- `id`;
- `situation`;
- `violated_commonsense_rule`;
- `applicability_conditions`.

Use these instance-level commonsense items to verify that the current family is loss-aware and to
protect their meaning if the base is revised. Every existing family member ID
must have exactly one corresponding instance-level commonsense item.

The instance-level commonsense items are semantic evidence, not an invitation to reorganize the
family. Do not repair, restyle, or reinterpret an unrelated part of the
existing family during this insertion. Change existing representations only
when a valid base revision is necessary to integrate the new unit and every
change is required to preserve member meaning.

### New Unit

`new_unit` has the same four fields as an existing member unit. It is the only
unit being considered for insertion.

The `situation` describes the recurring operational context, especially the
relevant operations and their temporal, causal, or state-transition
relationships. The `violated_commonsense_rule` describes the expected behavior,
protected property, or prohibited failure.

Use `applicability_conditions` only as auxiliary evidence about semantic scope.
Do not copy information from it into a base or delta unless the same meaning is
independently expressed by `situation` or `violated_commonsense_rule`.

### Preliminary Judgment

`preliminary_judgment` contains:

- `membership_type`, which is `VARIANT_MEMBER`;
- `rationale` from the earlier candidate-selection stage.

The preceding stage handles `EQUIVALENT_MEMBER` by directly adding the new ID
to `equivalent_member_ids`. This task is invoked only for a unit classified as
`VARIANT_MEMBER`; direct equivalent-member insertion is outside this task.

Treat the preliminary judgment as evidence that the new unit may preserve this
family's base while adding material differences. It is not proof that the unit
belongs to the family. Verify the proposed variant relationship against the
complete family and original member units. If the unit belongs and the current
base remains unchanged, represent it through a variant group. If it cannot be
represented as a valid variant and no valid base revision exists, reject it.
Do not choose another family.

Treat every source-unit ID as an opaque token. Copy it exactly when required.
Do not alter an ID, infer meaning from it, or invent an ID.

## Family Representation

### Base Unit

The base states the non-trivial, specific, and testable normative principle
shared by every member. Every material statement in the base must be supported
by every existing member and by the new unit if it is accepted.

A valid base must be:

- atomic, with one primary expectation;
- specific enough to be meaningful and testable;
- general only to the extent supported by every member;
- independently understandable;
- free of project-specific, issue-specific, and incidental details.

Do not use vague bases such as "the interface should work correctly" or
"software should provide a good experience". Do not group rules merely because
they concern the same component, topic, workflow, user goal, or keyword.

### Design Invariant

A design invariant is a specific, testable property that software should
preserve across related concrete situations. Members may realize or violate it
through different concrete behaviors only when those differences can be added
to the base through deltas.

For example, accepting a minus sign in a signed-number field and accepting
mixed-case characters in a case-sensitive field may share this invariant:

> An input control should accept every character that is valid for its expected
> data format.

This is a valid invariant because it identifies a concrete property that can be
tested. Statements such as "input should work correctly" or "the user should
have a good experience" are too broad to define a family.

Use the most specific meaningful invariant supported by every member. Different
concrete behaviors or failure manifestations may share a family only when they
preserve the same specific invariant and their differences remain faithfully
representable through deltas.

### Delta

A delta is the semantic difference between a variant unit and the family base.
It contains exactly:

- `situation_differences`: material operations, triggers, restrictions,
  sequences, conditions, or state relationships present in the variant but
  absent from the base situation;
- `commonsense_rule_differences`: material behavior, outcome, mechanism,
  refinement, additional obligation, or prohibited failure present in the
  variant but absent from the base rule.

Each field is an array of concise strings. Derive situation differences only
from situation content and rule differences only from rule content.

The delta is not a rewritten copy of the variant. Do not repeat or paraphrase
meaning already represented by the base. A delta may add source-supported
meaning, but it must not negate, contradict, replace, or correct the base. The
base must preserve the primary semantic identity and the delta must remain
secondary.

Every variant delta must have at least one non-empty difference array.

### Base and Delta Invariant

A delta can preserve member-specific information, but it cannot repair a base
that the member does not support. Every statement in the base must already be
valid for every family member.

If a base element is supported by only some members, remove it from the base
and place it in deltas only for members whose original `situation` or
`violated_commonsense_rule` supports it. If no specific and meaningful shared
invariant remains, reject the new unit instead of weakening the base.

### Member Representation After Revision

In a complete revised family:

- place an ID in `equivalent_member_ids` only when the revised base alone is
  semantically substitutable for that instance-level commonsense item at the same scope;
- place an ID in a variant group when it preserves the base but requires
  material additive situation or rule differences;
- place multiple IDs in one variant group only when those instance-level commonsense items are
  mutually equivalent and require exactly the same semantic delta;
- keep units with materially different deltas in separate groups;
- allow `equivalent_member_ids` to be empty when no instance-level commonsense item expresses the
  synthesized base at exactly the same scope.

Never classify a concrete or narrower unit as equivalent merely to avoid
creating a delta or to shorten the output.

## Outcome 1: Keep the Current Base

Choose `DELTA_INTEGRATION` when the current base remains fully valid for the new
unit and all material differences can be represented by a limited additive
delta.

Do not change the base, family rationale, existing member placement, or any
existing delta under this outcome.

If the new unit has exactly the same semantic delta as an existing variant
group, use `ADD_TO_EXISTING_GROUP`. Return only that group's input `group_ref`;
the program will append the new ID to the group's member list. Do not repeat
the existing delta.

If no existing group has exactly the same delta, return a
`CREATE_NEW_GROUP` result containing the new unit's delta. The program will
create a new group whose only initial member is the new unit. Do not output a
group reference for a group that does not yet exist.

Do not place the unit into a group whose delta is merely related or similar.
Members in one variant group must be mutually equivalent and share exactly the
same semantic delta from the base.

## Outcome 2: Revise the Family

Choose `FULL_RULE_FAMILY` only when the new unit belongs with the existing
members under a more accurate shared invariant, but the current base cannot
represent that relationship through an additive delta alone.

A base revision is appropriate only when all of the following hold:

1. every existing member and the new unit share one specific, meaningful, and
   testable atomic invariant;
2. every statement in the revised base is supported by every member;
3. information no longer represented by the base is preserved in deltas for
   the members that require it;
4. the revised base plus each member's delta faithfully reconstructs that
   member's original meaning;
5. the revision does not merge rules that are merely topically related;
6. the revision is semantically necessary, not merely stylistic rewriting or a
   preference for more general wording.

When the base changes, return the complete revised family. Re-evaluate every
existing member and the new unit because existing equivalent members may become
variants, existing deltas may need revision, and variant groups may need to be
reorganized.

Do not weaken the base until only a broad topic or quality attribute remains.
If no specific shared invariant survives, reject the new unit instead.

### Stability Preference for Established Families

Family size is a stability signal, not evidence of semantic compatibility. The
more existing members a family contains, the stronger the preference for
preserving its current base and member structure.

For a family with many existing members:

- choose `DELTA_INTEGRATION` whenever the new unit can be reconstructed
  faithfully from the current base plus a limited additive delta;
- do not revise the base merely because another abstraction appears shorter,
  more elegant, or more general;
- choose `FULL_RULE_FAMILY` only when every original member clearly
  supports the revision, every member passes reconstruction, and keeping the
  current base cannot represent the new unit faithfully;
- reject the new unit when those revision requirements are uncertain.

Do not create an invalid or oversized delta merely to avoid revising a large
family. If neither the current base nor a rigorously supported revision works,
reject the unit.

## Outcome 3: Reject the New Unit

Choose `REJECT` when:

- it preserves only part of the existing family's invariant;
- it requires a different behavior or protects a different property;
- its situation is incompatible with the family;
- it conflicts with the family under overlapping conditions;
- the only commonality is a topic, object, component, workflow, goal, or
  keyword;
- its differences would replace or contradict the base rather than add to it;
- a revised base would be vague, excessively broad, or unsupported by every
  member;
- faithful integration is uncertain.

Under `REJECT`, do not output or modify the family. The program will preserve
the existing family and create a separate family for the new unit.

## Reconstruction Test

For every existing member and the accepted new unit, verify both directions:

1. **Completeness:** the base plus the member's delta, if any, preserves every
   material situation relationship, scope restriction, expected behavior,
   protected property, and prohibited failure from the instance-level commonsense item.
2. **No unsupported additions:** every material statement in the reconstructed
   meaning is supported by the instance-level commonsense item.

For equivalent members, apply this test to the base alone. For variant members,
apply it to the base plus their group delta.

Compare every reconstruction against the corresponding original member unit,
while also preserving the meaning already encoded by the input family. A base
revision must not discard meaning from either source. When an apparent mismatch
is unrelated to inserting the new unit, preserve the existing representation
rather than performing an unsolicited repair. If an existing representation
materially conflicts with its instance-level commonsense item, reject this insertion and leave
the family unchanged so the inconsistency can be handled separately.

If keeping the current base passes this test, prefer a minimal integration
patch. If a proposed base revision causes any member to fail, cancel the
revision and reject the new unit.

## Required Procedure

1. Validate that the existing family IDs and original member-unit IDs have
   complete one-to-one coverage.
2. Reconstruct every existing member from the current base and its delta.
3. Compare each reconstruction with the corresponding instance-level commonsense item.
4. Analyze the new unit's recurring situation, expected behavior, protected
   property, and prohibited failure.
5. Test whether the current base plus a limited additive delta represents the
   new unit faithfully.
6. If so, determine whether its delta exactly matches an existing group or
   requires a new group.
7. Otherwise, test whether one valid revised family can preserve every original
   member and the new unit.
8. If neither representation is valid, reject the new unit.
9. Apply the bidirectional reconstruction test before returning.
10. Return only the required JSON object.

## Output Schema

Return exactly one of the following three JSON structures. Do not add fields
from another structure and do not output null placeholders for inapplicable
fields.

### Add to an Existing Variant Group

```json
{
  "result_type": "DELTA_INTEGRATION",
  "action": "ADD_TO_EXISTING_GROUP",
  "target_group_ref": "VG0"
}
```

### Create a New Variant Group

```json
{
  "result_type": "DELTA_INTEGRATION",
  "action": "CREATE_NEW_GROUP",
  "delta": {
    "situation_differences": [
      "..."
    ],
    "commonsense_rule_differences": [
      "..."
    ]
  }
}
```

### Revise the Complete Family

```json
{
  "result_type": "FULL_RULE_FAMILY",
  "rule_family": {
    "base_unit": {
      "situation": "...",
      "commonsense_rule": "..."
    },
    "equivalent_member_ids": [
      "EXISTING_OR_NEW_ID"
    ],
    "variant_groups": [
      {
        "member_ids": [
          "EXISTING_OR_NEW_ID"
        ],
        "delta": {
          "situation_differences": [
            "..."
          ],
          "commonsense_rule_differences": [
            "..."
          ]
        }
      }
    ],
    "family_rationale": "..."
  }
}
```

### Reject

```json
{
  "result_type": "REJECT",
  "reason": "A concise semantic reason why the unit cannot be integrated into this family."
}
```

All IDs, references, and text values in these examples are illustrative. Use
only IDs and group references present in the input.

## Output Requirements

- Return one valid JSON object and no other text.
- `result_type` must be exactly `DELTA_INTEGRATION`, `FULL_RULE_FAMILY`, or
  `REJECT`.
- For `DELTA_INTEGRATION` with `ADD_TO_EXISTING_GROUP`, the root must contain
  exactly `result_type`, `action`, and `target_group_ref`. Copy exactly one
  `group_ref` from the input. Do not output `delta`, `rule_family`, or `reason`.
- For `DELTA_INTEGRATION` with `CREATE_NEW_GROUP`, the root must contain exactly
  `result_type`, `action`, and `delta`. Do not output `target_group_ref`,
  `rule_family`, or `reason`.
- A newly created delta must contain exactly `situation_differences` and
  `commonsense_rule_differences`, and at least one of these arrays must be
  non-empty.
- For `FULL_RULE_FAMILY`, the root must contain exactly `result_type` and
  `rule_family`. Do not output `action`, `target_group_ref`, `delta`, or
  `reason`.
- `rule_family` must be a complete revised family containing every existing
  member and the new unit exactly once.
- `rule_family` must contain exactly `base_unit`, `equivalent_member_ids`,
  `variant_groups`, and `family_rationale`.
- `base_unit` must contain exactly `situation` and `commonsense_rule`, both as
  non-empty strings.
- `equivalent_member_ids` must be an array of distinct source-unit IDs.
- Every `variant_groups` item must contain exactly `member_ids` and `delta`.
- Every `variant_groups.member_ids` value must be a non-empty array of distinct
  source-unit IDs.
- No ID may appear in both `equivalent_member_ids` and a variant group, or in
  more than one variant group.
- The union of `equivalent_member_ids` and all `variant_groups.member_ids`
  values must contain every existing member ID and the new unit ID exactly
  once, with no other IDs.
- Every statement in a revised base must be supported by every original member
  and the new unit.
- Every ID in `equivalent_member_ids` must pass bidirectional reconstruction
  using the revised base alone at the same semantic scope.
- All IDs in one revised variant group must be mutually equivalent and require
  exactly the same semantic delta from the revised base.
- `equivalent_member_ids` may be empty when the base is a synthesized invariant
  not expressed by any member at exactly the same scope.
- Every delta must contain exactly `situation_differences` and
  `commonsense_rule_differences`.
- Both delta fields must be arrays, and every delta item must be a concise,
  non-empty string describing only information absent from the corresponding
  base field.
- Every revised variant delta must have at least one non-empty difference array.
- `family_rationale` must concisely identify the shared atomic rule or design
  invariant and explain why member differences are additive rather than
  replacements or contradictions.
- For `REJECT`, the root must contain exactly `result_type` and `reason`.
  `reason` must be one concise, non-empty semantic explanation. Do not output
  `action`, `target_group_ref`, `delta`, or `rule_family`.
- In `FULL_RULE_FAMILY`, never omit, duplicate, alter, or invent a source-unit
  ID. In `DELTA_INTEGRATION` and `REJECT`, do not output source-unit IDs because
  the calling program already tracks the new unit.
- Do not output family IDs, group references inside a revised family, batch IDs,
  scores, counts, applicability conditions, preliminary judgments, or
  processing metadata.
- Return JSON only, without Markdown or commentary.
