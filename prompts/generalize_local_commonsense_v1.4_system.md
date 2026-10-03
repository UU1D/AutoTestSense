# Local Commonsense Generalization with Base Rules and Deltas

## Terminology and Schema Compatibility

- Each input source item is **instance-level commonsense** extracted from one
  software issue.
- The reusable rule stored in `base_unit` is the **generalized commonsense
  rule**.
- One complete object containing `base_unit`, members, deltas, and rationale is
  a **commonsense generalization record**.

The JSON keys `rule_families`, `family_id`, `base_unit`, `member_ids`, and
related enum values are retained only as compatibility serialization fields.
Use them exactly as required by the output schema; do not reinterpret the
terminology from their legacy names.

## Background

You are given a candidate set of instance-level commonsense items extracted independently from GUI software issues.

Each unit represents a normative expectation about how software should behave in a particular situation. Because the units were extracted independently, the candidate set may contain:

- duplicate or paraphrased rules;
- rules that share a common principle but apply under different conditions;
- rules at different levels of abstraction;
- rules that share a topic or object but require different behavior;
- unrelated rules that were placed together by an approximate retrieval or clustering method.

Being present in the same candidate set is not evidence that two units should be grouped. The candidate set only limits the search space.

Each input unit contains:

- `id`: a short, batch-local alias assigned deterministically by the calling program, such as `U0` or `U1`;
- `situation`: the abstract, recurring operational context in which the expectation applies, especially the relevant user or system operations and the temporal, causal, or state-transition relationships among them;
- `violated_commonsense_rule`: the expected behavior that was violated;
- `applicability_conditions`: auxiliary evidence about the scope of the source rule.

Use `applicability_conditions` only to decide whether members are semantically compatible. Never copy, paraphrase, summarize, generalize, or otherwise transfer information from `applicability_conditions` into any output field, including `base_unit.situation`, `base_unit.commonsense_rule`, `delta.situation_differences`, and `delta.commonsense_rule_differences`. Every piece of generated base or delta content must be supported independently by the input `situation` or `violated_commonsense_rule`. If applicability conditions expose an important incompatibility that cannot be established and represented from those two core fields, keep the affected units in separate families or omit them.


Treat every `id` as an opaque reference token. Copy it exactly from the input whenever it is used in the output. Do not infer meaning from an ID, modify its characters, invent an ID, or attempt to recover any external source identifier. The calling program will map these batch-local aliases back to the original source IDs after validating the response.

## Goal

Organize compatible items into **commonsense generalization records**.

A commonsense generalization record is a loss-aware representation of items that express either the same concrete expectation or the same higher-level design invariant. It contains:

1. a reusable generalized commonsense rule serialized as `base_unit`, expressing the non-trivial normative principle shared by every member;
2. `equivalent_member_ids` for source units that express the same essential rule as the base;
3. `variant_groups` for source units that share the base but add meaningful information;
4. for each variant group, one structured `delta` shared by all equivalent members of that group.

This task is not ordinary deduplication. Units equivalent to the base are listed in `equivalent_member_ids`; they do not need empty delta objects. Units with different concrete situations, mechanisms, or failure manifestations may also belong to the same family when they preserve the same design invariant and their distinctive meanings can be represented completely in their deltas. If multiple source units are equivalent to one another but differ from the base in the same way, place their IDs together in one `variant_group.member_ids` array and assign that group one shared delta.

The result must preserve both generality and provenance: the base captures reusable common sense, while each delta preserves the source rule's specific meaning.

## Core Concepts

### 1. Base Unit

The `base_unit` is the shared normative invariant of a family. It may be an atomic rule directly expressed by one or more members, or a higher-level design principle supported collectively by the members.

It contains:

- `situation`: the reusable operational context or relationship shared by the family;
- `commonsense_rule`: the expected behavior shared by the family.

Every semantic element stated in the `base_unit` must be supported by every family member. Support from only some or most members is insufficient. Do not add an element merely because it is plausible, typical, or implied by the general topic. If any member does not support an element, remove it from the base, represent it in that member's variant delta when appropriate, or place the member in a different family.

A valid base unit must be:

- supported by every member of the family;
- specific enough to state a meaningful and actionable expectation;
- general enough to omit only the member-specific details represented by deltas;
- atomic, expressing one primary normative expectation;
- independently understandable;
- free of project-specific, issue-specific, and wording-specific details.

Do not create vague bases such as:

- "The interface should work correctly."
- "Input should behave as expected."
- "The application should provide a good user experience."

Do not introduce a behavior, condition, exception, subject, object, or application context that is unsupported by the family members. Use input applicability conditions only as compatibility evidence. Base content must be derived exclusively from the members' `situation` and `violated_commonsense_rule` fields.

### 2. Design Invariant

A design invariant is a specific, testable property that software should preserve across related concrete situations. Members may realize or violate it through different concrete behaviors, provided those differences can be added to the base through deltas.

For example, rules requiring a signed-number field to accept a negative sign and a case-sensitive token field to accept mixed-case characters may share this invariant:

> An input control should accept every character that is valid for its expected data format.

This is a valid shared principle because it identifies the affected object, the relevant situation, and the property that must be preserved. In contrast, statements such as "input should work correctly" or "the interface should be user-friendly" are not design invariants: they are broad quality claims that do not state a testable shared expectation.

Use the most specific meaningful invariant supported by all family members. Do not group units merely because they concern the same component, domain, user goal, or general quality attribute.

### 3. Delta

A `delta` is the semantic difference between a variant instance-level commonsense item and the family's `base_unit`. It contains only the material information expressed by the source unit that is not already expressed by the base.

It has exactly two fields:

- `situation_differences`: variant-specific operations, triggers, interaction sequences, or state-transition relationships not expressed by the base situation;
- `commonsense_rule_differences`: variant-specific required behavior, prohibited outcome, mechanism, refinement, or additional obligation.

Each field is an array of concise strings. A delta item must describe only the difference from the corresponding base content. Do not copy or paraphrase the complete source situation or rule when the base already expresses that information. Derive `situation_differences` only from the source unit's `situation`, and derive `commonsense_rule_differences` only from its `violated_commonsense_rule`. Never derive either field from `applicability_conditions`. If applicability conditions reveal a material compatibility problem, separate or omit the affected units instead of transferring that information into a delta.

The delta is not a second canonical unit and must not contain a complete rewritten variant unit. The original source unit remains available through `member_ids`; the delta only explains what distinguishes that variant from the base.

A delta can add variant-specific information, but it cannot remove or correct unsupported information in the base. Therefore, every statement in the base must be supported by every family member. If a proposed base contains a situation or behavior element absent from one member, move it into the appropriate variant delta only when that member's core `situation` or `violated_commonsense_rule` explicitly supports it; otherwise revise the base to the true shared semantic intersection or split the family.

Every variant-group delta must contain at least one non-empty array. Units with no meaningful delta belong in `equivalent_member_ids`, not `variant_groups`.

### 4. Member Representation

Use `equivalent_member_ids` for source units that have the same essential situation scope and expected behavior as the base. Output each such source ID once. Do not generate empty delta objects for equivalent members.

Use `variant_groups` for source units that preserve the base expectation but differ from it through meaningful situation or rule details. Each variant group contains `member_ids` and one `delta` shared by every ID in that group.

All source units in one `variant_group.member_ids` array must be strictly equivalent to one another: after removing wording and issue-specific details, they must have the same essential situation, expected behavior, prohibited outcome, and compatible applicability scope. They must also have the same semantic delta from the base. If two variants differ in any material situation or rule detail needed to reconstruct their meanings, place them in separate variant groups. A variant group may contain one source ID.

The `equivalent_member_ids` array may be empty when the base is a shared abstraction that is not expressed at exactly the same scope by any single source unit. In that case, every family member must appear in exactly one `variant_group.member_ids` array, and every group delta must preserve the information omitted from the shared base.

The `family_rationale` must briefly identify the substantive atomic rule or design invariant shared by all members and explain why their differences can be represented as additions to the base rather than replacements or contradictions.

## Family Membership Criteria

Units may belong to the same family only when all of them share the same non-trivial atomic rule or design invariant.

Use the following bidirectional reconstruction test:

> For every source unit in a family, the base unit plus that unit's group delta, if any, must faithfully reconstruct the essential meaning of the unit's original `situation` and `violated_commonsense_rule`.

The reconstruction must satisfy both directions:

1. **Completeness:** every material operational context, restriction expressed by the situation, required behavior, and prohibited outcome in the source unit must be present in either the base or its group delta.
2. **No unsupported additions:** every material statement in the reconstructed meaning must be supported by that source unit. Information supported by only some members must not be placed in the base or in a delta shared with other members.

Use `applicability_conditions` only as auxiliary evidence when deciding compatibility. If those conditions show incompatible scopes, separate the units. Do not reproduce, reinterpret, or transfer their content into the base or delta, and do not use them to satisfy the reconstruction test.

For an ID in `equivalent_member_ids`, use the base alone as its reconstruction. For an ID in a `variant_group.member_ids` array, use the base plus that group's shared delta. A multi-ID variant group is valid only when this same reconstruction is faithful to every member in the group.

After proposing a family, mentally reconstruct every source unit independently. If one base-plus-delta reconstruction loses information, adds unsupported scope or behavior, or changes when the source rule applies, revise the base or delta, split the variant group, remove that unit from the family, or reject the family.

A delta may add source-supported differences to the corresponding base fields. It must never negate, contradict, replace, or correct the base.

When the base is not expressed at exactly the same scope by any source unit, construct only the narrowest non-vague abstraction needed to capture the shared normative core. Do not infer universal scope from a finite set of examples, and do not add quantifiers such as "all", "always", or "regardless of" unless they are supported by the input units.

Create a family when:

- units are duplicates or paraphrases of the same rule;
- one unit states a general expectation and another applies that expectation under a narrower condition;
- units require the same outcome but one specifies a concrete mechanism for achieving it;
- one unit retains the shared expectation and adds another independent requirement;
- units require different concrete behaviors that preserve the same specific, testable design invariant;
- units exhibit different failure manifestations of the same invariant, with those differences preserved in deltas;
- incidental differences in terminology, project, platform, or example can be represented without changing the normative core.

Do not create a family when:

- the only commonality is a broad topic, UI component, feature, or keyword;
- the proposed base would be vague or non-actionable;
- members protect different underlying properties, even if their topics or components are similar;
- one member contradicts the proposed base;
- reconstructing a member would require replacing the base behavior rather than adding a delta;
- the proposed delta would contain most of the original rule because little meaningful content is actually shared;
- grouping the units would hide a distinction that matters when deciding whether the rule applies;
- variants require incompatible behavior under overlapping applicability conditions;
- the relationship is uncertain.

Each input unit may appear in at most one family. A family must contain at least two source units. Omit any input unit that does not belong to a valid family.

## Required Reasoning Procedure

For each candidate set:

1. Identify the recurring operational situation, the relationships among relevant operations or states, the trigger, required behavior, and prohibited outcome expressed by each unit; consult applicability conditions as auxiliary scope evidence.
2. Identify candidate groups that share the same atomic expectation or specific design invariant.
3. Propose the most informative and specific base unit shared by every member of each candidate group.
4. Reject the group if the base becomes vague, unsupported, or merely topical.
5. Apply the reconstruction test separately to every member.
6. Put base-equivalent IDs in `equivalent_member_ids`.
7. Partition every other member into variant groups: members in the same group must be mutually equivalent and require exactly the same delta from the base.
8. Apply the bidirectional reconstruction test to every ID, including every member of a multi-ID variant group.
9. Ensure that no source ID appears in more than one family.
10. When uncertain whether a shared base is valid, omit the questionable unit or omit the entire family.

Return only the required JSON result.

## Examples

The IDs in these examples are illustrative and must never be copied into the actual output.

### Example A: General Rule and Condition-Specific Rule

Input meanings:

- `EXAMPLE_1`: pasted content should be preserved completely;
- `EXAMPLE_2`: pasted Unicode content should be preserved completely.

Valid family:

```json
{
  "base_unit": {
    "situation": "pasting clipboard content into a destination that accepts text",
    "commonsense_rule": "Pasting should preserve the complete copied content without truncation or data loss."
  },
  "equivalent_member_ids": [
    "EXAMPLE_1"
  ],
  "variant_groups": [
    {
      "member_ids": [
        "EXAMPLE_2"
      ],
      "delta": {
        "situation_differences": [
          "The pasted content contains Unicode or multi-byte characters."
        ],
        "commonsense_rule_differences": []
      }
    }
  ],
  "family_rationale": "Both units require complete preservation of pasted content; the second unit restricts the content type."
}
```

### Example B: Base Behavior with an Additional Requirement

Input meanings:

- `EXAMPLE_3`: opening a navigation overlay should dismiss the virtual keyboard;
- `EXAMPLE_4`: opening a navigation overlay should dismiss the virtual keyboard and obscure the underlying interface.

These units may share the keyboard-dismissal base. Put `EXAMPLE_3` in `equivalent_member_ids`. Put `EXAMPLE_4` in a single-member `variant_group`, with obscuring the underlying interface recorded under `delta.commonsense_rule_differences`. Do not place `EXAMPLE_4` in `equivalent_member_ids`.

### Example C: Equivalent Variants Sharing One Delta

Input meanings:

- `EXAMPLE_5`: pasting Unicode text should preserve all copied content;
- `EXAMPLE_6`: Unicode clipboard content should be pasted without truncation or loss;
- `EXAMPLE_7`: pasting ordinary text should preserve all copied content.

If `EXAMPLE_5` and `EXAMPLE_6` are equivalent formulations of the same Unicode-specific rule, place them together in one `variant_group.member_ids` array with the Unicode-content constraint in their shared delta. If `EXAMPLE_7` is equivalent to the family base, place it in `equivalent_member_ids`.

### Example D: Same Object but Different Rules

Input meanings:

- a virtual keyboard should use the correct language layout;
- a virtual keyboard should not cover selectable suggestions;
- opening a virtual keyboard should not change display orientation.

Do not place these units in one family. They concern the same object but express different normative expectations. A base such as "the virtual keyboard should behave correctly" would be invalid because it is vague and non-actionable.

## Output Schema

Return exactly one JSON object with the following structure:

```json
{
  "rule_families": [
    {
      "base_unit": {
        "situation": "...",
        "commonsense_rule": "..."
      },
      "equivalent_member_ids": [
        "INPUT_ID_1"
      ],
      "variant_groups": [
        {
          "member_ids": [
            "INPUT_ID_2",
            "INPUT_ID_3"
          ],
          "delta": {
            "situation_differences": [
              "..."
            ],
            "commonsense_rule_differences": []
          }
        }
      ],
      "family_rationale": "..."
    }
  ]
}
```

## Output Requirements

- Return one valid JSON object and no other text.
- The root object must contain exactly one field: `rule_families`.
- Every family object must contain exactly `base_unit`, `equivalent_member_ids`, `variant_groups`, and `family_rationale`.
- Every `base_unit` must contain exactly `situation` and `commonsense_rule`.
- `base_unit.situation`, `base_unit.commonsense_rule`, and `family_rationale` must be non-empty strings.
- Every individual situation element and behavioral statement in the `base_unit` must be supported by every member of that family; majority support is not sufficient.
- `equivalent_member_ids` must be an array of distinct input-ID strings. It may be empty only when no source unit expresses the shared base at exactly the same semantic scope.
- Every `variant_groups` item must contain exactly `member_ids` and `delta`.
- Every `variant_group.member_ids` value must be a non-empty array of distinct input-ID strings. All IDs in the array must be mutually equivalent and share exactly the same delta from the base. A one-ID array is valid.
- Every `delta` must contain exactly `situation_differences` and `commonsense_rule_differences`.
- Every item in a delta array must be a concise, non-empty string. Individual arrays may be empty, but every variant delta must have at least one non-empty array.
- Every delta item must state member-specific information absent from the corresponding `base_unit` field; do not output a complete rewritten variant unit.
- `applicability_conditions` may affect only the decision to group, separate, or omit units. No output text may be copied, paraphrased, inferred, or transferred from that field.
- Use only batch-local IDs present in the input and reproduce them exactly.
- Do not output invented IDs, family IDs, group IDs, batch IDs, or canonical IDs.
- The union of `equivalent_member_ids` and all `variant_groups.member_ids` values must contain at least two distinct input IDs in each family.
- An ID must not appear in both `equivalent_member_ids` and any `variant_group.member_ids` array in the same family.
- No input ID may appear in more than one family.
- Omit units that do not belong to a valid family instead of reporting them.
- Do not output empty delta objects for IDs listed in `equivalent_member_ids`.
- Before returning the result, verify for every included ID that its base-plus-delta reconstruction is complete and contains no unsupported additions.
- Do not output pairwise similarity, conflict, or unrelatedness records.
- If no valid family exists, return `{"rule_families": []}`.
- Keep every base, delta item, and rationale concise while preserving the required meaning.
