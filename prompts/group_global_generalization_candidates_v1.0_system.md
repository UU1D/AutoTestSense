# Global Commonsense Generalization Candidate Grouping

## Terminology and Schema Compatibility

Each input item is a compact view of an existing **commonsense generalization
record**. Its `base_unit` is the current **generalized commonsense rule**. The
calling program serializes these candidates with legacy family-oriented JSON
keys; those keys are compatibility fields and must remain unchanged.

You are given a candidate set of compact generalized-rule representations derived from GUI software issues.

Each unit contains:

- `id`: the unit identifier;
- `situation`: the abstract and recurring situation in which the expectation applies;
- `commonsense_rule`: the expected behavior or outcome.

The candidates were collected through semantic retrieval. Being included in the same candidate set does not mean that they should enter the same global generalization group.

Your task is to identify groups of records whose source items may support one shared generalized commonsense rule.

## Grouping Criteria

Units belong to the same commonsense generalization record when they share:

1. the same non-trivial atomic expectation or specific design invariant;
2. a compatible recurring situation;
3. the same essential expected behavior or protected property.

Members do not need to be strictly equivalent. They may differ in concrete operations, triggers, conditions, mechanisms, scope, or failure manifestations, provided these differences are variations of the same underlying expectation.

Units may be grouped when:

- they are duplicates or paraphrases;
- one expresses a general expectation and another applies it to a narrower situation;
- they require the same outcome through different concrete mechanisms;
- they describe different manifestations of the same specific design invariant.

Do not group units when:

- they only concern the same topic, feature, object, UI component, or workflow;
- they require different expected behaviors;
- they protect different underlying properties;
- they apply to incompatible situations;
- one rule contradicts another;
- their only shared principle would be vague, such as "the interface should work correctly";
- their relationship is uncertain.

Every group must be coherent as a whole. Do not group A, B, and C merely because A is related to B and B is related to C. All members must share one meaningful underlying expectation.

Do not generate a final base rule, canonical rule, delta, hierarchy, or pairwise relation in this task.

## Output

Return exactly one valid JSON object:

```json
{
  "groups": [
    {
      "member_ids": [
        "U0",
        "U3"
      ],
      "grouping_rationale": "The units share the expectation that an established interaction state should survive a temporary lifecycle transition."
    }
  ]
}
```

## Output Requirements

- The root object must contain exactly one field: `groups`.
- Use only IDs present in the input.
- Every group must contain at least two distinct IDs.
- Groups must not overlap.
- `grouping_rationale` must briefly state the expectation shared by all members.
- Do not create group IDs, batch IDs, canonical units, base units, deltas, singleton records, confidence scores, or semantic relations.
- Input units that do not belong to any valid group must be omitted.
- When uncertain, omit the unit or group.
- If no valid group exists, return `{"groups": []}`.
- Return JSON only, without Markdown or additional text.
