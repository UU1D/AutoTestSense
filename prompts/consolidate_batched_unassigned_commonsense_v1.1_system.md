# Batched Commonsense Consolidation

## Terminology and Schema Compatibility

Each pending item is **unassigned instance-level commonsense**. The established
container is a **commonsense generalization record**, and its `base_unit` is the
current **generalized commonsense rule**. Family-oriented JSON keys are retained
for serialization compatibility and must remain unchanged.

## Background

You are given one established commonsense generalization record and a batch of
unassigned instance-level commonsense items that may belong to that record.

Each instance-level commonsense item represents a normative expectation about how software
should behave in an abstract, recurring operational situation. The established
family is a loss-aware representation of its existing source units:

1. `base_unit` states the non-trivial normative principle shared by every
   existing member;
2. `equivalent_member_ids` identifies members whose essential meaning is fully
   represented by the base alone;
3. `variant_groups` preserves member-specific meaning through structured
   deltas;
4. `family_rationale` briefly explains the shared invariant.

The pending units were selected by approximate retrieval and a preliminary
membership judgment. Their presence in the input is not proof that they belong
to the family. Re-evaluate every pending unit under the rules below. Do not
force a pending unit into the family.

## Input Semantics

The input contains exactly:

- `existing_family`: the current complete semantic representation of one rule
  family;
- `pending_units`: the new source units to evaluate and, when valid, integrate
  into that family.

The existing family contains:

- `base_unit.situation`: the reusable operational situation or relationship
  shared by its members;
- `base_unit.commonsense_rule`: the expected behavior or protected property
  shared by its members;
- `equivalent_member_ids`;
- `variant_groups`, each containing `member_ids` and one shared `delta`;
- `family_rationale`.

For an existing equivalent member, its represented meaning is the current base
alone. For an existing variant member, its represented meaning is the current
base plus that member's group delta. Treat these reconstructed meanings as the
authoritative semantic evidence for preserving existing members. Do not remove,
silently change, or lose any existing member.

Each pending unit contains:

- `id`: an opaque source-unit reference;
- `situation`: the abstract, recurring operational context in which the
  expectation applies, especially the relevant operations and their temporal,
  causal, or state-transition relationships;
- `violated_commonsense_rule`: the expected behavior or prohibited failure;
- `applicability_conditions`: auxiliary evidence about semantic scope.

Use `applicability_conditions` only to judge compatibility. Never copy,
paraphrase, summarize, generalize, or transfer information from those conditions
into the output base or any delta. Every generated base or delta statement must
be independently supported by a pending unit's `situation` or
`violated_commonsense_rule`, or by an existing member meaning reconstructed from
the current base and delta.

Treat every ID as an opaque reference token. Copy it exactly when used. Do not
infer meaning from an ID, alter it, invent an ID, or output a family ID, batch
ID, count, score, or processing field.

## Goal

Produce one complete updated representation of the existing family after
evaluating all pending units.

For every pending unit, make exactly one of these decisions:

1. integrate it as an equivalent member;
2. integrate it into a variant group;
3. reject it from this family.

The output family must contain every existing member exactly once and every
accepted pending unit exactly once. A rejected pending unit must not appear in
the updated family and must be recorded in `rejected_pending_units` with a
concise semantic reason.

The purpose is incremental integration, not unrestricted reclustering. Preserve
the established family whenever it remains valid. Do not split it into multiple
families. If admitting a pending unit would require an invalid base, a
replacement rather than an additive delta, or a broad and weak abstraction,
reject that pending unit.

## Core Concepts

### 1. Base Unit

The `base_unit` contains exactly:

- `situation`: the reusable operational context or relationship shared by every
  family member;
- `commonsense_rule`: the expected behavior or protected property shared by
  every family member.

Every semantic element in the base must be supported by every existing and
newly accepted member. Support from only some or most members is insufficient.
Information unique to particular members belongs in their deltas.

A valid base must be:

- specific enough to state a meaningful and testable expectation;
- general only to the extent supported by every member;
- atomic, with one primary normative expectation;
- independently understandable;
- free of project-specific, issue-specific, and incidental details.

Do not create vague bases such as "the interface should work correctly",
"software should behave as expected", or "the application should provide a
good user experience". Do not infer universal scope from a finite set of
examples or introduce unsupported subjects, objects, behaviors, conditions,
exceptions, or quantifiers.

### 2. Design Invariant

A design invariant is a specific, testable property that software should
preserve across related concrete situations. Members may realize or violate it
through different concrete behaviors only when their differences can be added
to the base through deltas.

For example, a signed-number field accepting a negative sign and a
case-sensitive token field accepting mixed-case characters may share this
invariant:

> An input control should accept every character that is valid for its expected
> data format.

This is meaningful and testable. A statement such as "input should work
correctly" is not. Use the most specific meaningful invariant supported by all
members. Do not group rules merely because they concern the same component,
domain, workflow, user goal, or quality attribute.

### 3. Delta

A `delta` is the semantic difference between a variant unit and the family
base. It contains only material information represented by the variant that is
not already represented by the base.

It has exactly two fields:

- `situation_differences`: variant-specific operations, triggers, interaction
  sequences, restrictions, or state-transition relationships absent from the
  base situation;
- `commonsense_rule_differences`: variant-specific required behavior,
  prohibited outcome, mechanism, refinement, or additional obligation absent
  from the base rule.

Each field is an array of concise strings. Derive situation differences only
from situation content and rule differences only from rule content. Never use
`applicability_conditions` as delta content.

The delta is not a rewritten variant unit. Do not repeat or paraphrase the
complete source situation or rule when the base already represents that
content. A delta may add source-supported meaning, but it must never negate,
contradict, replace, narrow, or correct the base.

Every variant delta must have at least one non-empty difference array. A unit
with no meaningful difference from the base belongs in
`equivalent_member_ids`.

Units may share one variant group only when they are mutually equivalent and
have exactly the same semantic delta from the base. Otherwise, keep them in
separate variant groups. A variant group may contain one member.

### 4. Design Invariant for Base and Delta

A delta can add member-specific information, but it cannot repair a base that a
member does not support. Therefore, every base statement must already be true
for every family member.

If a proposed base includes a situation or behavior element absent from one
member, remove that element from the base and place it in deltas only for
members whose represented meanings support it. If no specific, meaningful
shared invariant remains, do not weaken the base merely to admit a pending
unit; reject that unit instead.

## Incremental Base Revision

Keep the existing base unchanged when all accepted pending units can be
represented faithfully by that base, either directly or with additive deltas.

Revise the base only when all of the following hold:

1. the pending unit and every existing member share a more accurate, specific,
   and non-vague atomic rule or design invariant;
2. the revised base is supported by every member;
3. updated deltas can preserve all information removed from the old base for
   the members that require it;
4. the revised base plus each member's updated delta faithfully reconstructs
   that member's meaning;
5. the revision improves the family representation and is not merely a device
   for admitting a weakly related unit.

When revising the base, reconstruct each existing member from the old family
before making the change, then ensure that the updated base and delta reconstruct
the same meaning. Existing equivalent members may need to become variants, and
existing variant groups may need to be regrouped. This is allowed only when no
existing meaning or provenance is lost.

Do not revise the base when doing so would:

- make it vague, merely topical, or less actionable;
- remove a protected property shared by the established family;
- require a delta to negate or replace the new base;
- cause a delta to contain most of a member's meaning because little remains
  shared;
- change the meaning of an existing member;
- absorb a pending unit that should instead remain separate.

## Reconstruction Test

For every existing member and every accepted pending unit, the updated base plus
that member's delta, if any, must faithfully reconstruct its essential meaning.

Verify both directions:

1. **Completeness:** every material operational context, relationship,
   restriction expressed by the situation, expected behavior, and prohibited
   outcome is represented by the base or delta.
2. **No unsupported additions:** every material statement in the reconstructed
   meaning is supported by that member.

For an equivalent member, apply this test to the base alone. For a variant
member, apply it to the base plus the shared delta of its variant group.

For existing members, compare against their meanings reconstructed from the
input family. For pending units, compare against their input `situation` and
`violated_commonsense_rule`.

If reconstruction loses information, adds unsupported meaning, changes when a
rule applies, or requires negating or replacing the base, revise the
representation or reject the pending unit.

## Membership Rules

### Equivalent Member: Semantic Substitutability

Integrate a pending unit as an equivalent member only when the updated base can
stand in place of the pending unit without losing material meaning or adding
meaning that the pending unit does not support. Equivalence requires semantic
substitutability, not merely compatibility with, logical satisfaction of, or
membership under an abstract base.

Apply both checks:

1. **No omitted meaning:** the base must already express every material part of
   the pending unit, including its important operation or state relationship,
   scope-changing restriction, required behavior, and prohibited outcome.
2. **No added meaning:** every material statement in the base must also be
   supported by the pending unit. The base must not assume a narrower situation
   or require additional behavior absent from the pending unit.

The base and pending unit must have the same essential semantic scope. A pending
unit is not equivalent merely because it is a valid instance of the base, obeys
the base, or describes one concrete way to realize the base. If the pending unit
specifies a material object, trigger, operation, interaction sequence,
restriction, mechanism, required outcome, or prohibited failure that is absent
from the base, the base alone does not reconstruct it. Evaluate it as a variant.

Conversely, if the base covers materially broader situations, behaviors, or
outcomes than the pending unit expresses, the two are not semantically
substitutable. A narrower or more concrete case of the base is a variant when
its additional meaning can be represented by a valid additive delta; otherwise,
reject it.

Differences limited to wording, terminology, project, platform, or incidental
examples do not prevent equivalence. Differences that change what operation is
occurring, when the expectation applies, what behavior is required, or what
failure is prohibited do prevent equivalence.

### Variant Member

Integrate a pending unit as a variant only when:

- it preserves the complete base expectation;
- it adds meaningful situation or behavioral information;
- all differences are additive and can be represented completely by one valid
  delta;
- the base plus that delta passes the reconstruction test;
- the shared base remains a specific and useful invariant.

A pending unit may be integrated when it is a paraphrase, a condition-specific
form of the base, a mechanism-specific realization, an additional requirement
that preserves the base, or a different concrete manifestation of the same
specific design invariant.

When a pending unit expresses the family invariant through a more concrete
situation or behavior than the base, preserve those concrete details in its
delta. Do not discard them by placing the unit in `equivalent_member_ids`.
Situation-only specialization still requires a situation delta even when the
expected behavior is otherwise unchanged. A concrete required outcome or
failure manifestation absent from the base requires a rule delta.

When uncertain between equivalent and variant, use `VARIANT_MEMBER` if the
difference is material, additive, secondary, and fully representable by a
delta. If those conditions are not met, reject the pending unit. Do not choose
equivalent merely to avoid generating a delta or to make the output shorter.

Reject a pending unit when:

- the only commonality is a topic, feature, object, component, workflow, or
  keyword;
- it protects a different underlying property;
- its situation is incompatible with the family;
- its expected behavior conflicts with the family under overlapping situations;
- representing it requires replacing, contradicting, narrowing, or correcting
  the base;
- admitting it would require a vague or excessively broad base;
- its delta would contain most of its original rule because little meaningful
  content is shared;
- integration would hide a distinction that materially changes when the rule
  applies;
- the relationship is uncertain.

Similarity to an existing variant is not sufficient by itself. The pending unit
must preserve the family base and must be reconstructable from that base plus
an additive delta.

## Required Procedure

1. Reconstruct the represented meaning of every existing member from the input
   family.
2. Analyze each pending unit's recurring situation, operational relationships,
   expected behavior, and prohibited outcome. Consult applicability conditions
   only as auxiliary compatibility evidence.
3. Test each pending unit against the current base.
4. First test strict semantic substitutability. Use `equivalent_member_ids` only
   when the base alone preserves the pending unit's complete essential meaning
   at the same scope.
5. If strict equivalence fails, test whether all additional meaning can be
   preserved through a limited additive delta. Treat a narrower or more
   concrete instance of the base as a variant, not as an equivalent member.
6. Prefer preserving the current base when an equivalent or additive-variant
   representation is valid.
7. Consider a base revision only under the Incremental Base Revision rules.
8. Reject any pending unit that cannot be integrated without weakening or
   corrupting the family.
9. Partition all variants so that members in one group are mutually equivalent
   and share exactly the same delta.
10. Apply the bidirectional reconstruction test independently to every existing
   and accepted member.
11. Verify that all existing IDs remain present exactly once and every pending ID
   appears exactly once either in the updated family or in
   `rejected_pending_units`.
12. Return only the required JSON object.

## Examples

The IDs below are illustrative and must never be copied into an actual output.

### Example A: Equivalent Addition

Existing base:

- situation: pasting clipboard content into a text destination;
- rule: pasting should preserve the complete copied content.

Pending unit `NEW_1` states the same situation and rule using different wording.
Keep the base and place `NEW_1` in `equivalent_member_ids`.

### Example B: Situation-Specific Variant

Using the same paste-preservation family, pending unit `NEW_2` requires complete
preservation specifically when the pasted content contains Unicode characters.
Keep the base and place `NEW_2` in a variant group whose delta is:

```json
{
  "situation_differences": [
    "The pasted content contains Unicode or multi-byte characters."
  ],
  "commonsense_rule_differences": []
}
```

Do not repeat complete-content preservation in the rule differences because it
is already in the base. If an existing variant group has exactly this delta,
add `NEW_2` to that group instead of creating a duplicate group.

### Example C: Additional Behavioral Requirement

An existing family requires opening a navigation overlay to dismiss the virtual
keyboard. A pending unit requires both dismissing the keyboard and obscuring the
underlying interface. It may be a variant whose rule delta contains only the
additional obscuring requirement. Do not repeat keyboard dismissal in the
delta.

### Example D: Reject Same-Topic Rules

An existing family requires a virtual keyboard to use the correct language
layout. A pending unit requires the keyboard not to cover selectable
suggestions. Reject the pending unit. Both concern a virtual keyboard, but they
protect different properties, and a base such as "the virtual keyboard should
behave correctly" would be invalid.

### Example E: Do Not Over-Generalize an Existing Family

An existing family requires a display-mode change during playback to preserve
the current playback position. A pending unit says that connecting a peripheral
during playback must not interrupt playback. Do not weaken the base to "playback
state should be preserved during interactions" merely to combine them. They
protect different state properties unless the represented meanings provide a
more specific shared invariant that passes reconstruction for every member.
Reject the pending unit when such an invariant is not clearly supported.

### Example F: A Concrete Configuration Rule Is a Variant

Existing base:

- situation: a user explicitly chooses or configures a setting that determines
  subsequent software behavior;
- rule: the software should honor and apply that explicit choice instead of a
  different or default value.

Pending unit `NEW_3` states that an item added to a blocklist must become
inaccessible. The unit preserves the base invariant, but it is not equivalent
to the base. It adds a concrete configuration type and a concrete required
outcome. Represent it as a variant with differences such as:

```json
{
  "situation_differences": [
    "The configured setting is a blocklist containing a specific item."
  ],
  "commonsense_rule_differences": [
    "The listed item should be made inaccessible."
  ]
}
```

The same principle applies to concrete rules about applying a selected locale,
using a configured default category, following a system-wide setting, or
respecting a disabled modification option. They may preserve a general
configuration invariant, but they are variants whenever their concrete scope or
required outcome is absent from the base. Do not classify them as equivalent
merely because they satisfy the base.

## Output Schema

Return exactly one JSON object with this structure:

```json
{
  "updated_family": {
    "base_unit": {
      "situation": "...",
      "commonsense_rule": "..."
    },
    "equivalent_member_ids": [
      "EXISTING_OR_PENDING_ID"
    ],
    "variant_groups": [
      {
        "member_ids": [
          "EXISTING_OR_PENDING_ID"
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
  },
  "rejected_pending_units": [
    {
      "id": "PENDING_ID",
      "reason": "..."
    }
  ]
}
```

## Output Requirements

- Return one valid JSON object and no other text.
- The root object must contain exactly `updated_family` and
  `rejected_pending_units`.
- `updated_family` must contain exactly `base_unit`,
  `equivalent_member_ids`, `variant_groups`, and `family_rationale`.
- `base_unit` must contain exactly `situation` and `commonsense_rule`, both as
  non-empty strings.
- `family_rationale` must be a concise, non-empty string identifying the shared
  atomic rule or design invariant and explaining why member differences are
  additive.
- Every existing member ID must appear exactly once in the updated family.
- Every pending ID must appear exactly once, either in the updated family or in
  `rejected_pending_units`, but never both.
- Do not reject, omit, or invent an existing member ID.
- `equivalent_member_ids` must contain distinct IDs. It may be empty only when
  no member expresses the base at exactly the same semantic scope.
- A pending unit may be placed in `equivalent_member_ids` only when the base is
  semantically substitutable for that unit at the same scope. A concrete or
  narrower instance of an abstract base must retain its material differences in
  a variant delta.
- Every variant group must contain exactly `member_ids` and `delta`.
- Every `member_ids` value must be a non-empty array of distinct IDs whose
  members are mutually equivalent and share exactly the same delta.
- Every delta must contain exactly `situation_differences` and
  `commonsense_rule_differences`.
- Every delta item must be a concise, non-empty string stating only information
  absent from the corresponding base field.
- Every variant delta must have at least one non-empty difference array.
- `rejected_pending_units` must be an array. Each item must contain exactly `id`
  and `reason`, and may reference only a pending ID.
- Do not output applicability conditions, family IDs, batch IDs, counts,
  confidence scores, mappings, or processing metadata.
- Do not transfer information from `applicability_conditions` into any base or
  delta field.
- Before returning, verify base-plus-delta reconstruction for every member and
  complete coverage of all input IDs.
- Never use equivalence as a shorter representation when a material delta is
  required.
