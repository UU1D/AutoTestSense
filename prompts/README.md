# Prompt Reference

Prompt filenames use English `snake_case`:

```text
<action>_<object>_v<version>[_system|_user].md
```

System prompts contain stable task definitions, decision criteria, terminology,
and output constraints. User prompts contain the per-call data placeholder.
The extraction prompt has no role suffix because the caller assembles it as one
multimodal user message.

## Pipeline Prompts

### `extract_instance_level_commonsense_v1.2.md`

- **Purpose:** determine whether a software report supports a candidate
  commonsense violation and extract one Instance-Level Commonsense item.
- **Stage:** 1, Instance-Level Commonsense Extraction.
- **Caller:** `commonsense_repro.extraction.batch_build_llm_prompts`.
- **Placeholder:** `{BUG_REPORT}`.
- **Output:** candidate decision, confidence, evidence, and
  `violated_common_sense`.

### `generalize_local_commonsense_v1.4_{system,user}.md`

- **Purpose:** generalize compatible Instance-Level Commonsense items from one
  SNN/Leiden community into Commonsense Generalization Records.
- **Stage:** 3, Initial Commonsense Generalization.
- **Caller:**
  `commonsense_repro.commonsense_generalization.local.generalize_local_commonsense`.
- **Placeholder:** `{{units_json}}`.
- **Output:** `base_unit`, `equivalent_member_ids`, `variant_groups`, and
  `family_rationale`.

### `group_global_generalization_candidates_v1.0_{system,user}.md`

- **Purpose:** identify which candidate records should enter the same global
  generalization call without rewriting their content.
- **Stage:** 4, candidate grouping.
- **Caller:**
  `commonsense_repro.commonsense_generalization.global.candidate_grouping.group_global_generalization_candidates`.
- **Placeholder:** `{{units_json}}`.
- **Output:** `groups`; ungrouped records are retained by the caller.

### `generalize_global_commonsense_v1.2_{system,user}.md`

- **Purpose:** re-evaluate expanded source items from grouped records and
  produce final cross-community Commonsense Generalization Records.
- **Stage:** 4, Global Commonsense Generalization.
- **Caller:**
  `commonsense_repro.commonsense_generalization.global.generalization.generalize_global_commonsense`.
- **Placeholder:** `{{input_json}}`.
- **Output:** complete compatible `rule_families` objects.

### `judge_unassigned_commonsense_consolidation_v1.2_{system,user}.md`

- **Purpose:** decide whether one unassigned Instance-Level Commonsense item is
  an equivalent member, a variant member, or no match for all candidates.
- **Stage:** 5, preliminary Commonsense Consolidation decision.
- **Callers:**
  `commonsense_repro.commonsense_consolidation.serial_consolidation.run_serial_consolidation`
  and
  `commonsense_repro.commonsense_consolidation.consolidation_judgment.judge_consolidation_membership`.
- **Placeholder:** `{{formatted_input}}`.
- **Output:** `ADD_TO_FAMILY` or `NO_MATCH` with the decision fields required
  by the pipeline.

### `consolidate_unassigned_commonsense_v1.0_{system,user}.md`

- **Purpose:** consolidate one item judged as `VARIANT_MEMBER` into the selected
  record by reusing a delta group, creating a delta group, revising the complete
  record when necessary, or rejecting the item.
- **Stage:** 5, serial Commonsense Consolidation. Equivalent members are added
  deterministically and do not invoke this prompt.
- **Caller:**
  `commonsense_repro.commonsense_consolidation.serial_consolidation.run_serial_consolidation`.
- **Placeholder:** `{{input_json}}`.
- **Output:** `DELTA_INTEGRATION`, `FULL_RULE_FAMILY`, or `REJECT`.

## Optional Batch Prompt

### `consolidate_batched_unassigned_commonsense_v1.1_{system,user}.md`

- **Purpose:** consolidate a batch of provisionally related unassigned items
  into one record while rejecting incompatible items.
- **Caller:**
  `commonsense_repro.commonsense_consolidation.batch_consolidation.run_batch_consolidation`.
- **Placeholder:** `{{input_json}}`.
- **Output:** one complete updated record plus rejected items.
- **Pipeline status:** retained as an optional tool; the official Stage 5 entry
  uses strict serial consolidation.

## Stage Mapping

| Stage | Prompt |
|---|---|
| 1 | `extract_instance_level_commonsense_v1.2` |
| 2 | none: embedding, reranking, SNN, Louvain, and Leiden |
| 3 | `generalize_local_commonsense_v1.4` |
| 4 grouping | `group_global_generalization_candidates_v1.0` |
| 4 generalization | `generalize_global_commonsense_v1.2` |
| 5 judgment | `judge_unassigned_commonsense_consolidation_v1.2` |
| 5 consolidation | `consolidate_unassigned_commonsense_v1.0` |
| 6 | none: library and vector assembly |
| 7 | none: retrieval and reranking |

## Files

- `extract_instance_level_commonsense_v1.2.md`
- `generalize_local_commonsense_v1.4_system.md`
- `generalize_local_commonsense_v1.4_user.md`
- `group_global_generalization_candidates_v1.0_system.md`
- `group_global_generalization_candidates_v1.0_user.md`
- `generalize_global_commonsense_v1.2_system.md`
- `generalize_global_commonsense_v1.2_user.md`
- `judge_unassigned_commonsense_consolidation_v1.2_system.md`
- `judge_unassigned_commonsense_consolidation_v1.2_user.md`
- `consolidate_unassigned_commonsense_v1.0_system.md`
- `consolidate_unassigned_commonsense_v1.0_user.md`
- `consolidate_batched_unassigned_commonsense_v1.1_system.md`
- `consolidate_batched_unassigned_commonsense_v1.1_user.md`
