"""Pure state, validation, and transformation logic for serial no-match consolidation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from commonsense_repro.common import llm_pipeline_common as llm_io
from commonsense_repro.commonsense_consolidation.generalization_embedding_index.generalization_embedding_index import (
    canonical_base,
    embedding_text_for_base,
    sha256_json,
)
from commonsense_repro.commonsense_consolidation.batch_consolidation.run_batch_consolidation import (
    family_member_ids,
    family_prompt_value,
    normalize_delta,
    normalize_family,
    resolve_updated_base_origin,
)


INITIAL_ORIGIN = "INITIAL_CATALOG"
DYNAMIC_ORIGIN = "INGESTED_NO_MATCH"


def require_unit(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object.")
    return {
        "id": llm_io.require_nonempty_string(value.get("id"), f"{location}.id"),
        "situation": llm_io.require_nonempty_string(
            value.get("situation"), f"{location}.situation"
        ),
        "violated_commonsense_rule": llm_io.require_nonempty_string(
            value.get("violated_commonsense_rule"),
            f"{location}.violated_commonsense_rule",
        ),
        "applicability_conditions": llm_io.normalize_string_array(
            value.get("applicability_conditions", []),
            f"{location}.applicability_conditions",
        ),
    }


def load_source_units(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        values = llm_io.read_json(path)
        if not isinstance(values, list):
            raise ValueError(f"{path} must contain a JSON array.")
        for index, raw in enumerate(values):
            record = require_unit(raw, f"{path}[{index}]")
            unit_id = record["id"]
            if unit_id in output:
                raise ValueError(f"Duplicate source unit ID: {unit_id}")
            output[unit_id] = record
    return output


def load_no_match_queue(path: Path) -> list[dict[str, Any]]:
    rows = llm_io.read_jsonl(path)
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        record = require_unit(raw, f"{path}:{index + 1}")
        order = raw.get("serial_order")
        if order != index:
            raise ValueError(f"{path}:{index + 1} has invalid serial_order {order!r}.")
        if record["id"] in seen:
            raise ValueError(f"Duplicate no-match unit ID: {record['id']}")
        queue.append({**record, "serial_order": order})
        seen.add(record["id"])
    if not queue:
        raise ValueError(f"No-match queue is empty: {path}")
    return queue


def family_by_id(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    families = catalog.get("rule_families")
    if not isinstance(families, list):
        raise ValueError("Catalog must contain a rule_families array.")
    output: dict[str, dict[str, Any]] = {}
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise ValueError(f"rule_families[{index}] must be an object.")
        family_id = llm_io.require_nonempty_string(
            family.get("family_id"), f"rule_families[{index}].family_id"
        )
        if family_id in output:
            raise ValueError(f"Duplicate family ID: {family_id}")
        if set(family_member_ids(family)) != set(family.get("member_ids", [])):
            raise ValueError(f"Family {family_id} has inconsistent member IDs.")
        output[family_id] = family
    return output


def semantic_hash(family: dict[str, Any]) -> str:
    return sha256_json(family_prompt_value(family))


def initialize_state(catalog: dict[str, Any], queue_size: int) -> dict[str, Any]:
    current = copy.deepcopy(catalog)
    current["catalog_version"] = 0
    family_ids = set(family_by_id(current))
    states = current.get("family_states")
    if not isinstance(states, list):
        states = []
        current["family_states"] = states
    state_by_id = {
        item.get("family_id"): item
        for item in states
        if isinstance(item, dict) and isinstance(item.get("family_id"), str)
    }
    for family_id in family_ids:
        if family_id not in state_by_id:
            states.append(
                {
                    "family_id": family_id,
                    "version": 0,
                    "state_origin": INITIAL_ORIGIN,
                    "base_representation_origin": INITIAL_ORIGIN,
                }
            )
        else:
            state_by_id[family_id].setdefault("base_representation_origin", INITIAL_ORIGIN)
    return {
        "schema_version": "1.0",
        "next_serial_order": 0,
        "queue_size": queue_size,
        "processed_unit_ids": [],
        "catalog": current,
        "statistics": {
            "processed": 0,
            "cache_hit_new_family": 0,
            "membership_calls": 0,
            "equivalent_additions": 0,
            "variant_additions": 0,
            "full_family_revisions": 0,
            "generalization_rejections": 0,
            "new_families": 0,
        },
    }


def validate_state(state: dict[str, Any], queue: list[dict[str, Any]]) -> None:
    next_order = state.get("next_serial_order")
    if not isinstance(next_order, int) or not 0 <= next_order <= len(queue):
        raise ValueError("State has an invalid next_serial_order.")
    processed = state.get("processed_unit_ids")
    expected = [row["id"] for row in queue[:next_order]]
    if processed != expected:
        raise ValueError("State processed_unit_ids do not match the queue prefix.")
    catalog = state.get("catalog")
    if not isinstance(catalog, dict) or catalog.get("catalog_version") != next_order:
        raise ValueError("State catalog_version must equal next_serial_order.")
    members: set[str] = set()
    for family in family_by_id(catalog).values():
        overlap = members & set(family_member_ids(family))
        if overlap:
            raise ValueError(f"Units occur in multiple families: {sorted(overlap)[0]}")
        members.update(family_member_ids(family))
    if members & set(row["id"] for row in queue[next_order:]):
        raise ValueError("A pending queue unit is already present in the catalog.")


def family_state_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    states = catalog.get("family_states")
    if not isinstance(states, list):
        raise ValueError("Catalog family_states must be an array.")
    output: dict[str, dict[str, Any]] = {}
    for index, state in enumerate(states):
        if not isinstance(state, dict):
            raise ValueError(f"family_states[{index}] must be an object.")
        family_id = llm_io.require_nonempty_string(
            state.get("family_id"), f"family_states[{index}].family_id"
        )
        if family_id in output:
            raise ValueError(f"Duplicate family state: {family_id}")
        output[family_id] = state
    return output


def annotate_candidate_origins(
    candidates: list[dict[str, Any]],
    *,
    catalog: dict[str, Any],
    processed_no_match_ids: set[str],
) -> list[dict[str, Any]]:
    states = family_state_map(catalog)
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        item = copy.deepcopy(candidate)
        family_id = item["family_id"]
        matched = item.get("matched_representation")
        if not isinstance(matched, dict):
            raise ValueError(f"Candidate {family_id} lacks matched_representation.")
        if matched.get("type") == "BASE":
            origin = states[family_id].get("base_representation_origin", INITIAL_ORIGIN)
            source_id = states[family_id].get("base_source_unit_id")
        elif matched.get("type") == "VARIANT_MEMBER":
            source_id = matched.get("member_id")
            origin = DYNAMIC_ORIGIN if source_id in processed_no_match_ids else INITIAL_ORIGIN
        else:
            raise ValueError(f"Unsupported representation type for {family_id}.")
        matched["representation_origin"] = origin
        if source_id is not None:
            matched["source_unit_id"] = source_id
        item["contains_consolidateed_no_match_evidence"] = origin == DYNAMIC_ORIGIN
        output.append(item)
    return output


def dynamic_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if candidate_has_dynamic_evidence(candidate)]


def membership_candidates(
    candidates: list[dict[str, Any]], *, dynamic_only: bool
) -> list[dict[str, Any]]:
    """Choose LLM candidates after dynamic evidence has triggered a call."""
    return dynamic_candidates(candidates) if dynamic_only else list(candidates)


def candidate_has_dynamic_evidence(candidate: dict[str, Any]) -> bool:
    """Recognize dynamic evidence even if an intermediate stage rebuilt the row."""
    if candidate.get("contains_consolidateed_no_match_evidence") is True:
        return True
    matched = candidate.get("matched_representation")
    return (
        isinstance(matched, dict)
        and matched.get("representation_origin") == DYNAMIC_ORIGIN
    )


def model_membership_candidates(
    candidates: list[dict[str, Any]],
    *,
    catalog: dict[str, Any],
    source_units: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    families = family_by_id(catalog)
    model_rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        alias = f"C{index}"
        family_id = candidate["family_id"]
        family = families[family_id]
        matched = candidate["matched_representation"]
        if matched["type"] == "BASE":
            model_matched: dict[str, Any] = {"type": "BASE"}
        else:
            member_id = llm_io.require_nonempty_string(
                matched.get("member_id"), f"candidate[{index}].member_id"
            )
            group_index = matched.get("variant_group_index")
            groups = family["variant_groups"]
            if not isinstance(group_index, int) or not 0 <= group_index < len(groups):
                raise ValueError(f"Invalid variant group for {family_id}/{member_id}.")
            original = source_units[member_id]
            model_matched = {
                "type": "VARIANT_MEMBER",
                "delta": groups[group_index]["delta"],
                "instance_level_commonsense": original,
            }
        model_rows.append(
            {
                "candidate_id": alias,
                "family_id": family_id,
                "base_unit": family["base_unit"],
                "matched_representation": model_matched,
            }
        )
        mapping[alias] = family_id
    return model_rows, mapping


def map_family_for_single_generalization(
    family: dict[str, Any], source_to_model: dict[str, str]
) -> dict[str, Any]:
    return {
        "base_unit": family["base_unit"],
        "equivalent_member_ids": [source_to_model[item] for item in family["equivalent_member_ids"]],
        "variant_groups": [
            {
                "group_ref": f"VG{index}",
                "member_ids": [source_to_model[item] for item in group["member_ids"]],
                "delta": group["delta"],
            }
            for index, group in enumerate(family["variant_groups"])
        ],
        "family_rationale": family["family_rationale"],
    }


def build_single_generalization_input(
    *,
    family: dict[str, Any],
    new_unit: dict[str, Any],
    preliminary_rationale: str,
    source_units: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    existing_ids = family_member_ids(family)
    all_ids = [*existing_ids, new_unit["id"]]
    model_to_source = {f"U{index}": item for index, item in enumerate(all_ids)}
    source_to_model = {value: key for key, value in model_to_source.items()}
    existing_units = []
    for member_id in existing_ids:
        original = source_units.get(member_id)
        if original is None:
            raise ValueError(f"Missing original source unit {member_id}.")
        existing_units.append({**original, "id": source_to_model[member_id]})
    model_new = {**new_unit, "id": source_to_model[new_unit["id"]]}
    return (
        {
            "existing_family": map_family_for_single_generalization(family, source_to_model),
            "existing_member_units": existing_units,
            "new_unit": model_new,
            "preliminary_judgment": {
                "membership_type": "VARIANT_MEMBER",
                "rationale": preliminary_rationale,
            },
        },
        model_to_source,
    )


def parse_single_generalization_output(
    raw_content: str,
    *,
    model_to_source: dict[str, str],
    variant_group_count: int,
) -> dict[str, Any]:
    parsed = json.loads(llm_io.strip_json_fence(raw_content))
    if not isinstance(parsed, dict):
        raise ValueError("Single-unit generalization output must be an object.")
    result_type = parsed.get("result_type")
    if result_type == "DELTA_INTEGRATION":
        action = parsed.get("action")
        if action == "ADD_TO_EXISTING_GROUP":
            llm_io.require_exact_keys(
                parsed, {"result_type", "action", "target_group_ref"}, "output"
            )
            group_ref = llm_io.require_nonempty_string(
                parsed["target_group_ref"], "output.target_group_ref"
            )
            allowed_refs = {f"VG{index}" for index in range(variant_group_count)}
            if group_ref not in allowed_refs:
                raise ValueError(f"Unknown target_group_ref: {group_ref}")
            return {"result_type": result_type, "action": action, "target_group_ref": group_ref}
        if action == "CREATE_NEW_GROUP":
            llm_io.require_exact_keys(parsed, {"result_type", "action", "delta"}, "output")
            return {
                "result_type": result_type,
                "action": action,
                "delta": normalize_delta(parsed["delta"], "output.delta"),
            }
        raise ValueError(f"Unsupported DELTA_INTEGRATION action: {action!r}")
    if result_type == "FULL_RULE_FAMILY":
        llm_io.require_exact_keys(parsed, {"result_type", "rule_family"}, "output")
        aliases = set(model_to_source)
        family = normalize_family(
            parsed["rule_family"], allowed_ids=aliases, location="output.rule_family"
        )
        if set(family["member_ids"]) != aliases:
            raise ValueError("FULL_RULE_FAMILY must contain every input ID exactly once.")
        family["equivalent_member_ids"] = llm_io.restore_ids(
            family["equivalent_member_ids"], model_to_source
        )
        for group in family["variant_groups"]:
            group["member_ids"] = llm_io.restore_ids(group["member_ids"], model_to_source)
        family["variant_groups"].sort(
            key=lambda item: llm_io.natural_sort_key(item["member_ids"][0])
        )
        family["member_ids"] = family_member_ids(family)
        family["member_count"] = len(family["member_ids"])
        return {"result_type": result_type, "rule_family": family}
    if result_type == "REJECT":
        llm_io.require_exact_keys(parsed, {"result_type", "reason"}, "output")
        return {
            "result_type": result_type,
            "reason": llm_io.require_nonempty_string(parsed["reason"], "output.reason"),
        }
    raise ValueError(f"Unsupported result_type: {result_type!r}")


def _refresh_family_metadata(
    family: dict[str, Any], *, family_id: str, previous: dict[str, Any] | None
) -> dict[str, Any]:
    family["family_id"] = family_id
    family["member_ids"] = family_member_ids(family)
    family["member_count"] = len(family["member_ids"])
    if previous is None:
        family["family_origin"] = "SERIAL_NO_MATCH_CREATED"
        family["base_origin"] = "ANCHORED"
        family["provenance"] = {"input_preparation_origin": family["member_ids"][0]}
    else:
        family["family_origin"] = "SERIAL_NO_MATCH_UPDATED"
        family["base_origin"] = resolve_updated_base_origin(
            previous_base_unit=previous["base_unit"],
            previous_base_origin=previous["base_origin"],
            updated_base_unit=family["base_unit"],
        )
        provenance = copy.deepcopy(previous.get("provenance", {}))
        provenance["input_preparation_updated"] = True
        family["provenance"] = provenance
    return family


def create_new_family(unit: dict[str, Any], family_id: str) -> dict[str, Any]:
    family = {
        "base_unit": {
            "situation": unit["situation"],
            "commonsense_rule": unit["violated_commonsense_rule"],
        },
        "equivalent_member_ids": [unit["id"]],
        "variant_groups": [],
        "family_rationale": (
            "The base directly preserves the situation and expected behavior of "
            "the family's currently sole source unit."
        ),
    }
    return _refresh_family_metadata(family, family_id=family_id, previous=None)


def apply_equivalent(family: dict[str, Any], unit_id: str) -> dict[str, Any]:
    updated = copy.deepcopy(family)
    if unit_id in set(family_member_ids(updated)):
        raise ValueError(f"Unit {unit_id} is already in family {family['family_id']}.")
    updated["equivalent_member_ids"].append(unit_id)
    updated["equivalent_member_ids"].sort(key=llm_io.natural_sort_key)
    return _refresh_family_metadata(updated, family_id=family["family_id"], previous=family)


def apply_single_generalization(
    family: dict[str, Any], unit_id: str, result: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    if result["result_type"] == "REJECT":
        return None, False
    if result["result_type"] == "FULL_RULE_FAMILY":
        semantic = copy.deepcopy(result["rule_family"])
        updated = _refresh_family_metadata(
            semantic, family_id=family["family_id"], previous=family
        )
        return updated, updated["base_unit"] != family["base_unit"]
    updated = copy.deepcopy(family)
    if result["action"] == "ADD_TO_EXISTING_GROUP":
        index = int(result["target_group_ref"][2:])
        updated["variant_groups"][index]["member_ids"].append(unit_id)
        updated["variant_groups"][index]["member_ids"].sort(
            key=llm_io.natural_sort_key
        )
    else:
        updated["variant_groups"].append(
            {"member_ids": [unit_id], "delta": copy.deepcopy(result["delta"])}
        )
    updated = _refresh_family_metadata(
        updated, family_id=family["family_id"], previous=family
    )
    return updated, False


def replace_family(catalog: dict[str, Any], updated_family: dict[str, Any]) -> None:
    family_id = updated_family["family_id"]
    for index, family in enumerate(catalog["rule_families"]):
        if family["family_id"] == family_id:
            catalog["rule_families"][index] = updated_family
            return
    raise ValueError(f"Cannot replace unknown family {family_id}.")


def append_family(catalog: dict[str, Any], family: dict[str, Any]) -> None:
    if family["family_id"] in family_by_id(catalog):
        raise ValueError(f"Family ID already exists: {family['family_id']}")
    catalog["rule_families"].append(family)


def update_family_state(
    catalog: dict[str, Any],
    *,
    family: dict[str, Any],
    unit_id: str,
    new_family: bool,
    base_changed: bool,
) -> None:
    states = family_state_map(catalog)
    family_id = family["family_id"]
    if new_family:
        state = {
            "family_id": family_id,
            "version": 0,
            "state_origin": "SERIAL_NO_MATCH_CREATED",
            "base_representation_origin": DYNAMIC_ORIGIN,
            "base_source_unit_id": unit_id,
        }
        catalog["family_states"].append(state)
        return
    state = states[family_id]
    state["version"] = int(state.get("version", 0)) + 1
    state["state_origin"] = "SERIAL_NO_MATCH_UPDATED"
    state["last_consolidateed_unit_id"] = unit_id
    if base_changed:
        state["base_representation_origin"] = DYNAMIC_ORIGIN
        state["base_source_unit_id"] = unit_id
    state["semantic_content_sha256"] = semantic_hash(family)


def next_new_family_id(catalog: dict[str, Any], serial_order: int) -> str:
    existing = set(family_by_id(catalog))
    candidate = f"NM{serial_order + 1:04d}"
    if candidate in existing:
        raise ValueError(f"Deterministic new family ID already exists: {candidate}")
    return candidate


def base_record_for_family(
    family: dict[str, Any], vector: list[float], *, model: str, dimensions: int
) -> dict[str, Any]:
    base = canonical_base(family)
    text = embedding_text_for_base(base)
    return {
        "family_id": family["family_id"],
        "base_unit": base,
        "base_unit_sha256": sha256_json(base),
        "embedding_text": text,
        "embedding_text_sha256": llm_io.sha256_text(text),
        "model": model,
        "dimensions": dimensions,
        "embedding_fields": ["violated_commonsense_rule", "situation"],
        "embedding": vector,
    }

