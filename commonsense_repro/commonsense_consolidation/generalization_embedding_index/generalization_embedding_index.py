"""Core planning and validation for incremental commonsense generalization record base embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from commonsense_repro.clustering.build_instance_level_commonsense_embeddings import build_embedding_text
from commonsense_repro.commonsense_consolidation.build_generalization_retrieval_index import (
    EMBEDDING_FIELDS,
    build_member_mapping,
)


EMBEDDING_TEMPLATE_VERSION = "combined_rule_situation_v1"


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    return value.strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_base(family: dict[str, Any]) -> dict[str, str]:
    family_id = family.get("family_id", "<unknown>")
    base = family.get("base_unit")
    if not isinstance(base, dict):
        raise ValueError(f"Family {family_id} has no base_unit object.")
    return {
        "situation": require_string(
            base.get("situation"), f"{family_id}.base_unit.situation"
        ),
        "commonsense_rule": require_string(
            base.get("commonsense_rule"),
            f"{family_id}.base_unit.commonsense_rule",
        ),
    }


def embedding_text_for_base(base: dict[str, str]) -> str:
    return build_embedding_text(
        {
            "violated_commonsense_rule": base["commonsense_rule"],
            "situation": base["situation"],
        },
        EMBEDDING_FIELDS,
    )


def catalog_bases(
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = catalog.get("rule_families")
    if not isinstance(families, list):
        raise ValueError("Catalog must contain a rule_families array.")
    state_versions = {
        state.get("family_id"): state.get("version")
        for state in catalog.get("family_states", [])
        if isinstance(state, dict)
    }
    bases: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise ValueError(f"rule_families[{index}] must be an object.")
        family_id = require_string(family.get("family_id"), f"family[{index}].family_id")
        if family_id in seen:
            raise ValueError(f"Duplicate family_id: {family_id}")
        seen.add(family_id)
        base = canonical_base(family)
        text = embedding_text_for_base(base)
        version = state_versions.get(family_id, 0)
        if not isinstance(version, int) or version < 0:
            raise ValueError(f"Invalid family version for {family_id}: {version}")
        bases.append(
            {
                "family_id": family_id,
                "family_version": version,
                "base_unit": base,
                "base_unit_sha256": sha256_json(base),
                "embedding_text": text,
                "embedding_text_sha256": sha256_text(text),
            }
        )
        mappings.extend(build_member_mapping(family))
    return bases, mappings


def validate_vector_record(
    record: dict[str, Any], *, model: str, dimensions: int, location: str
) -> None:
    if record.get("model") != model:
        raise ValueError(f"{location} uses model {record.get('model')!r}, expected {model!r}.")
    if record.get("dimensions") != dimensions:
        raise ValueError(f"{location} has unexpected dimensions.")
    if record.get("embedding_fields") != EMBEDDING_FIELDS:
        raise ValueError(f"{location} has unexpected embedding_fields.")
    vector = record.get("embedding")
    if not isinstance(vector, list) or len(vector) != dimensions:
        raise ValueError(f"{location} has an invalid embedding vector.")


@dataclass(frozen=True)
class SyncAction:
    family: dict[str, Any]
    action: str
    source_record: dict[str, Any] | None = None
    previous_base_unit_sha256: str | None = None


def plan_sync_actions(
    bases: list[dict[str, Any]],
    *,
    active_by_family: dict[str, dict[str, Any]],
    baseline_by_family: dict[str, dict[str, Any]],
    source_by_text: dict[str, dict[str, Any]],
) -> list[SyncAction]:
    """Plan updates without making API calls.

    An existing family's changed base always receives GENERATE_CHANGED_BASE.
    Exact source-vector reuse is allowed only for a genuinely new family.
    """
    actions: list[SyncAction] = []
    previous = active_by_family or baseline_by_family
    for base in bases:
        family_id = base["family_id"]
        old = previous.get(family_id)
        if old is not None:
            old_hash = old.get("base_unit_sha256")
            if old_hash == base["base_unit_sha256"]:
                actions.append(
                    SyncAction(base, "KEEP_EXISTING", old, old_hash)
                )
            else:
                actions.append(
                    SyncAction(base, "GENERATE_CHANGED_BASE", None, old_hash)
                )
            continue

        source = source_by_text.get(base["embedding_text"])
        if source is not None:
            actions.append(SyncAction(base, "REUSE_SOURCE_FOR_NEW_FAMILY", source))
        else:
            actions.append(SyncAction(base, "GENERATE_NEW_BASE"))
    return actions


def manifest_active_records(
    manifest: dict[str, Any], events_by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows = manifest.get("active_embeddings")
    if not isinstance(rows, list):
        raise ValueError("Manifest must contain active_embeddings.")
    active: dict[str, dict[str, Any]] = {}
    for index, pointer in enumerate(rows):
        if not isinstance(pointer, dict):
            raise ValueError(f"active_embeddings[{index}] must be an object.")
        family_id = require_string(pointer.get("family_id"), f"active[{index}].family_id")
        event_id = require_string(pointer.get("event_id"), f"active[{index}].event_id")
        event = events_by_id.get(event_id)
        if event is None:
            raise ValueError(f"Manifest references missing event {event_id}.")
        if event.get("family_id") != family_id:
            raise ValueError(f"Manifest pointer {event_id} has inconsistent family_id.")
        if pointer.get("base_unit_sha256") != event.get("base_unit_sha256"):
            raise ValueError(f"Manifest pointer {event_id} has inconsistent base hash.")
        if family_id in active:
            raise ValueError(f"Duplicate active family pointer: {family_id}")
        active[family_id] = event
    return active


