from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_reference_counts_and_coverage() -> None:
    unit_ids: list[str] = []
    for path in sorted((ROOT / "data/checkpoints/instance_level_commonsense").glob("*.json")):
        unit_ids.extend(row["id"] for row in json.loads(path.read_text(encoding="utf-8")))

    catalog = json.loads(
        (ROOT / "data/reference/commonsense_library/final_commonsense_generalization_records.json")
        .read_text(encoding="utf-8")
    )
    families = catalog["rule_families"]
    member_ids = [member for family in families for member in family["member_ids"]]

    assert len(unit_ids) == 3708
    assert len(set(unit_ids)) == 3708
    assert len(families) == 1399
    assert len(member_ids) == 3708
    assert set(member_ids) == set(unit_ids)


def test_rq1_sample_index_resolves_to_packaged_data() -> None:
    index = json.loads(
        (ROOT / "rqs/rq1/rq1_sample_index.json").read_text(encoding="utf-8")
    )
    records = index["sampled_records"]
    sampled_member_ids = [
        member_id for record in records for member_id in record["member_ids"]
    ]

    catalog = json.loads(
        (ROOT / index["data_locations"]["generalization_records"])
        .read_text(encoding="utf-8")
    )
    catalog_members = {
        family["family_id"]: set(family["member_ids"])
        for family in catalog["rule_families"]
    }

    available_unit_ids: set[str] = set()
    for relative_path in index["data_locations"]["instance_level_commonsense"]:
        rows = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        available_unit_ids.update(row["id"] for row in rows)

    assert index["sample"]["population_generalization_record_count"] == 1399
    assert index["sample"]["sampled_generalization_record_count"] == 302
    assert index["sample"]["sampled_instance_level_commonsense_count"] == 848
    assert len(records) == 302
    assert len({record["family_id"] for record in records}) == 302
    assert len(sampled_member_ids) == 848
    assert len(set(sampled_member_ids)) == 848
    assert set(sampled_member_ids) <= available_unit_ids
    for record in records:
        assert set(record["member_ids"]) == catalog_members[record["family_id"]]


def test_issue_manifest_keeps_all_ids_before_url_deduplication() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/input/issue_manifest.jsonl")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 4634
    assert len({row["id"] for row in rows}) == 4634
    assert len({row["url"] for row in rows}) == 4594
    assert sum(row["source"] == "extract" for row in rows) == 972


def test_package_has_no_source_tree_imports() -> None:
    for path in (ROOT / "commonsense_repro").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from src." not in source, path
        assert "import src." not in source, path


def test_prompt_filenames_are_ascii_and_documented() -> None:
    prompt_dir = ROOT / "prompts"
    prompt_files = sorted(path for path in prompt_dir.glob("*.md") if path.name != "README.md")
    documentation = (prompt_dir / "README.md").read_text(encoding="utf-8")
    assert len(prompt_files) == 13
    for path in prompt_files:
        assert path.name.isascii()
        assert path.name in documentation


def test_reference_artifact_names_counts_and_hashes() -> None:
    reference_dir = ROOT / "data/reference/commonsense_library"
    expected = {
        "final_commonsense_generalization_records.json": (
            "5079fe3129d8581eb1deaa7a49923c799d183e16233fca238d17b76bd1503c7a"
        ),
        "generalized_rule_situation_embeddings.jsonl": (
            "733b9607a9c96067ba3f8cd474e30fc720cab85ece011f2c32b6bf8799b094f5"
        ),
        "commonsense_library_situation_embeddings.jsonl": (
            "ff2bd074d60dac8509bbca9d2f2b25138ef5752ec7ebf63d5d43bde7507c0251"
        ),
    }
    assert {path.name for path in reference_dir.iterdir()} == {
        *expected,
        "commonsense_library_situation_embeddings_summary.json",
    }
    for filename, digest in expected.items():
        assert sha256_file(reference_dir / filename) == digest

    with (reference_dir / "generalized_rule_situation_embeddings.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        base_count = sum(1 for line in file if line.strip())
    with (reference_dir / "commonsense_library_situation_embeddings.jsonl").open(
        "r", encoding="utf-8"
    ) as file:
        complete_count = sum(1 for line in file if line.strip())
    assert base_count == 1399
    assert complete_count == 3547
