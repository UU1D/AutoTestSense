from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_PROMPTS = {
    "extract_instance_level_commonsense_v1.2.md": "{BUG_REPORT}",
    "generalize_local_commonsense_v1.4_system.md": None,
    "generalize_local_commonsense_v1.4_user.md": "{{units_json}}",
    "group_global_generalization_candidates_v1.0_system.md": None,
    "group_global_generalization_candidates_v1.0_user.md": "{{units_json}}",
    "generalize_global_commonsense_v1.2_system.md": None,
    "generalize_global_commonsense_v1.2_user.md": "{{input_json}}",
    "judge_unassigned_commonsense_consolidation_v1.2_system.md": None,
    "judge_unassigned_commonsense_consolidation_v1.2_user.md": "{{formatted_input}}",
    "consolidate_unassigned_commonsense_v1.0_system.md": None,
    "consolidate_unassigned_commonsense_v1.0_user.md": "{{input_json}}",
    "consolidate_batched_unassigned_commonsense_v1.1_system.md": None,
    "consolidate_batched_unassigned_commonsense_v1.1_user.md": "{{input_json}}",
}


EXPECTED_SCRIPTS = {
    "clustering/build_instance_level_commonsense_embeddings.py",
    "clustering/build_instance_level_commonsense_neighbors.py",
    "clustering/rerank_instance_level_commonsense_neighbors.py",
    "clustering/discover_snn_communities.py",
    "commonsense_generalization/local/generalize_local_commonsense.py",
    "commonsense_generalization/local/export_generalization_records.py",
    "commonsense_generalization/global/build_generalized_rule_embeddings.py",
    "commonsense_generalization/global/build_generalized_rule_neighbors.py",
    "commonsense_generalization/global/rerank_generalized_rule_neighbors.py",
    "commonsense_generalization/global/build_global_generalization_graph.py",
    "commonsense_generalization/global/candidate_grouping/group_global_generalization_candidates.py",
    "commonsense_generalization/global/generalization/generalize_global_commonsense.py",
    "commonsense_generalization/global/catalog_assembly/build_global_generalization_records.py",
    "commonsense_consolidation/build_generalization_retrieval_index.py",
    "commonsense_consolidation/recall_generalization_records.py",
    "commonsense_consolidation/rerank_consolidation_candidates.py",
    "commonsense_consolidation/consolidation_judgment/judge_consolidation_membership.py",
    "commonsense_consolidation/consolidation_judgment/analyze_consolidation_decisions.py",
    "commonsense_consolidation/input_preparation/prepare_unassigned_commonsense.py",
    "commonsense_consolidation/serial_consolidation/run_serial_consolidation.py",
    "commonsense_consolidation/batch_consolidation/build_batch_consolidation_plan.py",
    "commonsense_consolidation/batch_consolidation/run_batch_consolidation.py",
    "library_construction/build_library_situation_embeddings.py",
    "library_construction/copy_generalization_records.py",
    "retrieval/batch_retrieve_instance_level_commonsense.py",
    "retrieval/enrich_retrieved_instance_level_commonsense.py",
    "retrieval/batch_retrieve_commonsense_library.py",
    "retrieval/enrich_retrieved_commonsense_library.py",
}


def test_expected_prompt_names_and_placeholders() -> None:
    prompt_dir = ROOT / "prompts"
    actual = {path.name for path in prompt_dir.glob("*.md") if path.name != "README.md"}
    assert actual == set(EXPECTED_PROMPTS)
    for name, placeholder in EXPECTED_PROMPTS.items():
        text = (prompt_dir / name).read_text(encoding="utf-8")
        if placeholder is not None:
            assert placeholder in text, name


def test_expected_paper_terminology_script_names() -> None:
    code_root = ROOT / "commonsense_repro"
    for relative_path in EXPECTED_SCRIPTS:
        assert (code_root / relative_path).is_file(), relative_path


def test_old_module_names_and_user_facing_terms_are_absent() -> None:
    forbidden_paths = {
        "local_induction",
        "global_merge",
        "incremental_consolidateion",
        "final_library",
    }
    for path in (ROOT / "commonsense_repro").rglob("*"):
        assert path.name not in forbidden_paths, path

    checked_files = [ROOT / "README.md", ROOT / "reproduce.py"]
    checked_files.extend((ROOT / "prompts").glob("*.md"))
    checked_files.extend((ROOT / "commonsense_repro").rglob("*.py"))
    forbidden_text = (
        re.compile(r"\binduction\b", re.IGNORECASE),
        re.compile(r"\bconsolidateion\b", re.IGNORECASE),
        re.compile(r"\boriginal units?\b", re.IGNORECASE),
        re.compile(r"\brule famil(?:y|ies)\b", re.IGNORECASE),
    )
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_text:
            assert not pattern.search(text), f"{pattern.pattern} in {path}"


def test_readme_declares_all_seven_paper_stage_names() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = (
        "Instance-Level Commonsense Extraction",
        "Similarity and Community Discovery",
        "Initial Commonsense Generalization",
        "Global Commonsense Generalization",
        "Commonsense Consolidation",
        "Commonsense Library Assembly",
        "Commonsense Retrieval",
    )
    for title in expected:
        assert title in text


def test_readmes_do_not_describe_old_schema_migrations() -> None:
    forbidden = (
        "backward compatible",
        "backward compatibility",
        "legacy output",
        "legacy json",
        "schema compatibility",
        "kept unchanged",
        "retained for compatibility",
    )
    for path in ROOT.rglob("README*.md"):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase not in text, f"{phrase!r} in {path}"
