"""Unified entry point for the seven-stage reproduction package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
load_dotenv(ROOT / ".env")

SOURCES = ("extract", "github", "negative")
UNIT_FILENAMES = {
    "extract": "gemini_v1_2_extract.json",
    "github": "gemini_v1_2_github.json",
    "negative": "gemini_v1_2_negative.json",
}
MODEL_ENV_KEYS = (
    "OUTER_MODEL",
    "EMBEDDING_MODEL",
    "RERANK_MODEL",
    "DEEPSEEK_V4_FLASH",
)

REPRODUCTION_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "expected": {
        "manifest_ids": 4634,
        "manifest_unique_urls": 4594,
        "instance_level_commonsense": 3708,
        "final_generalization_records": 1399,
        "final_covered_instance_level_commonsense": 3708,
        "embedding_dimensions": 1024,
        "prompt_files": 13,
        "reference_sha256": {
            "final_commonsense_generalization_records.json": (
                "5079fe3129d8581eb1deaa7a49923c799d183e16233fca238d17b76bd1503c7a"
            ),
            "generalized_rule_situation_embeddings.jsonl": (
                "733b9607a9c96067ba3f8cd474e30fc720cab85ece011f2c32b6bf8799b094f5"
            ),
            "commonsense_library_situation_embeddings.jsonl": (
                "ff2bd074d60dac8509bbca9d2f2b25138ef5752ec7ebf63d5d43bde7507c0251"
            ),
        },
    },
    "stage2": {
        "embedding_fields": ["violated_commonsense_rule", "situation"],
        "embedding_batch_size": 10,
        "recall_k": 60,
        "rerank_k": 30,
        "min_shared_neighbors": 15,
        "mutual_only": False,
        "louvain_resolution": 0.8,
        "max_community_size": 25,
        "leiden_resolution": 1.0,
        "random_seed": 42,
    },
    "stage4": {
        "recall_k": 20,
        "rerank_k": 10,
        "situation_weight": 0.35,
        "rule_weight": 0.65,
        "graph_top_k": 3,
        "graph_min_score": 0.7,
        "mutual_only": True,
    },
    "stage5": {
        "recall_k": 20,
        "rerank_k": 10,
        "min_rerank_score": 0.7,
        "min_candidates": 3,
        "reuse_initial_no_match": False,
    },
    "stage7": {
        "recall_k": 60,
        "rerank_k": 40,
        "enrich_top_k": 30,
    },
}

STAGE_NAMES = {
    1: "Instance-Level Commonsense Extraction",
    2: "Similarity and Community Discovery",
    3: "Initial Commonsense Generalization",
    4: "Global Commonsense Generalization",
    5: "Commonsense Consolidation",
    6: "Commonsense Library Assembly",
    7: "Commonsense Retrieval",
}


@dataclass
class Step:
    name: str
    module: str
    arguments: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.arguments]


def path_text(path: Path) -> str:
    return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_sha256(config: dict[str, Any]) -> str:
    content = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def command_input_files(arguments: list[str]) -> list[Path]:
    """Return existing files referenced by a step before that step runs."""
    files: list[Path] = []
    seen: set[Path] = set()
    for value in arguments:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in seen:
            files.append(resolved)
            seen.add(resolved)
    return files


def collect_stage_summaries(stage: int) -> list[dict[str, Any]]:
    root = WORK / f"stage{stage}"
    if not root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*summary*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            summaries.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "content": value,
                }
            )
    return summaries


def aggregate_usage(value: Any, totals: dict[str, int]) -> None:
    """Collect token/request counters from summary objects without assuming one SDK."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.casefold()
            if isinstance(item, int) and (
                "token" in normalized or normalized in {"requests", "request_count"}
            ):
                totals[key] = totals.get(key, 0) + item
            else:
                aggregate_usage(item, totals)
    elif isinstance(value, list):
        for item in value:
            aggregate_usage(item, totals)


def module_step(name: str, module: str, *arguments: Any, outputs: list[Path] | None = None) -> Step:
    return Step(
        name=name,
        module=module,
        arguments=[str(value) for value in arguments],
        outputs=outputs or [],
    )


def common_suffix(args: argparse.Namespace, *, allow_workers: bool = False) -> list[str]:
    suffix: list[str] = []
    if args.limit is not None:
        suffix += ["--limit", str(args.limit)]
    if allow_workers:
        suffix += ["--workers", str(args.workers)]
    if args.overwrite:
        suffix.append("--overwrite")
    return suffix


def source_unit_files(prefer_work: bool = True) -> list[Path]:
    work_files = [WORK / "stage1" / "units" / "positive" / UNIT_FILENAMES[name] for name in SOURCES]
    if prefer_work and all(path.is_file() for path in work_files):
        return work_files
    return [DATA / "checkpoints" / "instance_level_commonsense" / UNIT_FILENAMES[name] for name in SOURCES]


def source_embedding_files(directory: Path, *, situation_only: bool = False) -> list[Path]:
    suffix = "_situation_embeddings.jsonl" if situation_only else "_embeddings.jsonl"
    return [directory / f"gemini_v1_2_{name}{suffix}" for name in SOURCES]


def repeat_option(name: str, values: list[Path]) -> list[str]:
    output: list[str] = []
    for value in values:
        output += [name, path_text(value)]
    return output


def list_option(name: str, values: list[Path]) -> list[str]:
    return [name, *[path_text(value) for value in values]]


def stage1_steps(args: argparse.Namespace, config: dict[str, Any]) -> list[Step]:
    root = WORK / "stage1"
    steps = [
        module_step(
            "fetch fixed issue manifest",
            "commonsense_repro.extraction.fetch_issue_manifest",
            "--manifest", path_text(DATA / "input" / "issue_manifest.jsonl"),
            "--output-dir", path_text(root / "reports"),
            "--cache-dir", path_text(root / "issue_cache"),
            *common_suffix(args),
        )
    ]
    for source in SOURCES:
        prompts = root / "prompts" / source
        output_json = root / "llm_outputs" / source / "json"
        output_md = root / "llm_outputs" / source / "md"
        raw_units = root / "units" / "raw" / UNIT_FILENAMES[source]
        positive_units = root / "units" / "positive" / UNIT_FILENAMES[source]
        overwrite = ["--overwrite"] if args.overwrite else []
        limit = ["--limit", str(args.limit)] if args.limit is not None else []
        steps.extend(
            [
                module_step(
                    f"build {source} extraction prompts",
                    "commonsense_repro.extraction.batch_build_llm_prompts",
                    "--input-dir", path_text(root / "reports" / source),
                    "--template", path_text(PROMPTS / "extract_instance_level_commonsense_v1.2.md"),
                    "--output-dir", path_text(prompts),
                    "--log-file", path_text(root / "logs" / f"build_{source}.log"),
                    *limit,
                    *overwrite,
                ),
                module_step(
                    f"run {source} extraction model",
                    "commonsense_repro.extraction.batch_run_llm_inference",
                    "--input-dir", path_text(prompts),
                    "--output-json-dir", path_text(output_json),
                    "--output-md-dir", path_text(output_md),
                    "--provider", "outer",
                    "--workers", str(args.workers),
                    "--log-file", path_text(root / "logs" / f"infer_{source}.log"),
                    *limit,
                    *overwrite,
                ),
                module_step(
                    f"parse {source} model JSON",
                    "commonsense_repro.extraction.extract_json_from_md",
                    path_text(output_md),
                    "--output-file", path_text(raw_units),
                    *overwrite,
                    outputs=[raw_units],
                ),
                module_step(
                    f"filter positive {source} units",
                    "commonsense_repro.extraction.filter_instance_level_commonsense",
                    "--input-file", path_text(raw_units),
                    "--output-file", path_text(positive_units),
                    *overwrite,
                    outputs=[positive_units],
                ),
            ]
        )
    return steps


def stage2_steps(args: argparse.Namespace, config: dict[str, Any]) -> list[Step]:
    cfg = config["stage2"]
    root = WORK / "stage2"
    units = source_unit_files()
    embeddings = source_embedding_files(root / "embeddings")
    steps: list[Step] = []
    for source, input_file, output_file in zip(SOURCES, units, embeddings):
        summary = output_file.with_name(output_file.stem + "_summary.json")
        suffix = common_suffix(args)
        steps.append(
            module_step(
                f"embed {source} units",
                "commonsense_repro.clustering.build_instance_level_commonsense_embeddings",
                "--input-file", path_text(input_file),
                "--output-file", path_text(output_file),
                "--summary-file", path_text(summary),
                "--fields", *cfg["embedding_fields"],
                "--batch-size", cfg["embedding_batch_size"],
                *suffix,
            )
        )
    if args.limit is not None and not args.dry_run:
        return steps
    neighbors = root / "neighbors"
    rerank = root / "rerank"
    snn = root / "snn"
    refined = root / "refined"
    overwrite = ["--overwrite"] if args.overwrite else []
    mutual_option = ["--mutual-only" if cfg["mutual_only"] else "--no-mutual-only"]
    steps.extend(
        [
            module_step(
                "build cosine neighbors",
                "commonsense_repro.clustering.build_instance_level_commonsense_neighbors",
                *list_option("--embeddings-files", embeddings),
                "--output-dir", path_text(neighbors),
                "--k-values", cfg["rerank_k"], cfg["recall_k"],
                *overwrite,
                outputs=[neighbors / f"embedding_neighbors_k{cfg['recall_k']}.jsonl"],
            ),
            module_step(
                "rerank unit neighbors",
                "commonsense_repro.clustering.rerank_instance_level_commonsense_neighbors",
                "--neighbors-file", path_text(neighbors / f"embedding_neighbors_k{cfg['recall_k']}.jsonl"),
                *list_option("--metadata-files", units),
                "--output-dir", path_text(rerank),
                "--final-k", cfg["rerank_k"],
                *overwrite,
                outputs=[rerank / "reranked_neighbors.jsonl"],
            ),
            module_step(
                "construct SNN and Louvain communities",
                "commonsense_repro.clustering.discover_snn_communities",
                "--neighbors-file", path_text(rerank / "reranked_neighbors.jsonl"),
                *list_option("--metadata-files", units),
                "--output-dir", path_text(snn),
                "--min-shared-neighbors", cfg["min_shared_neighbors"],
                *mutual_option,
                "--resolution", cfg["louvain_resolution"],
                "--random-seed", cfg["random_seed"],
                *overwrite,
                outputs=[snn / "snn_communities.json"],
            ),
            module_step(
                "refine large communities with Leiden",
                "commonsense_repro.clustering.refine_snn_with_leiden",
                "--input-dir", path_text(snn),
                *list_option("--metadata-files", units),
                "--output-dir", path_text(refined),
                "--max-community-size", cfg["max_community_size"],
                "--resolution", cfg["leiden_resolution"],
                "--random-seed", cfg["random_seed"],
                *overwrite,
                outputs=[refined / "llm_batches.jsonl", refined / "singletons.jsonl"],
            ),
        ]
    )
    return steps


def stage3_steps(args: argparse.Namespace, config: dict[str, Any]) -> list[Step]:
    root = WORK / "stage3"
    generalization = root / "generalization"
    catalog = root / "commonsense_generalization_records.json"
    units = source_unit_files()
    overwrite = ["--overwrite"] if args.overwrite else []
    return [
        module_step(
            "generalize local commonsense",
            "commonsense_repro.commonsense_generalization.local.generalize_local_commonsense",
            "--batches-file", path_text(WORK / "stage2" / "refined" / "llm_batches.jsonl"),
            *list_option("--source-files", units),
            "--system-prompt-file", path_text(PROMPTS / "generalize_local_commonsense_v1.4_system.md"),
            "--user-prompt-file", path_text(PROMPTS / "generalize_local_commonsense_v1.4_user.md"),
            "--output-dir", path_text(generalization),
            *common_suffix(args, allow_workers=True),
        ),
        module_step(
            "export local commonsense generalization records",
            "commonsense_repro.commonsense_generalization.local.export_generalization_records",
            "--results-file", path_text(generalization / "commonsense_generalization_results.jsonl"),
            "--output-file", path_text(catalog),
            outputs=[catalog],
        ),
    ]


def stage4_steps(args: argparse.Namespace, config: dict[str, Any]) -> list[Step]:
    cfg = config["stage4"]
    root = WORK / "stage4"
    catalog = WORK / "stage3" / "commonsense_generalization_records.json"
    embeddings = root / "embeddings" / "generalized_rule_embeddings.jsonl"
    recall = root / "recall"
    rerank = root / "rerank"
    graph = root / "candidate_graph"
    grouping = root / "candidate_grouping"
    generalization = root / "generalization"
    catalog_assembly = root / "catalog_assembly"
    units = source_unit_files()
    overwrite = ["--overwrite"] if args.overwrite else []
    mutual_option = [] if cfg["mutual_only"] else ["--non-mutual"]
    steps = [
        module_step(
            "embed local generalized commonsense rules",
            "commonsense_repro.commonsense_generalization.global.build_generalized_rule_embeddings",
            "--input-file", path_text(catalog),
            "--output-file", path_text(embeddings),
            "--summary-file", path_text(root / "embeddings" / "generalized_rule_embeddings_summary.json"),
            *overwrite,
        ),
        module_step(
            "recall similar generalized commonsense rules",
            "commonsense_repro.commonsense_generalization.global.build_generalized_rule_neighbors",
            "--input-file", path_text(embeddings),
            "--output-dir", path_text(recall),
            "--top-k", cfg["recall_k"],
            "--situation-weight", cfg["situation_weight"],
            "--rule-weight", cfg["rule_weight"],
        ),
        module_step(
            "rerank generalized-rule neighbors",
            "commonsense_repro.commonsense_generalization.global.rerank_generalized_rule_neighbors",
            "--neighbors-file", path_text(recall / f"generalized_rule_neighbors_k{cfg['recall_k']}.jsonl"),
            "--catalog-file", path_text(catalog),
            "--output-dir", path_text(rerank),
            "--final-k", cfg["rerank_k"],
            "--workers", args.workers,
            *overwrite,
        ),
        module_step(
            "build mutual candidate graph",
            "commonsense_repro.commonsense_generalization.global.build_global_generalization_graph",
            "--catalog", path_text(catalog),
            "--rerank", path_text(rerank / f"reranked_neighbors_k{cfg['rerank_k']}.jsonl"),
            "--output-dir", path_text(graph),
            "--top-k", cfg["graph_top_k"],
            "--min-score", cfg["graph_min_score"],
            *mutual_option,
        ),
        module_step(
            "group global generalization candidates",
            "commonsense_repro.commonsense_generalization.global.candidate_grouping.group_global_generalization_candidates",
            "--batches-file", path_text(graph / "llm_batches.jsonl"),
            "--system-prompt-file", path_text(PROMPTS / "group_global_generalization_candidates_v1.0_system.md"),
            "--user-prompt-file", path_text(PROMPTS / "group_global_generalization_candidates_v1.0_user.md"),
            "--output-dir", path_text(grouping),
            *common_suffix(args, allow_workers=True),
        ),
        module_step(
            "generalize commonsense globally",
            "commonsense_repro.commonsense_generalization.global.generalization.generalize_global_commonsense",
            "--stage1-results-file", path_text(grouping / "grouping_results.jsonl"),
            "--catalog-file", path_text(catalog),
            *list_option("--source-files", units),
            "--system-prompt-file", path_text(PROMPTS / "generalize_global_commonsense_v1.2_system.md"),
            "--user-prompt-file", path_text(PROMPTS / "generalize_global_commonsense_v1.2_user.md"),
            "--output-dir", path_text(generalization),
            *common_suffix(args, allow_workers=True),
        ),
    ]
    if args.limit is not None and not args.dry_run:
        return steps
    steps.append(
        module_step(
            "assemble global commonsense generalization records",
            "commonsense_repro.commonsense_generalization.global.catalog_assembly.build_global_generalization_records",
            "--catalog-file", path_text(catalog),
            "--unmatched-file", path_text(graph / "unmatched_generalization_records.json"),
            "--stage1-results-file", path_text(grouping / "grouping_results.jsonl"),
            "--stage2-results-file", path_text(generalization / "final_generalization_results.jsonl"),
            *repeat_option("--source-file", units),
            "--output-dir", path_text(catalog_assembly),
            outputs=[catalog_assembly / "final_commonsense_generalization_records.json"],
        )
    )
    return steps


def stage5_steps(args: argparse.Namespace, config: dict[str, Any]) -> list[Step]:
    cfg = config["stage5"]
    root = WORK / "stage5"
    units = source_unit_files()
    source_embeddings = source_embedding_files(WORK / "stage2" / "embeddings")
    serial_inputs = root / "serial_inputs"
    generalization_index = root / "initial_generalization_index"
    result = root / "serial_consolidation"
    overwrite = ["--overwrite"] if args.overwrite else []
    reuse_option = [
        "--reuse-initial-no-match"
        if cfg["reuse_initial_no_match"]
        else "--no-reuse-initial-no-match"
    ]
    return [
        module_step(
            "prepare unassigned instance-level commonsense",
            "commonsense_repro.commonsense_consolidation.input_preparation.prepare_unassigned_commonsense",
            "--base-catalog", path_text(WORK / "stage4" / "catalog_assembly" / "final_commonsense_generalization_records.json"),
            *repeat_option("--source-file", units),
            "--output-dir", path_text(serial_inputs),
        ),
        module_step(
            "build initial generalization retrieval index",
            "commonsense_repro.commonsense_consolidation.build_generalization_retrieval_index",
            "--catalog-file", path_text(serial_inputs / "initial_commonsense_generalization_records.json"),
            *repeat_option("--source-embedding-file", source_embeddings),
            "--output-dir", path_text(generalization_index),
            *overwrite,
        ),
        module_step(
            "serially consolidate unassigned instance-level commonsense",
            "commonsense_repro.commonsense_consolidation.serial_consolidation.run_serial_consolidation",
            "--initial-catalog", path_text(serial_inputs / "initial_commonsense_generalization_records.json"),
            "--queue-file", path_text(serial_inputs / "no_match_units.jsonl"),
            "--initial-embedding-dir", path_text(generalization_index),
            *repeat_option("--source-file", units),
            *repeat_option("--source-embedding-file", source_embeddings),
            "--membership-system-prompt", path_text(PROMPTS / "judge_unassigned_commonsense_consolidation_v1.2_system.md"),
            "--membership-user-prompt", path_text(PROMPTS / "judge_unassigned_commonsense_consolidation_v1.2_user.md"),
            "--generalization-system-prompt", path_text(PROMPTS / "consolidate_unassigned_commonsense_v1.0_system.md"),
            "--generalization-user-prompt", path_text(PROMPTS / "consolidate_unassigned_commonsense_v1.0_user.md"),
            "--output-dir", path_text(result),
            "--recall-k", cfg["recall_k"],
            "--final-k", cfg["rerank_k"],
            "--min-rerank-score", cfg["min_rerank_score"],
            "--min-candidates", cfg["min_candidates"],
            *reuse_option,
            *common_suffix(args),
        ),
    ]


def stage6_steps(args: argparse.Namespace, config: dict[str, Any]) -> list[Step]:
    root = WORK / "stage6"
    situation_embeddings = source_embedding_files(
        DATA / "reference" / "instance_level_commonsense_situation_embeddings"
    )
    catalog = WORK / "stage5" / "serial_consolidation" / "current_commonsense_generalization_records.json"
    if not catalog.is_file():
        catalog = DATA / "reference" / "commonsense_library" / "final_commonsense_generalization_records.json"
    steps: list[Step] = []
    final_dir = root / "commonsense_library"
    steps.append(
        module_step(
            "copy final commonsense generalization records",
            "commonsense_repro.library_construction.copy_generalization_records",
            "--input-file", path_text(catalog),
            "--output-file", path_text(final_dir / "final_commonsense_generalization_records.json"),
            outputs=[final_dir / "final_commonsense_generalization_records.json"],
        )
    )
    steps.append(
        module_step(
            "build commonsense library situation embeddings",
            "commonsense_repro.library_construction.build_library_situation_embeddings",
            "--input-file", path_text(catalog),
            *repeat_option("--source-embedding-file", situation_embeddings),
            "--base-output-file", path_text(final_dir / "generalized_rule_situation_embeddings.jsonl"),
            "--output-file", path_text(final_dir / "commonsense_library_situation_embeddings.jsonl"),
            "--summary-file", path_text(final_dir / "commonsense_library_situation_embeddings_summary.json"),
            *common_suffix(args),
        )
    )
    return steps


def stage7_steps(args: argparse.Namespace, config: dict[str, Any]) -> list[Step]:
    cfg = config["stage7"]
    situation_tracks = {
        "visiondroid": ROOT / "rqs" / "rq2" / "data" / "situations" / "sequence_context",
        "vanilla_mllm": ROOT / "rqs" / "rq2" / "data" / "situations" / "image_only",
    }
    dataset = situation_tracks[args.dataset]
    input_dir = dataset / "extracted"
    root = WORK / "stage7" / args.dataset
    final_dir = WORK / "stage6" / "commonsense_library"
    embeddings = final_dir / "commonsense_library_situation_embeddings.jsonl"
    catalog = WORK / "stage5" / "serial_consolidation" / "current_commonsense_generalization_records.json"
    if not embeddings.is_file():
        embeddings = DATA / "reference" / "commonsense_library" / "commonsense_library_situation_embeddings.jsonl"
    if not catalog.is_file():
        catalog = DATA / "reference" / "commonsense_library" / "final_commonsense_generalization_records.json"
    units = source_unit_files()
    instance_embeddings = source_embedding_files(
        DATA / "reference" / "instance_level_commonsense_situation_embeddings"
    )
    overwrite = ["--overwrite"] if args.overwrite else []
    limit = ["--limit", str(args.limit)] if args.limit is not None else []
    steps: list[Step] = []
    if args.mode in ("library", "both"):
        raw = root / "library" / "retrieval_output"
        enriched = root / "library" / "retrieval_output_with_commonsense"
        steps.extend(
            [
                module_step(
                    "retrieve from the commonsense library",
                    "commonsense_repro.retrieval.batch_retrieve_commonsense_library",
                    "--dataset-dir", path_text(dataset),
                    "--input-dir", path_text(input_dir),
                    "--output-dir", path_text(raw),
                    "--embeddings-file", path_text(embeddings),
                    "--recall-k", cfg["recall_k"],
                    "--rerank-top-k", cfg["rerank_k"],
                    *limit,
                    *overwrite,
                ),
                module_step(
                    "enrich retrieved commonsense generalization records",
                    "commonsense_repro.retrieval.enrich_retrieved_commonsense_library",
                    "--dataset-dir", path_text(dataset),
                    "--input-dir", path_text(raw),
                    "--output-dir", path_text(enriched),
                    "--catalog-file", path_text(catalog),
                    *list_option("--source-files", units),
                    "--top-k", cfg["enrich_top_k"],
                ),
            ]
        )
    if args.mode in ("instance_level", "both"):
        raw = root / "instance_level" / "retrieval_output"
        enriched = root / "instance_level" / "retrieval_output_with_commonsense"
        steps.extend(
            [
                module_step(
                    "retrieve instance-level commonsense",
                    "commonsense_repro.retrieval.batch_retrieve_instance_level_commonsense",
                    "--dataset-dir", path_text(dataset),
                    "--input-dir", path_text(input_dir),
                    "--output-dir", path_text(raw),
                    *list_option("--embeddings-files", instance_embeddings),
                    *list_option("--metadata-files", units),
                    "--recall-k", cfg["recall_k"],
                    "--rerank-top-k", cfg["rerank_k"],
                    *limit,
                    *overwrite,
                ),
                module_step(
                    "enrich retrieved instance-level commonsense",
                    "commonsense_repro.retrieval.enrich_retrieved_instance_level_commonsense",
                    "--dataset-dir", path_text(dataset),
                    "--input-dir", path_text(raw),
                    "--output-dir", path_text(enriched),
                    *list_option("--source-files", units),
                    "--top-k", cfg["enrich_top_k"],
                ),
            ]
        )
    return steps


STAGE_BUILDERS = {
    1: stage1_steps,
    2: stage2_steps,
    3: stage3_steps,
    4: stage4_steps,
    5: stage5_steps,
    6: stage6_steps,
    7: stage7_steps,
}


def run_steps(
    stage: int,
    steps: list[Step],
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    started = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    status = "success"
    hash_cache: dict[Path, str] = {}
    try:
        for step in steps:
            command = step.command()
            inputs = []
            for path in command_input_files(step.arguments):
                if path not in hash_cache:
                    hash_cache[path] = sha256_file(path)
                inputs.append({"path": str(path), "sha256": hash_cache[path]})
            if step.outputs and all(path.exists() for path in step.outputs) and not args.overwrite:
                print(f"SKIP {step.name}: outputs already exist")
                records.append(
                    {"name": step.name, "status": "skipped", "command": command, "inputs": inputs}
                )
                continue
            print(f"RUN  {step.name}")
            print("     " + subprocess.list2cmdline(command))
            if args.dry_run:
                records.append(
                    {"name": step.name, "status": "dry_run", "command": command, "inputs": inputs}
                )
                continue
            step_started = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            elapsed = time.monotonic() - step_started
            records.append(
                {
                    "name": step.name,
                    "status": "success" if completed.returncode == 0 else "failed",
                    "return_code": completed.returncode,
                    "elapsed_seconds": round(elapsed, 3),
                    "command": command,
                    "inputs": inputs,
                }
            )
            if completed.returncode != 0:
                status = "failed"
                raise RuntimeError(f"Step failed: {step.name}")
    finally:
        if not args.dry_run:
            manifest_dir = WORK / "run_manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            ended = datetime.now(timezone.utc)
            summaries = collect_stage_summaries(stage)
            api_usage: dict[str, int] = {}
            for summary in summaries:
                aggregate_usage(summary["content"], api_usage)
            manifest = {
                "stage": stage,
                "stage_name": STAGE_NAMES[stage],
                "status": status,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "elapsed_seconds": round((ended - started).total_seconds(), 3),
                "reproduction_configuration": config,
                "configuration_sha256": configuration_sha256(config),
                "parameters": {
                    "limit": args.limit,
                    "workers": args.workers,
                    "overwrite": args.overwrite,
                    "dataset": getattr(args, "dataset", None),
                    "mode": getattr(args, "mode", None),
                },
                "models": {key: os.environ.get(key) for key in MODEL_ENV_KEYS},
                "steps": records,
                "output_summaries": summaries,
                "reported_api_usage": api_usage,
            }
            stamp = started.strftime("%Y%m%dT%H%M%SZ")
            path = manifest_dir / f"{stamp}_stage{stage}.json"
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Non-object JSONL row in {path}")
                rows.append(value)
    return rows


def doctor(config: dict[str, Any]) -> int:
    expected = config["expected"]
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0]))
    for module in ("numpy", "networkx", "dotenv", "httpx", "openai", "igraph", "leidenalg"):
        checks.append((f"dependency {module}", importlib.util.find_spec(module) is not None, "installed"))

    manifest = read_jsonl(DATA / "input" / "issue_manifest.jsonl")
    manifest_ids = [row.get("id") for row in manifest]
    manifest_urls = [row.get("url") for row in manifest]
    checks.append(("issue manifest rows", len(manifest) == expected["manifest_ids"], str(len(manifest))))
    checks.append(("unique issue IDs", len(set(manifest_ids)) == len(manifest), str(len(set(manifest_ids)))))
    checks.append(("unique issue URLs", len(set(manifest_urls)) == expected["manifest_unique_urls"], str(len(set(manifest_urls)))))

    units: list[dict[str, Any]] = []
    for path in source_unit_files(prefer_work=False):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Expected array: {path}")
        units.extend(value)
    unit_ids = [row.get("id") for row in units]
    checks.append(("checkpoint unit count", len(units) == expected["instance_level_commonsense"], str(len(units))))
    checks.append(("checkpoint IDs unique", len(set(unit_ids)) == len(units), str(len(set(unit_ids)))))

    catalog_path = DATA / "reference" / "commonsense_library" / "final_commonsense_generalization_records.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    families = catalog.get("rule_families", [])
    member_ids = [member for family in families for member in family.get("member_ids", [])]
    checks.append(("reference generalization record count", len(families) == expected["final_generalization_records"], str(len(families))))
    checks.append(("reference covered instance-level commonsense", len(member_ids) == expected["final_covered_instance_level_commonsense"], str(len(member_ids))))
    checks.append(("reference member uniqueness", len(set(member_ids)) == len(member_ids), str(len(set(member_ids)))))

    summary = json.loads((DATA / "reference" / "commonsense_library" / "commonsense_library_situation_embeddings_summary.json").read_text(encoding="utf-8"))
    checks.append(("embedding dimensions", summary.get("dimensions") == expected["embedding_dimensions"], str(summary.get("dimensions"))))
    reference_dir = DATA / "reference" / "commonsense_library"
    for filename, expected_hash in expected["reference_sha256"].items():
        actual_hash = sha256_file(reference_dir / filename)
        checks.append(
            (
                f"reference hash {filename}",
                actual_hash == expected_hash,
                actual_hash,
            )
        )
    prompt_count = len([path for path in PROMPTS.glob("*.md") if path.name != "README.md"])
    checks.append(("prompt set", prompt_count == expected["prompt_files"], str(prompt_count)))

    failures = 0
    for name, passed, detail in checks:
        print(f"{'OK  ' if passed else 'FAIL'} {name}: {detail}")
        failures += int(not passed)
    env_path = ROOT / ".env"
    print(f"INFO .env: {'present' if env_path.is_file() else 'not configured; copy .env.example'}")
    print(f"Doctor completed: checks={len(checks)}, failures={failures}")
    return 1 if failures else 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="Validate package and reference data.")
    for stage in range(1, 7):
        stage_parser = subparsers.add_parser(
            f"stage{stage}", help=STAGE_NAMES[stage]
        )
        add_common_arguments(stage_parser)
    stage7_parser = subparsers.add_parser("stage7", help=STAGE_NAMES[7])
    add_common_arguments(stage7_parser)
    stage7_parser.add_argument("--dataset", choices=("visiondroid", "vanilla_mllm"), default="visiondroid")
    stage7_parser.add_argument("--mode", choices=("library", "instance_level", "both"), default="both")
    all_parser = subparsers.add_parser("all")
    add_common_arguments(all_parser)
    all_parser.add_argument("--start-stage", type=int, choices=range(1, 8), default=1)
    all_parser.add_argument("--end-stage", type=int, choices=range(1, 8), default=7)
    all_parser.add_argument("--dataset", choices=("visiondroid", "vanilla_mllm"), default="visiondroid")
    all_parser.add_argument("--mode", choices=("library", "instance_level", "both"), default="both")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = REPRODUCTION_CONFIG
    if args.command == "doctor":
        raise SystemExit(doctor(config))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.command == "all":
        if args.start_stage > args.end_stage:
            parser.error("--start-stage cannot exceed --end-stage")
        stages = range(args.start_stage, args.end_stage + 1)
    else:
        stages = [int(args.command.removeprefix("stage"))]
    for stage in stages:
        steps = STAGE_BUILDERS[stage](args, config)
        run_steps(stage, steps, args, config)


if __name__ == "__main__":
    main()
