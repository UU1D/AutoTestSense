"""Run KuiTest with flat instance-level commonsense retrieval."""

from __future__ import annotations

import os

import run_commonsense as shared
from commonsense_mapping import (
    build_case_mapping as build_family_case_mapping,
    mapping_fingerprint,
)
from rq4_instance_level_context import normalise_as_singleton_families


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITUATIONS_DIR = os.path.join(BASE_DIR, "situations")
MAPPING_DIR = os.path.join(BASE_DIR, "commonsense_mapping_cache")
RESPONSE_CACHE_DIR = os.path.join(BASE_DIR, "response_cache_with_commonsense")
BUG_DIR = os.path.join(BASE_DIR, "bug_with_commonsense")
BUGFREE_DIR = os.path.join(BASE_DIR, "bugfree_with_commonsense")
USAGE_FILE = os.path.join(BASE_DIR, "usage_records_commonsense.tsv")


def build_ablation_case_mapping(case, document, top_k, source_file=None):
    converted = normalise_as_singleton_families(document, top_k=top_k)
    mapping = build_family_case_mapping(
        case, converted, top_k=top_k, source_file=source_file
    )
    mapping["retrieval_schema"] = "instance_level_commonsense"
    mapping["ablation"] = "retrieval_without_commonsense_generalization"
    mapping["fingerprint"] = mapping_fingerprint(mapping)
    return mapping


def configure():
    shared.BASE_DIR = BASE_DIR
    shared.SITUATIONS_DIR = SITUATIONS_DIR
    shared.MAPPING_DIR = MAPPING_DIR
    shared.RESPONSE_CACHE_DIR = RESPONSE_CACHE_DIR
    shared.BUG_DIR = BUG_DIR
    shared.BUGFREE_DIR = BUGFREE_DIR
    shared.USAGE_FILE = USAGE_FILE
    shared.VARIANT_NAME = "kuitest_instance_level_commonsense_ablation"
    shared.build_case_mapping = build_ablation_case_mapping
    for directory in (MAPPING_DIR, RESPONSE_CACHE_DIR, BUG_DIR, BUGFREE_DIR):
        os.makedirs(directory, exist_ok=True)


def main():
    configure()
    shared.main()


if __name__ == "__main__":
    main()
