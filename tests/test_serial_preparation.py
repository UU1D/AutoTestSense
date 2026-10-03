from __future__ import annotations

import json

from commonsense_repro.commonsense_consolidation.input_preparation.prepare_unassigned_commonsense import (
    covered_ids,
    load_units,
)


def test_serial_inputs_select_only_uncovered_units(tmp_path) -> None:
    source = tmp_path / "units.json"
    source.write_text(
        json.dumps(
            [
                {"id": "U1", "situation": "s1", "violated_commonsense_rule": "r1"},
                {"id": "U2", "situation": "s2", "violated_commonsense_rule": "r2"},
            ]
        ),
        encoding="utf-8",
    )
    catalog = {"rule_families": [{"family_id": "F1", "member_ids": ["U1"]}]}
    units = load_units([source])
    pending = sorted(set(units) - covered_ids(catalog))
    assert pending == ["U2"]
