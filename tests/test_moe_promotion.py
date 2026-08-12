import pandas as pd
import pytest
import json

from fpl.model.moe_promotion import assess_manifest
from fpl.model import moe_tournament as tournament


def _write_run(path, points):
    pd.DataFrame({"gameweek": range(153, 184),
                  "actual_total_points": points}).to_csv(path, index=False)
    return str(path)


def test_assess_manifest_promotes_simpler_credible_candidate(tmp_path):
    control_std = _write_run(tmp_path / "control_std.csv", [50.0] * 31)
    control_org = _write_run(tmp_path / "control_org.csv", [50.0] * 31)
    hard_std = _write_run(tmp_path / "hard_std.csv", [55.0] * 31)
    hard_org = _write_run(tmp_path / "hard_org.csv", [51.0] * 31)
    gate_std = _write_run(tmp_path / "gate_std.csv", [56.0] * 31)
    gate_org = _write_run(tmp_path / "gate_org.csv", [52.0] * 31)
    selection = tmp_path / "selection.json"
    payload = {"champion_map": {p: "catboost" for p in tournament.POSITIONS}}
    payload["artifact_sha256"] = tournament.artifact_hash(payload)
    selection.write_text(json.dumps(payload))
    seed_paths, controls = {}, {}
    for seed in tournament.SEEDS:
        candidate = _write_run(tmp_path / f"seed_candidate_{seed}.csv", [55.0] * 31)
        origin = _write_run(tmp_path / f"seed_origin_{seed}.csv", [51.0] * 31)
        seed_paths[seed] = {"standard": candidate, "origin": origin}
        controls[seed] = {"standard": control_std, "origin": control_org}
    frozen = tournament.finalize_seed_artifacts(selection, tmp_path, seed_paths, controls)
    frozen_hash = json.loads(frozen.read_text())["artifact_sha256"]
    result = assess_manifest({
        "frozen_finalists": str(frozen),
        "candidates": {
            "position_map": {"finalists_sha256": frozen_hash,
                             "finalists": str(frozen), "headline_seed": 0, "complexity": 0},
            "mid_gate": {"finalists_sha256": frozen_hash,
                         "finalists": str(frozen), "headline_seed": 0, "complexity": 1},
        },
    }, n_boot=200)
    assert result["decisions"]["position_map"]["promote"] is True
    assert result["decisions"]["mid_gate"]["promote"] is True
    assert result["winner"] == "position_map"


def test_assess_manifest_rejects_incomplete_or_diagnostic_family(tmp_path):
    with pytest.raises(ValueError, match="Holm family"):
        assess_manifest({"candidates": {}})
    with pytest.raises(ValueError, match="Holm family"):
        assess_manifest({"candidates": {str(i): {} for i in range(3)}})
