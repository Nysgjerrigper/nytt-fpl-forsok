"""End-to-end causal tournament artifact and rejection contracts."""
import json

import numpy as np
import pandas as pd
import pytest

from fpl.model import moe_tournament as tournament


def _frame():
    rows = []
    for gw in range(1, 9):
        for pid in range(4):
            rows.append({"player_id": pid, "position": "MID", "GW_global": gw,
                         "total_points": float(pid + gw), "x": float(pid * gw), "minutes": 90,
                         "mins60_rate_roll5": 1.0})
    return pd.DataFrame(rows)


def test_selection_writes_deterministic_causal_oof_and_frozen_map(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl.model import tuning
    calls = []
    def validated(path, **kwargs):
        calls.append((path, kwargs))
        return {"n_estimators": 2, "num_leaves": 2, "learning_rate": 0.1,
                "min_data_in_leaf": 1, "verbosity": -1, "random_state": 7}
    monkeypatch.setattr(tuning, "load_validated_params", validated)
    cutoffs = {"discovery_max_gw": 3, "selection_min_gw": 5, "selection_max_gw": 8,
               "backtest_min_gw": 9, "backtest_max_gw": 12}
    oof, path = tournament.generate_selection_oof(_frame(), ["x"], ("lightgbm",), tmp_path,
                                                      seed=7, cutoffs=cutoffs,
                                                      tuned_artifacts={("MID", "lightgbm"): "validated.json"})
    assert calls and calls[0][1]["stage"] == "selection"
    assert path.parts[-3:] == ("selection", "seed-7", "oof.csv")
    assert (oof.train_max_gw < oof.GW_global).all()
    assert set(oof.seed) == {7}
    tournament.validate_lineage(oof, cutoffs)
    frozen = tournament.select_from_oof(oof, tmp_path, cutoffs=cutoffs)
    payload = json.loads(frozen.read_text())
    assert payload["champion_map"] == {"MID": "lightgbm"}
    assert payload["oof_sha256"]


def test_lineage_rejects_spent_or_noncausal_rows():
    cutoffs = {"selection_min_gw": 153, "selection_max_gw": 153}
    valid = pd.DataFrame({"position": ["MID"], "expert": ["lightgbm"], "GW_global": [153],
                          "prediction": [1.0], "actual_total_points": [1.0], "train_max_gw": [152],
                          "seed": [0], "params_hash": ["abc"], "stage": ["selection"], "mase_scale": [1.0]})
    tournament.validate_lineage(valid, cutoffs)
    spent_start, _ = tournament.spent_window()
    spent = valid.assign(GW_global=spent_start)
    with pytest.raises(ValueError, match="spent"):
        tournament.validate_lineage(spent, {"selection_min_gw": spent_start,
                                            "selection_max_gw": spent_start})
    future_train = valid.assign(train_max_gw=153)
    with pytest.raises(ValueError, match="non-causal"):
        tournament.validate_lineage(future_train, cutoffs)


def test_mid_gate_refuses_arbitrary_or_in_sample_prediction_frames():
    training = pd.DataFrame({"position": ["MID"], "GW_global": [1], "player_id": [1],
                             "total_points": [1.0], "mins60_rate_roll5": [1.0]})
    arbitrary = pd.DataFrame({"position": ["MID"], "expert": ["lightgbm"], "GW_global": [5],
                              "prediction": [1.0], "actual_total_points": [1.0]})
    with pytest.raises(ValueError, match="OOF lineage missing"):
        tournament.select_mid_gate_from_oof(training, arbitrary, champion="lightgbm",
                                            candidates=("lightgbm",),
                                            cutoffs={"discovery_max_gw": 3, "selection_min_gw": 5,
                                                     "selection_max_gw": 5, "backtest_min_gw": 6,
                                                     "backtest_max_gw": 8})


def test_protocol_cutoffs_are_season_derived():
    cutoffs = tournament.protocol_cutoffs()
    assert cutoffs["backtest_min_gw"] == 153
    assert cutoffs["selection_max_gw"] + 1 == cutoffs["backtest_min_gw"]


def test_tuned_manifest_is_self_hashed_complete_and_tamper_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    artifact = tmp_path / "params.json"
    artifact.write_text("{}")
    payload = {"seeds": {"7": {"MID": {"lightgbm": {
        "path": str(artifact), "sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()}}}}}
    payload["sha256"] = tournament.artifact_hash(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    assert tournament.load_tuned_artifact_manifest(manifest, experts=("lightgbm",), seed=7) == {("MID", "lightgbm"): str(artifact)}
    artifact.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="bytes"):
        tournament.load_tuned_artifact_manifest(manifest, experts=("lightgbm",), seed=7)


def test_tuned_manifest_rejects_extra_top_level_position(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    artifact = tmp_path / "params.json"
    artifact.write_text("{}")
    digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    payload = {"seeds": {"7": {"MID": {"lightgbm": {"path": str(artifact), "sha256": digest}},
                                "EXTRA": {"lightgbm": {"path": str(artifact), "sha256": digest}}}}}
    payload["sha256"] = tournament.artifact_hash(payload)
    manifest = tmp_path / "extra.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extra position"):
        tournament.load_tuned_artifact_manifest(manifest, experts=("lightgbm",), seed=7)


def test_finalist_seed_stability_is_derived_from_hashed_csv_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    frozen = tmp_path / "frozen.json"
    frozen_payload = {"champion_map": {"MID": "lightgbm"}}
    frozen_payload["artifact_sha256"] = tournament.artifact_hash(frozen_payload)
    frozen.write_text(json.dumps(frozen_payload))
    candidates, controls = {}, {}
    for seed in tournament.SEEDS:
        candidate = tmp_path / f"candidate-{seed}.csv"
        control = tmp_path / f"control-{seed}.csv"
        weeks = range(153, 184)
        pd.DataFrame({"gameweek": weeks, "actual_total_points": [10 + seed] * 31}).to_csv(candidate, index=False)
        pd.DataFrame({"gameweek": weeks, "actual_total_points": [5] * 31}).to_csv(control, index=False)
        origin = tmp_path / f"origin-{seed}.csv"
        origin.write_text(candidate.read_text())
        candidates[seed] = {"standard": str(candidate), "origin": str(origin)}
        controls[seed] = {"standard": str(control), "origin": str(control)}
    artifact = tournament.finalize_seed_artifacts(frozen, tmp_path, candidates, controls)
    assert tournament.seed_differences_from_artifacts(artifact) == [155.0, 186.0, 217.0]
    with pytest.raises(ValueError, match="exactly registered"):
        tournament.finalize_seed_artifacts(frozen, tmp_path, {0: candidates[0]}, {0: controls[0]})
