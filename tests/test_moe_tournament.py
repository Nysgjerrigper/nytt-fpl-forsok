import hashlib
import json

import pandas as pd
import pytest

from fpl.model import moe_tournament as tournament
from fpl.model.moe_tournament import TournamentPlan, protocol_cutoffs, write_plan


def test_tournament_manifest_has_causal_cutoffs_and_all_milp_protocols(tmp_path):
    plan = TournamentPlan(("catboost_rmse",), str(tmp_path))
    path = write_plan(plan, {"GK": "catboost", "DEF": "catboost", "MID": "catboost_rmse", "FWD": "catboost"},
                      "gate.json")
    text = path.read_text()
    cutoffs = protocol_cutoffs()
    assert str(cutoffs["discovery_max_gw"]) in text
    assert str(cutoffs["selection_min_gw"]) in text and str(cutoffs["selection_max_gw"]) in text
    assert "position_map_standard.csv" in text and "position_map_origin.csv" in text
    assert "mid_gate_standard.csv" in text and "--wc1-gw" in text
    select = next(command for command in plan.commands() if command[2:4] == ["fpl.model.moe_tournament", "select"])
    assert "--tuned-artifact-manifest" in select
    assert select[select.index("--tuned-artifact-manifest") + 1] == str(plan.tuned_artifact_manifest)


def test_generated_select_command_reaches_cli_artifact_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    plan = TournamentPlan(("lightgbm",), str(tmp_path))
    params = tmp_path / "params.json"
    params.write_text("{}")
    manifest_payload = {"seeds": {"0": {"MID": {"lightgbm": {
        "path": str(params), "sha256": hashlib.sha256(params.read_bytes()).hexdigest()}}}}}
    manifest_payload["sha256"] = tournament.artifact_hash(manifest_payload)
    plan.tuned_artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
    plan.tuned_artifact_manifest.write_text(json.dumps(manifest_payload))
    command = next(c for c in plan.commands() if c[2:4] == ["fpl.model.moe_tournament", "select"])
    monkeypatch.setattr(tournament.pd, "read_csv", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr("fpl.features.build_feature_frame", lambda frame: frame)
    monkeypatch.setattr("fpl.features.feature_columns", lambda frame: ["x"])
    seen = {}
    monkeypatch.setattr(tournament, "generate_selection_oof", lambda *a, **k: (seen.setdefault("oof", pd.DataFrame()), tmp_path / "oof.csv"))
    monkeypatch.setattr(tournament, "select_from_oof", lambda *a, **k: tmp_path / "frozen.json")
    tournament.main(command[3:])
    assert "oof" in seen


def test_plan_following_selection_manifest_then_select(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    plan = TournamentPlan(("lightgbm",), str(tmp_path))
    commands = plan.commands()
    tune = next(c for c in commands if c[2] == "fpl.model.tuning")
    assert tune[tune.index("--stage") + 1] == "selection"
    assert tune[tune.index("--train-max-gw") + 1] == str(protocol_cutoffs()["discovery_max_gw"])
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    tuning.save_best_params("MID", "lightgbm", {"n_estimators": 2, "num_leaves": 2,
                            "learning_rate": 0.1, "min_data_in_leaf": 1, "verbosity": -1,
                            "random_state": 0}, train_max_gw=protocol_cutoffs()["discovery_max_gw"],
                            seed=0, stage="selection")
    build = next(c for c in commands if c[2:4] == ["fpl.model.moe_tournament", "build-tuned-manifest"])
    tournament.main(build[3:])
    assert plan.tuned_artifact_manifest.exists()
    select = next(c for c in commands if c[2:4] == ["fpl.model.moe_tournament", "select"])
    monkeypatch.setattr(tournament.pd, "read_csv", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr("fpl.features.build_feature_frame", lambda frame: frame)
    monkeypatch.setattr("fpl.features.feature_columns", lambda frame: ["x"])
    monkeypatch.setattr(tournament, "generate_selection_oof", lambda *a, **k: (pd.DataFrame(), tmp_path / "oof.csv"))
    monkeypatch.setattr(tournament, "select_from_oof", lambda *a, **k: tmp_path / "frozen.json")
    tournament.main(select[3:])


def test_builder_rejects_empty_tunable_artifact_without_writing_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    tuning.save_best_params("MID", "lightgbm", {}, train_max_gw=protocol_cutoffs()["discovery_max_gw"],
                            seed=0, stage="selection")
    output = tmp_path / "manifest.json"
    with pytest.raises(SystemExit):
        tournament.main(["--artifact-dir", str(tmp_path), "--experts", "lightgbm", "--seed", "0",
                         "--output", str(output), "build-tuned-manifest"])
    assert not output.exists()


def test_builder_rejects_extra_same_namespace_model_without_writing_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    cutoff = protocol_cutoffs()["discovery_max_gw"]
    for model in ("lightgbm", "catboost"):
        tuning.save_best_params("MID", model, {"n_estimators": 2} if model == "lightgbm" else {"iterations": 2},
                                train_max_gw=cutoff, seed=0, stage="selection")
    output = tmp_path / "manifest.json"
    with pytest.raises(SystemExit):
        tournament.main(["--artifact-dir", str(tmp_path), "--experts", "lightgbm", "--output", str(output),
                         "build-tuned-manifest"])
    assert not output.exists()


@pytest.mark.parametrize("content", ["{bad", "{}", '{"_meta": {"position": "BAD", "model": "lightgbm", "seed": 0, "stage": "selection", "train_max_gw": 136}}'])
def test_builder_fails_closed_on_malformed_or_unknown_scanned_artifact(tmp_path, monkeypatch, content):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    tuning.save_best_params("MID", "lightgbm", {"n_estimators": 2}, train_max_gw=protocol_cutoffs()["discovery_max_gw"], seed=0, stage="selection")
    (config.MODELS_DIR / "tuned_params_BAD_unknown.json").write_text(content)
    output = tmp_path / "manifest.json"
    with pytest.raises(SystemExit):
        tournament.main(["--artifact-dir", str(tmp_path), "--experts", "lightgbm", "--output", str(output), "build-tuned-manifest"])
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("seed", "bad"), ("stage", "bad"), ("train_max_gw", "bad")])
def test_builder_rejects_distinct_invalid_namespace_fields(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    path = tuning.save_best_params("MID", "lightgbm", {"n_estimators": 2}, train_max_gw=protocol_cutoffs()["discovery_max_gw"], seed=0, stage="selection")
    payload = json.loads(path.read_text()); payload["_meta"][field] = value; path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        tournament.main(["--artifact-dir", str(tmp_path), "--experts", "lightgbm", "build-tuned-manifest"])


@pytest.mark.parametrize("filename", ["tuned_params_DEF_lightgbm.json", "tuned_params_MID_catboost.json"])
def test_builder_rejects_filename_metadata_disagreement(tmp_path, monkeypatch, filename):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    path = tuning.save_best_params("MID", "lightgbm", {"n_estimators": 2}, train_max_gw=protocol_cutoffs()["discovery_max_gw"], seed=0, stage="selection")
    path.rename(config.MODELS_DIR / filename)
    with pytest.raises(SystemExit):
        tournament.main(["--artifact-dir", str(tmp_path), "--experts", "lightgbm", "build-tuned-manifest"])


@pytest.mark.parametrize("seed,stage,cutoff", [(1, "selection", 136), (0, "discovery", 136), (0, "selection", 1)])
def test_builder_ignores_other_artifact_namespaces(tmp_path, monkeypatch, seed, stage, cutoff):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    current = protocol_cutoffs()["discovery_max_gw"]
    tuning.save_best_params("MID", "lightgbm", {"n_estimators": 2}, train_max_gw=current, seed=0, stage="selection")
    tuning.save_best_params("MID", "catboost", {"iterations": 2}, train_max_gw=cutoff, seed=seed, stage=stage)
    output = tmp_path / "manifest.json"
    tournament.main(["--artifact-dir", str(tmp_path), "--experts", "lightgbm", "--output", str(output),
                     "build-tuned-manifest"])
    assert output.exists()


@pytest.mark.parametrize("reason", ["wrong model", "wrong position", "wrong seed", "wrong stage",
                                     "wrong cutoff", "embedded hash", "byte tamper"])
def test_builder_boundary_validation_failure_never_writes_manifest(tmp_path, monkeypatch, reason):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    from fpl import config
    from fpl.model import tuning
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    config.MODELS_DIR.mkdir()
    artifact = config.MODELS_DIR / "tuned_params_MID_lightgbm.json"
    artifact.write_text("{}")
    monkeypatch.setattr(tuning, "load_validated_params", lambda *a, **k: (_ for _ in ()).throw(ValueError(reason)))
    output = tmp_path / "manifest.json"
    with pytest.raises(SystemExit):
        tournament.main(["--artifact-dir", str(tmp_path), "--experts", "lightgbm", "--output", str(output),
                         "build-tuned-manifest"])
    assert not output.exists()
