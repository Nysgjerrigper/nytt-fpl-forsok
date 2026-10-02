"""Persistent search identity, clean-resume determinism and fail-closed ownership."""
import pandas as pd
import pytest

from fpl.model import tuning_checkpoint as checkpoint

optuna = pytest.importorskip("optuna")


def _identity():
    return {"protocol": checkpoint.PROTOCOL, "seed": 7, "data": "registered-test-input"}


def _objective(trial):
    x = trial.suggest_float("x", -5., 5.)
    y = trial.suggest_int("y", 1, 8)
    return (x - 1.25) ** 2 + (y - 3) ** 2


def _run(path, n=15, identity=None, timeout=None):
    return checkpoint.run_checkpoint(_objective, storage_path=path, study_name="test",
        identity=_identity() if identity is None else identity, seed=7,
        n_trials=n, timeout=timeout)


def _sequence(study):
    return [(trial.number, trial.params, trial.value, trial.state) for trial in study.trials]


def test_resumed_tpe_matches_uninterrupted_beyond_startup(tmp_path):
    uninterrupted = _run(tmp_path / "full.db")
    first = _run(tmp_path / "resumed.db", n=3)
    assert len(first.trials) == 3
    resumed = _run(tmp_path / "resumed.db")
    assert _sequence(resumed) == _sequence(uninterrupted)
    assert _sequence(_run(tmp_path / "resumed.db")) == _sequence(resumed)


def test_identity_mismatch_never_adds_or_relabels_trials(tmp_path):
    path = tmp_path / "study.db"
    before = _run(path, n=2)
    with pytest.raises(ValueError, match="identity mismatch"):
        _run(path, identity={**_identity(), "data": "changed"})
    after = optuna.load_study(study_name="test", storage=f"sqlite:///{path}")
    assert _sequence(after) == _sequence(before)
    assert after.user_attrs["resume_identity"] == _identity()


@pytest.mark.parametrize("state", ["RUNNING", "WAITING"])
def test_orphan_trials_are_preserved_and_refused(tmp_path, state):
    path = tmp_path / "study.db"
    study = _run(path, n=1)
    if state == "RUNNING":
        study.ask()
    else:
        study.enqueue_trial({"x": 0., "y": 2})
    before = _sequence(study)
    with pytest.raises(ValueError, match="RUNNING/WAITING"):
        _run(path)
    assert _sequence(optuna.load_study(study_name="test", storage=f"sqlite:///{path}")) == before


def test_old_study_without_identity_is_not_adopted(tmp_path):
    path = tmp_path / "legacy.db"
    study = optuna.create_study(storage=f"sqlite:///{path}", study_name="test")
    with pytest.raises(ValueError, match="identity mismatch"):
        _run(path)
    assert study.user_attrs == {}


def test_direction_is_validated_even_if_identity_matches(tmp_path):
    path = tmp_path / "wrong.db"
    study = optuna.create_study(storage=f"sqlite:///{path}", study_name="test", direction="maximize")
    study.set_user_attr("resume_identity", _identity())
    with pytest.raises(ValueError, match="direction"):
        _run(path)


def test_active_owner_is_refused_without_creating_database(tmp_path):
    path = tmp_path / "locked.db"
    with checkpoint._exclusive_checkpoint(path):
        with pytest.raises(ValueError, match="another process"):
            _run(path)
    assert not path.exists()


def test_timeout_preserves_incomplete_study_without_returning_best(tmp_path, monkeypatch):
    ticks = iter([0., 2.])
    monkeypatch.setattr(checkpoint.time, "monotonic", lambda: next(ticks))
    path = tmp_path / "timeout.db"
    with pytest.raises(ValueError, match="incomplete after timeout"):
        _run(path, timeout=1.)
    study = optuna.load_study(study_name="test", storage=f"sqlite:///{path}")
    assert study.trials == []


def _frame():
    return pd.DataFrame({"position": ["MID", "MID"], "player_id": [1, 1],
                         "GW_global": [1, 2], "total_points": [1., 3.],
                         "x": [0., 2.], "z": [4., 5.]})


def _frame_identity(frame=None, **overrides):
    kwargs = dict(position="MID", model_name="catboost", seed=0, stage="selection",
                  train_max_gw=2, n_splits=1, folds=[({1}, {2})])
    kwargs.update(overrides)
    return checkpoint.checkpoint_identity(_frame() if frame is None else frame,
                                         ["x", "z"], **kwargs)


@pytest.mark.parametrize("column", ["x", "total_points", "player_id", "GW_global"])
def test_identity_binds_actual_frame_values(column):
    frame = _frame()
    frame.loc[0, column] += 1
    assert _frame_identity(frame) != _frame_identity()


@pytest.mark.parametrize("change", [dict(n_splits=2), dict(folds=[({1, 2}, {3})]),
    dict(train_max_gw=3), dict(seed=1), dict(stage="discovery"), dict(model_name="lightgbm")])
def test_identity_binds_folds_cutoff_seed_stage_and_search(change):
    assert _frame_identity(**change) != _frame_identity()


def test_identity_binds_dataset_hash_feature_order_dtype_source_and_runtime(tmp_path, monkeypatch):
    base = _frame_identity()
    frame = _frame()
    frame.attrs["frozen_dataset_sha256"] = "a" * 64
    assert _frame_identity(frame) != base
    frame = _frame().astype({"x": "float32"})
    assert _frame_identity(frame) != base
    kwargs = dict(position="MID", model_name="catboost", seed=0, stage="selection",
                  train_max_gw=2, n_splits=1, folds=[({1}, {2})])
    assert checkpoint.checkpoint_identity(_frame(), ["z", "x"], **kwargs) != base
    source = tmp_path / "modified_features.py"
    source.write_text("# altered feature construction\n")
    monkeypatch.setattr(checkpoint.features, "__file__", str(source))
    assert _frame_identity()["sources"] != base["sources"]
    monkeypatch.setattr(checkpoint.platform, "platform", lambda: "changed-runtime")
    assert _frame_identity()["runtime"] != base["runtime"]


def test_tune_position_records_checkpoint_and_rejects_changed_inputs(tmp_path, monkeypatch):
    from fpl.model import tuning

    frame = pd.DataFrame([{"position": "MID", "player_id": pid, "GW_global": gw,
        "total_points": float(gw + pid), "x": float(gw)}
        for gw in range(1, 7) for pid in (1, 2)])
    monkeypatch.setattr(tuning, "_suggest_params", lambda trial, model_name, seed:
                        {"gain": trial.suggest_float("gain", .1, 2.)})

    class Estimator:
        def __init__(self, gain):
            self.gain = gain
        def fit(self, X, y):
            return self
        def predict(self, X):
            return X["x"].to_numpy() * self.gain

    monkeypatch.setattr(tuning, "_build_model", lambda model_name, params, **kw: Estimator(**params))
    report = {}
    kwargs = dict(n_trials=2, n_splits=2, train_max_gw=6, seed=0, stage="selection",
                  storage_path=tmp_path / "integrated.db", checkpoint_report=report)
    best = tuning.tune_position(frame, ["x"], "MID", "catboost", **kwargs)
    assert report["sampler_protocol"] == checkpoint.PROTOCOL
    assert report["terminal_trials"] == 2
    assert len(report["identity_sha256"]) == 64
    assert tuning.tune_position(frame, ["x"], "MID", "catboost", **kwargs) == best
    changed = frame.copy()
    changed.loc[0, "x"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        tuning.tune_position(changed, ["x"], "MID", "catboost", **kwargs)


def test_reducing_trial_target_cannot_claim_fewer_trials(tmp_path):
    path = tmp_path / "study.db"
    _run(path, n=3)
    with pytest.raises(ValueError, match="exceeds"):
        _run(path, n=2)


def test_timeout_cli_never_writes_tuned_artifact(tmp_path, monkeypatch):
    from fpl import features
    from fpl.model import tuning
    monkeypatch.setattr(tuning, "_load_features", _frame)
    monkeypatch.setattr(features, "feature_columns", lambda _: ["x"])
    monkeypatch.setattr(tuning.config, "MODELS_DIR", tmp_path)
    ticks = iter([0., 2.])
    monkeypatch.setattr(checkpoint.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="incomplete after timeout"):
        tuning.main(["--position", "MID", "--model", "catboost", "--checkpoint",
                     str(tmp_path / "partial.db"), "--n-trials", "2", "--n-splits", "1",
                     "--time-budget-seconds", "1"])
    assert not list(tmp_path.glob("tuned_params*.json"))


@pytest.mark.parametrize("change", [{"protocol": "legacy"}, {"seed": 8}])
def test_direct_sampler_identity_mismatch_fails_before_database(tmp_path, change):
    path = tmp_path / "mismatch.db"
    with pytest.raises(ValueError, match="protocol/seed"):
        _run(path, identity={**_identity(), **change})
    assert not path.exists()


def test_checkpoint_exports_cannot_replace_canonical_or_existing_artifacts(tmp_path, monkeypatch):
    from fpl.model import tuning
    monkeypatch.setattr(tuning.config, "MODELS_DIR", tmp_path / "canonical")
    canonical = tuning.save_best_params("MID", "catboost", {"iterations": 10}, train_max_gw=136)
    old = canonical.read_bytes()
    metadata = {"sampler_protocol": checkpoint.PROTOCOL}
    with pytest.raises(ValueError, match="separate output"):
        tuning.save_best_params("MID", "catboost", {"iterations": 20},
                               checkpoint_metadata=metadata)
    assert canonical.read_bytes() == old
    kwargs = dict(train_max_gw=136, seed=0, stage="selection", checkpoint_metadata=metadata,
                  output_dir=tmp_path / "exports")
    exported = tuning.save_best_params("MID", "catboost", {"iterations": 20}, **kwargs)
    old_export = exported.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        tuning.save_best_params("MID", "catboost", {"iterations": 30}, **kwargs)
    assert exported.read_bytes() == old_export
    request = dict(position="MID", model_name="catboost", seed=0,
                   stage="selection", max_train_gw=136)
    with pytest.raises(ValueError, match="sampler protocol"):
        tuning.load_validated_params(exported, **request)
    assert tuning.load_validated_params(exported, **request,
        sampler_protocol=checkpoint.PROTOCOL) == {"iterations": 20}


def test_failed_and_pruned_trials_count_attempts_without_silent_repair(tmp_path):
    path = tmp_path / "states.db"
    def objective(trial):
        if trial.number == 0:
            raise RuntimeError("ordinary failure")
        if trial.number == 1:
            raise optuna.TrialPruned()
        return _objective(trial)
    kwargs = dict(storage_path=path, study_name="test", identity=_identity(),
                  seed=7, n_trials=3, timeout=None)
    with pytest.raises(RuntimeError, match="ordinary failure"):
        checkpoint.run_checkpoint(objective, **kwargs)
    resumed = checkpoint.run_checkpoint(objective, **kwargs)
    assert [trial.state.name for trial in resumed.trials] == ["FAIL", "PRUNED", "COMPLETE"]


def test_tabr_dependency_versions_are_identity_bound(monkeypatch):
    original = checkpoint.importlib.metadata.version
    monkeypatch.setattr(checkpoint.importlib.metadata, "version",
        lambda name: "test-version" if name in ("skorch", "faiss-cpu") else original(name))
    packages = _frame_identity()["runtime"]["packages"]
    assert packages["skorch"] == packages["faiss-cpu"] == "test-version"


def test_successful_checkpoint_cli_exports_separately_with_protocol_metadata(tmp_path, monkeypatch):
    import json
    from fpl import features
    from fpl.model import tuning

    monkeypatch.setattr(tuning.config, "MODELS_DIR", tmp_path)
    canonical = tuning.save_best_params("MID", "catboost", {"iterations": 10}, train_max_gw=136)
    original_bytes = canonical.read_bytes()
    monkeypatch.setattr(tuning, "_load_features", lambda: pd.DataFrame({
        "position": ["MID"] * 6, "player_id": [1] * 6, "GW_global": range(1, 7),
        "total_points": [1., 3., 2., 4., 3., 6.], "x": [0., 1., 2., 3., 4., 5.]}))
    monkeypatch.setattr(features, "feature_columns", lambda _: ["x"])
    monkeypatch.setattr(tuning, "_suggest_params", lambda trial, model_name, seed:
                        {"gain": trial.suggest_float("gain", .1, 2.)})

    class Estimator:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return X["x"].to_numpy()

    monkeypatch.setattr(tuning, "_build_model", lambda *args, **kwargs: Estimator())
    tuning.main(["--position", "MID", "--model", "catboost", "--checkpoint",
                 str(tmp_path / "search.db"), "--n-trials", "2", "--n-splits", "1"])
    assert canonical.read_bytes() == original_bytes
    exports = list((tmp_path / ".optuna_exports").rglob("tuned_params*.json"))
    assert len(exports) == 1
    meta = json.loads(exports[0].read_text())["_meta"]["checkpoint"]
    assert meta["sampler_protocol"] == checkpoint.PROTOCOL
    assert meta["terminal_trials"] == meta["trial_target"] == 2
    assert meta["trial_states"]["COMPLETE"] == 2


def test_no_completed_trial_never_returns_a_parameter_artifact(tmp_path):
    with pytest.raises(ValueError, match="no completed trial"):
        checkpoint.run_checkpoint(lambda trial: float("nan"), storage_path=tmp_path / "nan.db",
            study_name="test", identity=_identity(), seed=7, n_trials=1, timeout=None)


@pytest.mark.parametrize("cutoff", [True, "136", -1, 136.5, float("nan"), float("inf")])
def test_hashed_but_malformed_cutoff_is_not_causal_provenance(tmp_path, monkeypatch, cutoff):
    from fpl.model import tuning
    monkeypatch.setattr(tuning.config, "MODELS_DIR", tmp_path)
    path = tuning.save_best_params("MID", "catboost", {"iterations": 2},
                                  train_max_gw=cutoff, seed=0, stage="selection")
    with pytest.raises(ValueError, match="cutoff"):
        tuning.load_validated_params(path, position="MID", model_name="catboost",
                                    seed=0, stage="selection", max_train_gw=136)
