"""Player-round scoring and fail-closed tournament OOF contracts."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fpl.model import moe_tournament as tournament, tuning


CUTOFFS = {"discovery_max_gw": 3, "selection_min_gw": 5, "selection_max_gw": 8,
           "backtest_min_gw": 9, "backtest_max_gw": 12}
EXPERTS = ("lightgbm_l2", "catboost_rmse")


def fixture_frame():
    """Use two fixture rows per player-round and an observed training gap."""
    return pd.DataFrame([
        {"player_id": pid, "position": "MID", "GW_global": gw,
         "fixture": gw * 10 + fixture, "x": fixture,
         "total_points": float(pid + gw + fixture), "minutes": 90,
         "mins60_rate_roll5": pid / 4 if gw >= 5 else np.nan}
        for gw in (1, 2, 5, 6, 7, 8) for pid in range(4) for fixture in (1, 2)
    ])


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID",))
    monkeypatch.setattr(tuning, "load_validated_params", lambda *a, **k: {})
    fits = []

    class FixturePredictor:
        def __init__(self, name):
            self.name = name

        def predict(self, X):
            return X.x.to_numpy(dtype=float) + (1 if self.name == "catboost_rmse" else 0)

    def fit(name, X, y, **kwargs):
        fits.append((name, X.copy(), y.copy(), kwargs))
        return FixturePredictor(name)

    monkeypatch.setattr(tournament.models, "fit_model", fit)
    return fits


def generate(tmp_path, frame=None, experts=EXPERTS):
    return tournament.generate_selection_oof(
        fixture_frame() if frame is None else frame, ["x"], experts, tmp_path,
        cutoffs=CUTOFFS, tuned_artifacts={("MID", name): "validated.json" for name in experts},
    )


def test_dgw_fits_fixtures_and_emits_summed_round_panel(tmp_path, pipeline):
    frame = fixture_frame()
    oof, path = generate(tmp_path, frame)
    assert len(pipeline[0][1]) == 16
    pd.testing.assert_series_equal(pipeline[0][2], frame.loc[frame.GW_global < 5, "total_points"])
    assert len(oof) == 4 * 4 * 2
    assert not oof.duplicated(["player_id", "position", "GW_global", "expert"]).any()
    assert (oof.fixture_count == 2).all()
    first = oof[oof.GW_global == 5]
    assert (first.train_max_gw == 2).all()
    assert (first.mase_scale == 2).all()
    expected = frame.groupby(["player_id", "position", "GW_global"]).total_points.sum()
    for expert, prediction in (("lightgbm_l2", 3), ("catboost_rmse", 5)):
        actual = oof[oof.expert == expert].set_index(["player_id", "position", "GW_global"])
        assert (actual.prediction == prediction).all()
        pd.testing.assert_series_equal(actual.actual_total_points.sort_index(),
                                       expected.reindex(actual.index).sort_index(), check_names=False)
    metadata = json.loads(path.with_name("oof_metadata.json").read_text())
    assert metadata["schema"] == "player_round_v1"
    assert metadata["oof_csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["oof_sha256"] == tournament.artifact_hash(oof.to_dict("records"))


def test_round_scale_uses_strictly_earlier_targets(tmp_path, pipeline):
    first, _ = generate(tmp_path / "first")
    changed = fixture_frame()
    changed.loc[changed.GW_global >= 5, "total_points"] += 1000
    second, _ = generate(tmp_path / "second", changed)
    np.testing.assert_array_equal(first.loc[first.GW_global == 5, "mase_scale"],
                                  second.loc[second.GW_global == 5, "mase_scale"])


@pytest.mark.parametrize("fault", ["week", "training", "duplicate", "fixture_missing", "fixture_column",
                                   "target", "identity", "fractional_gw", "routing", "routing_disagreement"])
def test_preflight_rejects_invalid_frame_before_any_fit(tmp_path, pipeline, fault):
    frame = fixture_frame()
    if fault == "week":
        frame = frame[frame.GW_global != 6]
    elif fault == "training":
        frame = frame[frame.GW_global >= 5]
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif fault == "fixture_missing":
        frame.loc[0, "fixture"] = np.nan
    elif fault == "fixture_column":
        frame = frame.drop(columns="fixture")
    elif fault == "target":
        frame.loc[0, "total_points"] = np.inf
    elif fault == "identity":
        frame.loc[0, "player_id"] = np.nan
    elif fault == "fractional_gw":
        frame["GW_global"] = frame.GW_global.astype(float)
        frame.loc[0, "GW_global"] = 1.5
    elif fault == "routing":
        frame.loc[frame.GW_global == 5, "mins60_rate_roll5"] = np.inf
    else:
        frame.loc[frame.GW_global == 5, "mins60_rate_roll5"] = np.tile([0.1, 0.9], 4)
    with pytest.raises(ValueError):
        generate(tmp_path, frame)
    assert pipeline == []
    assert not (tmp_path / "selection" / "seed-0" / "oof.csv").exists()


@pytest.mark.parametrize("prediction", [1.0, np.array([[1.0]]), np.ones(1), np.full(8, np.inf)])
def test_invalid_prediction_shapes_and_values_never_emit_oof(tmp_path, pipeline, monkeypatch, prediction):
    class Invalid:
        def predict(self, X):
            return prediction

    monkeypatch.setattr(tournament.models, "fit_model", lambda *a, **k: Invalid())
    with pytest.raises(ValueError, match="invalid predictions"):
        generate(tmp_path)
    assert not (tmp_path / "selection" / "seed-0" / "oof.csv").exists()


@pytest.mark.parametrize("artifact", ["oof.csv", "oof_metadata.json"])
def test_existing_seed_artifact_is_refused_before_fit(tmp_path, pipeline, artifact):
    path = tmp_path / "selection" / "seed-0" / artifact
    path.parent.mkdir(parents=True)
    path.write_text("preserved bytes")
    with pytest.raises(ValueError, match="already exist"):
        generate(tmp_path)
    assert pipeline == []
    assert path.read_text() == "preserved bytes"


@pytest.mark.parametrize("fault", ["target", "missing_expert", "routing", "seed", "week", "duplicate"])
def test_selection_and_mid_gate_refuse_incomparable_panels(tmp_path, pipeline, fault):
    oof, _ = generate(tmp_path)
    if fault == "target":
        oof.loc[0, "actual_total_points"] += 1
    elif fault == "missing_expert":
        oof = oof.iloc[1:]
    elif fault == "routing":
        oof.loc[0, "mins60_rate_roll5"] += 0.1
    elif fault == "seed":
        oof.loc[0, "seed"] = 1
    elif fault == "week":
        oof = oof[oof.GW_global != 6]
    else:
        oof = pd.concat([oof, oof.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        tournament.select_from_oof(oof, tmp_path / "selection-report", cutoffs=CUTOFFS)
    with pytest.raises(ValueError):
        tournament.select_mid_gate_from_oof(fixture_frame(), oof, champion=EXPERTS[0],
                                            candidates=EXPERTS, cutoffs=CUTOFFS)


def test_mid_gate_scale_uses_summed_training_rounds(tmp_path, pipeline):
    frame = fixture_frame()
    # Threshold fitting uses historical observed routing; only first-round values may be missing.
    frame.loc[frame.GW_global == 2, "mins60_rate_roll5"] = frame.loc[frame.GW_global == 2, "player_id"] / 4
    oof, _ = generate(tmp_path, frame)
    training = frame[frame.GW_global < 5]
    gate = tournament.select_mid_gate_from_oof(training, oof, champion=EXPERTS[0],
                                              candidates=EXPERTS, cutoffs=CUTOFFS)
    assert gate.mase_scale == 2
    assert gate.mase_scale_training_max_gw == 2


def test_missing_other_position_is_rejected_before_fit(tmp_path, pipeline, monkeypatch):
    monkeypatch.setattr(tournament, "POSITIONS", ("MID", "DEF"))
    with pytest.raises(ValueError, match="DEF lacks complete"):
        generate(tmp_path)
    assert pipeline == []


def test_zero_training_round_scale_is_rejected_before_fit(tmp_path, pipeline):
    frame = fixture_frame()
    frame.loc[frame.GW_global < 5, "total_points"] = 2.0
    with pytest.raises(ValueError, match="MASE scale must be finite and positive"):
        generate(tmp_path, frame)
    assert pipeline == []



def test_future_malformed_labels_do_not_affect_selection(tmp_path, pipeline):
    original, _ = generate(tmp_path / "original")
    future = fixture_frame().iloc[:2].copy()
    future["GW_global"] = 999
    future["total_points"] = np.inf
    future["mins60_rate_roll5"] = np.nan
    future["fixture"] = 0
    frame = pd.concat([fixture_frame(), future], ignore_index=True)
    after, _ = generate(tmp_path / "after", frame)
    pd.testing.assert_frame_equal(original, after)


@pytest.mark.parametrize("disagreement", [np.nan, 0.9])
def test_mid_training_fixture_routing_must_agree_including_nan(tmp_path, pipeline, disagreement):
    frame = fixture_frame()
    frame.loc[frame.GW_global == 2, "mins60_rate_roll5"] = frame.loc[frame.GW_global == 2, "player_id"] / 4
    oof, _ = generate(tmp_path, frame)
    training = frame[frame.GW_global < 5].copy()
    affected = training.index[(training.GW_global == 2) & (training.player_id == 0)][0]
    training.loc[affected, "mins60_rate_roll5"] = disagreement
    with pytest.raises(ValueError, match="MID training fixtures disagree"):
        tournament.select_mid_gate_from_oof(training, oof, champion=EXPERTS[0],
                                            candidates=EXPERTS, cutoffs=CUTOFFS)


@pytest.mark.parametrize("artifact", ["expert_ranking.csv", "frozen_selection.json"])
def test_selection_reports_refuse_existing_artifacts_before_scoring(tmp_path, pipeline, monkeypatch, artifact):
    oof, _ = generate(tmp_path / "oof")
    path = tmp_path / "report" / "selection" / artifact
    path.parent.mkdir(parents=True)
    path.write_text("preserved selection bytes")
    monkeypatch.setattr(tournament, "mase", lambda *a: pytest.fail("existing artifact reached scoring"))
    with pytest.raises(ValueError, match="already exist"):
        tournament.select_from_oof(oof, tmp_path / "report", cutoffs=CUTOFFS)
    assert path.read_text() == "preserved selection bytes"
    assert sorted(p.name for p in path.parent.iterdir()) == [artifact]


def test_selection_exclusive_write_preserves_concurrent_frozen_artifact(tmp_path, pipeline, monkeypatch):
    oof, _ = generate(tmp_path / "oof")
    path = tmp_path / "report" / "selection" / "frozen_selection.json"
    real_mase = tournament.mase

    def concurrent_writer(*args):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("concurrently preserved")
        return real_mase(*args)

    monkeypatch.setattr(tournament, "mase", concurrent_writer)
    with pytest.raises(FileExistsError):
        tournament.select_from_oof(oof, tmp_path / "report", cutoffs=CUTOFFS)
    assert path.read_text() == "concurrently preserved"


@pytest.mark.parametrize("all_missing", [False, True])
def test_selection_missing_history_survives_pivot_and_routes_low(tmp_path, pipeline, monkeypatch, all_missing):
    frame = fixture_frame()
    frame.loc[frame.GW_global == 2, "mins60_rate_roll5"] = frame.loc[frame.GW_global == 2, "player_id"] / 4
    missing = (frame.GW_global >= 5) & ((frame.player_id == 0) | all_missing)
    frame.loc[missing, "mins60_rate_roll5"] = np.nan
    oof, _ = generate(tmp_path, frame)
    assert len(oof) == 32
    assert oof.mins60_rate_roll5.isna().sum() == (32 if all_missing else 8)
    tournament.select_from_oof(oof, tmp_path / "report", cutoffs=CUTOFFS)
    real_select = tournament.select_mid_gate
    observed = []

    def capture(training, validation, *args, **kwargs):
        observed.append((training.copy(), validation.copy()))
        return real_select(training, validation, *args, **kwargs)

    monkeypatch.setattr(tournament, "select_mid_gate", capture)
    gate = tournament.select_mid_gate_from_oof(frame[frame.GW_global < 5], oof,
                                              champion=EXPERTS[0], candidates=EXPERTS, cutoffs=CUTOFFS)
    training, validation = observed[0]
    assert len(validation) == 16
    assert validation.mins60_rate_roll5.isna().sum() == (16 if all_missing else 4)
    assert training.mins60_rate_roll5.isna().sum() == 4
    missing_validation = validation[validation.mins60_rate_roll5.isna()]
    assert (gate.thresholds.route(missing_validation.mins60_rate_roll5) == "low").all()
    assert gate.selections["low"].rows >= len(missing_validation)
    assert gate.mase_scale == 2


def test_selection_fixture_missingness_disagreement_fails_before_fit(tmp_path, pipeline):
    frame = fixture_frame()
    affected = frame.index[(frame.GW_global == 5) & (frame.player_id == 0)][0]
    frame.loc[affected, "mins60_rate_roll5"] = np.nan
    with pytest.raises(ValueError, match="fixture rows disagree"):
        generate(tmp_path, frame)
    assert pipeline == []


def test_oof_experts_must_agree_on_routing_missingness(tmp_path, pipeline):
    frame = fixture_frame()
    frame.loc[(frame.GW_global >= 5) & (frame.player_id == 0), "mins60_rate_roll5"] = np.nan
    oof, _ = generate(tmp_path, frame)
    affected = oof.index[(oof.GW_global == 5) & (oof.player_id == 0)][0]
    oof.loc[affected, "mins60_rate_roll5"] = 0.1
    with pytest.raises(ValueError, match="experts disagree"):
        tournament.select_from_oof(oof, tmp_path / "report", cutoffs=CUTOFFS)
