import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl.model.mid_gate import (
    MidGateConfig,
    TertileThresholds,
    check_diversity_eligibility,
    fit_tertile_thresholds,
    select_mid_gate,
    validate_routing_features,
)


def test_diversity_trigger_requires_accuracy_correlation_and_four_wins_each():
    weeks = np.repeat(np.arange(137, 153), 2)
    truth = np.tile([0.0, 10.0], 16)
    champion = truth.copy()
    challenger = truth.copy()
    # Alternate which expert wins; residuals are intentionally anti-correlated.
    champion[::4] += 1.0
    challenger[2::4] += 1.0
    result = check_diversity_eligibility(
        truth, champion, challenger, weeks, mase_scale=1.0
    )
    assert result.eligible
    assert result.champion_gw_wins == 8
    assert result.challenger_gw_wins == 8
    assert result.residual_correlation < 0.95

    too_similar = check_diversity_eligibility(
        truth, champion, champion + 0.01, weeks, mase_scale=1.0
    )
    assert not too_similar.eligible


def test_diversity_trigger_rejects_truncated_selection_window():
    weeks = np.repeat(np.arange(137, 152), 2)
    truth = np.zeros(len(weeks))
    result = check_diversity_eligibility(
        truth, np.tile([0.0, 1.0], 15), np.tile([1.0, 0.0], 15), weeks, 1.0
    )
    assert not result.eligible
    assert "expected 16" in result.reasons[0]


def test_tertiles_are_fit_from_training_only_and_boundaries_are_stable():
    training = pd.DataFrame({"mins60_rate_roll5": np.arange(9, dtype=float)})
    thresholds = fit_tertile_thresholds(training)
    validation = pd.DataFrame({"mins60_rate_roll5": [-1000.0, 1000.0]})
    assert thresholds == fit_tertile_thresholds(training)  # validation cannot affect fit
    assert thresholds.route(
        [np.nan, thresholds.low_upper, thresholds.low_upper + 0.01,
         thresholds.medium_upper, thresholds.medium_upper + 0.01]
    ).tolist() == ["low", "low", "medium", "medium", "high"]


def _selection_frames(rows_per_regime=8):
    training = pd.DataFrame(
        {
            "GW_global": np.repeat(np.arange(129, 137), 3),
            "player_id": np.tile([1, 2, 3], 8),
            "mins60_rate_roll5": np.tile([0.1, 0.5, 0.9], 8),
            "total_points": np.repeat(np.arange(8, dtype=float), 3) + np.tile([0.0, 1.0, 3.0], 8),
        }
    )
    rows = []
    for regime_value in (0.1, 0.5, 0.9):
        for i in range(rows_per_regime):
            truth = float(i % 3)
            rows.append(
                {
                    "GW_global": 137 + i % 8,
                    "mins60_rate_roll5": regime_value,
                    "total_points": truth,
                    "pred_champion": truth + 1.0,
                    "pred_alt": truth if regime_value == 0.5 else truth + 0.99,
                }
            )
    return training, pd.DataFrame(rows)


def test_regime_selection_switches_only_with_support_and_material_improvement():
    training, validation = _selection_frames()
    config = select_mid_gate(
        training,
        validation,
        {"catboost": "pred_champion", "tabm": "pred_alt"},
        champion="catboost",
        min_rows=8,
        min_gameweeks=8,
        training_max_gw=136,
        provenance={"seed": 0, "package": "test"},
    )
    assert config.selections["low"].expert == "catboost"  # gain 0.01 < 0.02
    assert config.selections["medium"].expert == "tabm"
    assert config.selections["high"].expert == "catboost"

    routed = config.route(pd.DataFrame({"mins60_rate_roll5": [0.1, 0.5, 0.9]}))
    assert routed.tolist() == ["catboost", "tabm", "catboost"]
    predicted = config.predict(
        pd.DataFrame({"mins60_rate_roll5": [0.1, 0.5, 0.9]}),
        {"catboost": [1.0, 2.0, 3.0], "tabm": [4.0, 5.0, 6.0]},
    )
    assert predicted.tolist() == [1.0, 5.0, 3.0]


def test_sparse_regime_falls_back_to_champion():
    training, validation = _selection_frames(rows_per_regime=7)
    config = select_mid_gate(
        training,
        validation,
        {"catboost": "pred_champion", "tabm": "pred_alt"},
        champion="catboost",
        min_rows=8,
        min_gameweeks=4,
    )
    assert all(selection.expert == "catboost" for selection in config.selections.values())
    assert all(selection.fallback_reason == "insufficient validation support"
               for selection in config.selections.values())


def test_routing_rejects_target_minutes_and_player_identity():
    for forbidden in ("minutes", "target_minutes", "player_id", "name"):
        with pytest.raises(ValueError, match="forbidden MID routing"):
            validate_routing_features(("mins60_rate_roll5", forbidden))
    with pytest.raises(ValueError, match="must route only"):
        validate_routing_features(("some_other_feature",))


def test_config_round_trip_is_json_serializable_and_deterministic():
    training, validation = _selection_frames()
    config = select_mid_gate(
        training,
        validation,
        {"catboost": "pred_champion", "tabm": "pred_alt"},
        champion="catboost",
        min_rows=8,
        min_gameweeks=8,
        training_max_gw=136,
        provenance={"seed": 0, "versions": {"numpy": np.__version__}},
    )
    restored = MidGateConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    frame = pd.DataFrame({"mins60_rate_roll5": [0.2, 0.6, np.nan]})
    predictions = {"catboost": [1.0, 2.0, 3.0], "tabm": [4.0, 5.0, 6.0]}
    np.testing.assert_array_equal(config.route(frame), restored.route(frame))
    np.testing.assert_allclose(config.predict(frame, predictions), restored.predict(frame, predictions))
    assert restored.provenance["seed"] == 0


def test_declared_training_cutoff_is_enforced():
    training, validation = _selection_frames()
    training.loc[len(training)] = {"GW_global": 137, "mins60_rate_roll5": 0.5}
    with pytest.raises(ValueError, match="chronological cutoff"):
        select_mid_gate(
            training,
            validation,
            {"catboost": "pred_champion", "tabm": "pred_alt"},
            champion="catboost",
            min_rows=8,
            min_gameweeks=8,
            training_max_gw=136,
        )

    clean_training, overlapping_validation = _selection_frames()
    overlapping_validation.loc[0, "GW_global"] = 136
    with pytest.raises(ValueError, match="strictly after"):
        select_mid_gate(
            clean_training,
            overlapping_validation,
            {"catboost": "pred_champion", "tabm": "pred_alt"},
            champion="catboost",
            min_rows=8,
            min_gameweeks=8,
            training_max_gw=136,
        )


def test_gate_scale_is_training_only_and_records_its_cutoff():
    training, validation = _selection_frames()
    baseline = select_mid_gate(
        training, validation, {"catboost": "pred_champion", "tabm": "pred_alt"},
        champion="catboost", min_rows=8, min_gameweeks=8,
    )
    changed_validation = validation.copy()
    changed_validation["total_points"] = 10_000.0
    unchanged = select_mid_gate(
        training, changed_validation, {"catboost": "pred_champion", "tabm": "pred_alt"},
        champion="catboost", min_rows=8, min_gameweeks=8,
    )
    assert baseline.mase_scale == unchanged.mase_scale
    assert baseline.mase_scale_training_max_gw == 136
    assert baseline.provenance == unchanged.provenance


def test_threshold_validation_rejects_reversed_boundaries():
    with pytest.raises(ValueError, match="cannot exceed"):
        TertileThresholds(low_upper=0.8, medium_upper=0.2)
