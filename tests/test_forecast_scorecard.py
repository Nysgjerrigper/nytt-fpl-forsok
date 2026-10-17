"""Hand-computed diagnostics, strict paired identity, and origin lead separation."""

import json

import numpy as np
import pandas as pd
import pytest

from fpl.forecast_scorecard import summarize_forecast_pair


def _round():
    return pd.DataFrame({"player_id": [1, 2, 3], "GW": [153] * 3, "position": ["MID"] * 3,
                         "predicted_total_points": [5., 4., 1.], "actual_total_points": [6., 2., 0.]})


def _summarize(a, b=None, *, protocol="standard", grain="player_round", weeks=None):
    return summarize_forecast_pair(a, a.copy() if b is None else b,
                                   information_protocol=protocol, grain=grain,
                                   gameweeks=[153] if weeks is None else weeks,
                                   population="Optimizer common pool including zero-filled rows")


def test_duplicate_column_labels_rejected_explicitly():
    frame = _round()
    frame = pd.concat([frame, frame[["position"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column"):
        _summarize(frame)


def test_same_target_position_cannot_change_across_origins():
    first = _round().assign(origin_gw=152)
    second = _round().assign(origin_gw=153, position="FWD")
    frame = pd.concat([first, second], ignore_index=True)
    with pytest.raises(ValueError, match="positions disagree"):
        _summarize(frame, protocol="origin")


def test_double_gameweek_sums_before_metrics_and_preserves_fixture_counts():
    a = pd.DataFrame({"player_id": [1, 1, 2, 3], "GW": [153] * 4, "fixture": [10, 11, 10, 10],
                      "position": ["MID"] * 4, "predicted_total_points": [2., 3., 4., 1.],
                      "actual_total_points": [1., 5., 2., 0.]})
    b = a.copy()
    b["predicted_total_points"] = [1., 1., 5., 0.]
    result = _summarize(a, b, grain="fixture")
    candidate = result["summaries"][0]["candidate"]["overall"]
    control = result["summaries"][0]["control"]["overall"]
    assert candidate["player_rounds"] == 3 and candidate["fixture_rows"] == 4
    assert candidate["mae"] == pytest.approx(4 / 3)
    assert candidate["rmse"] == pytest.approx(np.sqrt(2))
    assert candidate["bias"] == pytest.approx(2 / 3)
    assert candidate["total_calibration"] == 1.25
    assert candidate["spearman"] == 1
    assert candidate["top1_capture"] == 1
    assert candidate["aggregate_true_total"] == 8
    assert candidate["top1_true_max_sum"] == 6
    assert candidate["ranking_groups"] == candidate["spearman_used_groups"] == 1
    assert candidate["spearman_skipped_groups"] == candidate["tied_top_groups"] == 0
    assert control["mae"] == pytest.approx(7 / 3)
    assert control["rmse"] == pytest.approx(np.sqrt(25 / 3))
    assert control["bias"] == pytest.approx(-1 / 3)
    assert control["total_calibration"] == 7 / 8
    assert control["spearman"] == .5
    assert control["top1_capture"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("change", ["drop_player", "changed_truth", "changed_position"])
def test_pair_requires_full_population_positions_and_outcomes(change):
    a, b = _round(), _round()
    if change == "drop_player":
        b = b.iloc[:-1]
    elif change == "changed_truth":
        b.loc[0, "actual_total_points"] += 1
    else:
        b.loc[0, "position"] = "FWD"
    with pytest.raises(ValueError, match="match exactly"):
        _summarize(a, b)


def test_fixture_identity_mismatch_is_not_hidden_by_equal_player_round_totals():
    a = _round().assign(fixture=10)
    b = a.copy()
    b.loc[0, "fixture"] = 11
    with pytest.raises(ValueError, match="match exactly"):
        _summarize(a, b, grain="fixture")


@pytest.mark.parametrize("grain", ["player_round", "fixture"])
def test_duplicate_ambiguous_keys_rejected(grain):
    a = _round()
    if grain == "fixture":
        a["fixture"] = 10
    a = pd.concat([a, a.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate|ambiguous"):
        _summarize(a, grain=grain)


def test_fixture_position_consistency_required_within_player_round():
    a = _round().assign(fixture=10)
    extra = a.iloc[[0]].assign(fixture=11, position="FWD")
    a = pd.concat([a, extra], ignore_index=True)
    with pytest.raises(ValueError, match="position"):
        _summarize(a, grain="fixture")


def test_tied_top_pick_uses_lowest_numeric_player_code_independent_of_row_order():
    a = _round()
    a["player_id"] = [2, 10, 3]
    a["predicted_total_points"] = [5., 5., 1.]
    a["actual_total_points"] = [2., 10., 0.]
    first = _summarize(a)
    shuffled = _summarize(a.iloc[::-1], a.sample(frac=1, random_state=3))
    assert first == shuffled
    stats = first["summaries"][0]["candidate"]["overall"]
    assert stats["top1_capture"] == .2
    assert stats["tied_top_groups"] == 1


def test_zero_outcomes_produce_json_null_ratios_and_skipped_spearman():
    a = _round()
    a["actual_total_points"] = 0.
    result = _summarize(a)
    stats = result["summaries"][0]["candidate"]["overall"]
    assert stats["total_calibration"] is None
    assert stats["top1_capture"] is None
    assert stats["spearman"] is None
    assert stats["aggregate_true_total"] == stats["top1_true_max_sum"] == 0
    assert stats["spearman_used_groups"] == 0 and stats["spearman_skipped_groups"] == 1
    assert '"top1_capture": null' in json.dumps(result, allow_nan=False)


def test_negative_outcome_ratios_remain_unclamped():
    a = _round()
    a["actual_total_points"] = [-2., -5., -10.]
    a["predicted_total_points"] = [0., 1., 2.]
    result = _summarize(a)
    stats = result["summaries"][0]["candidate"]["overall"]
    assert stats["top1_capture"] == 5
    assert stats["total_calibration"] == pytest.approx(-3 / 17)
    json.dumps(result, allow_nan=False)


def _origins():
    current = _round().assign(GW=154, origin_gw=154)
    earlier = _round().assign(GW=154, origin_gw=153)
    earlier["predicted_total_points"] = [0., 2., 6.]
    return pd.concat([current, earlier], ignore_index=True)


def test_origin_metrics_are_only_lead_strata_without_repeated_target_pooled_metric():
    result = _summarize(_origins(), protocol="origin", weeks=[154])
    assert [summary["lead"] for summary in result["summaries"]] == [0, 1]
    assert "overall" not in result and "candidate" not in result
    assert result["summaries"][0]["candidate"]["overall"]["player_rounds"] == 3
    assert result["summaries"][1]["candidate"]["overall"]["player_rounds"] == 3
    assert result["summaries"][0]["candidate"]["overall"]["mae"] == pytest.approx(4 / 3)
    assert result["summaries"][1]["candidate"]["overall"]["mae"] == 4


def test_inconsistent_realized_targets_across_origins_rejected_even_when_pair_matches():
    a = _origins()
    a.loc[3, "actual_total_points"] += 1
    with pytest.raises(ValueError, match="across forecast origins"):
        _summarize(a, protocol="origin", weeks=[154])


def test_fixture_origin_targets_checked_after_round_aggregation():
    a = _origins().assign(fixture=10)
    # Fixture-level decomposition differs across origins but summed truth is identical.
    extra = a.iloc[[0, 3]].copy().assign(fixture=11)
    a.loc[0, "actual_total_points"] -= 2
    a.loc[3, "actual_total_points"] -= 1
    extra["actual_total_points"] = [2., 1.]
    a = pd.concat([a, extra], ignore_index=True)
    assert _summarize(a, protocol="origin", grain="fixture", weeks=[154])["status"] == "MEASURED"


def test_position_support_counts_and_missing_training_scale_are_explicit():
    a = pd.concat([_round(), _round().assign(player_id=[4, 5, 6], position="GK")], ignore_index=True)
    result = _summarize(a)
    summary = result["summaries"][0]["candidate"]
    assert set(summary["by_position"]) == {"GK", "MID"}
    assert summary["by_position"]["GK"]["player_rounds"] == 3
    assert summary["overall"]["ranking_groups"] == 2
    assert summary["overall"]["fixture_rows"] is None
    assert result["coverage_comparison"] == "EXACT_MATCH"
    assert result["original_model_coverage"] == "NOT ASSESSED"
    assert result["training_scaled_mase"]["status"] == "NOT MEASURED"
    assert "Training-only" in result["training_scaled_mase"]["reason"]
    assert "captaincy" in " ".join(result["notes"])


@pytest.mark.parametrize("column", ["player_id", "GW", "fixture", "origin_gw"])
@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, np.nan, np.inf])
def test_identity_fields_require_finite_positive_integers(column, bad):
    a = _round().assign(fixture=10, origin_gw=153)
    a[column] = a[column].astype(object)
    a.loc[0, column] = bad
    with pytest.raises(ValueError):
        _summarize(a, protocol="origin", grain="fixture")


@pytest.mark.parametrize("column", ["predicted_total_points", "actual_total_points"])
@pytest.mark.parametrize("bad", [True, np.nan, np.inf, "2"])
def test_forecast_and_outcome_values_must_be_numeric_finite_nonboolean(column, bad):
    a = _round()
    a[column] = a[column].astype(object)
    a.loc[0, column] = bad
    with pytest.raises(ValueError):
        _summarize(a)


@pytest.mark.parametrize("column", ["player_id", "GW", "position", "predicted_total_points", "actual_total_points"])
def test_required_fields_cannot_be_omitted(column):
    with pytest.raises(ValueError):
        _summarize(_round().drop(columns=column))


def test_protocol_grain_population_and_requested_weeks_are_explicit():
    with pytest.raises(ValueError):
        _summarize(_round(), protocol="unknown")
    with pytest.raises(ValueError):
        _summarize(_round(), grain="unknown")
    with pytest.raises(ValueError):
        summarize_forecast_pair(_round(), _round(), information_protocol="standard", grain="player_round",
                                gameweeks=[153], population=" ")
    with pytest.raises(ValueError):
        _summarize(_round(), weeks=[153, 154])
    with pytest.raises(ValueError):
        _summarize(_round().assign(position="BENCH"))
    with pytest.raises(ValueError):
        _summarize(_round().assign(origin_gw=153))
    with pytest.raises(ValueError):
        _summarize(_round(), protocol="origin")
    with pytest.raises(ValueError):
        _summarize(_round().assign(origin_gw=154), protocol="origin")
