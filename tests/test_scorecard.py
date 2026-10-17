"""Scorecards retain measured outcomes and explicitly unassessed evidence."""

import json

import pytest

from fpl import playground
from fpl.scorecard import build_scorecard, render_scorecard
from fpl.artifacts import sha256_file
import pandas as pd


def _forecast_demo(tmp_path):
    path = playground.init_demo(tmp_path / "demo")
    spec = json.loads(path.read_text())
    panel = pd.DataFrame([
        {"player_id": player, "GW": gw, "position": "MID", "actual_total_points": actual,
         "predicted_total_points": predicted}
        for gw in (153, 154, 155)
        for player, actual, predicted in ((101, 0, 1), (102, 2, 2), (103, 4, 3))
    ])
    forecasts = {"information_protocol": "standard", "grain": "player_round",
                 "population": "Synthetic complete MID pool"}
    for side in ("candidate", "control"):
        target = path.parent / f"{side}-forecasts.csv"
        panel.to_csv(target, index=False)
        forecasts[side] = {"path": target.name, "sha256": sha256_file(target)}
    spec["forecasts"] = forecasts
    path.write_text(json.dumps(spec))
    return path


def test_pinned_forecasts_are_captured_and_rendered_without_promotion(tmp_path):
    path = _forecast_demo(tmp_path)
    directory = playground.run_experiment(path, runs_dir=tmp_path / "runs")
    card = playground.read_run(directory)
    assert card["forecast_quality"]["status"] == "MEASURED"
    overall = card["forecast_quality"]["summaries"][0]["candidate"]["overall"]
    assert overall["mae"] == pytest.approx(2 / 3)
    assert overall["rmse"] == pytest.approx((2 / 3) ** 0.5)
    report = (directory / "REPORT.md").read_text()
    assert report == render_scorecard(card)
    assert "| MAE | 0.6667 | 0.6667 |" in report
    assert "MASE: **NOT MEASURED**" in report
    assert card["verdict"]["promotion_eligible"] is False
    for side in ("candidate", "control"):
        captured = directory / f"{side}-forecasts.csv"
        assert sha256_file(captured) == card["forecast_quality"]["input_sha256"][side]
    # Completed reports remain readable after external inputs are removed.
    for source in path.parent.glob("*.csv"):
        source.unlink()
    assert playground.read_run(directory) == card


def test_forecast_output_corruption_and_changed_input_are_rejected(tmp_path):
    path = _forecast_demo(tmp_path)
    directory = playground.run_experiment(path, runs_dir=tmp_path / "runs")
    captured = directory / "candidate-forecasts.csv"
    captured.write_bytes(captured.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        playground.read_run(directory)
    source = path.parent / "candidate-forecasts.csv"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input bytes changed"):
        playground.validate_spec(path)


def test_different_forecast_coverage_is_rejected_before_creating_run(tmp_path):
    path = _forecast_demo(tmp_path)
    spec = json.loads(path.read_text())
    source = path.parent / "candidate-forecasts.csv"
    panel = pd.read_csv(source).iloc[1:]
    panel.to_csv(source, index=False)
    spec["forecasts"]["candidate"]["sha256"] = sha256_file(source)
    path.write_text(json.dumps(spec))
    runs = tmp_path / "runs"
    with pytest.raises(ValueError):
        playground.run_experiment(path, runs_dir=runs)
    assert not runs.exists()


def test_gross_and_net_cards_use_exact_comparison_values_and_unknown_diagnostics(tmp_path):
    path = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    net = playground.read_run(playground.run_experiment(path, runs_dir=root))
    spec = json.loads(path.read_text())
    spec["metric"] = "actual_total_points"
    path.write_text(json.dumps(spec))
    gross = playground.read_run(playground.run_experiment(path, attempt="gross", runs_dir=root))
    for card, candidate, control, difference in ((net, 171, 170, 1), (gross, 179, 170, 9)):
        outcomes = card["decision_outcomes"]
        assert (outcomes["total_a"], outcomes["total_b"], outcomes["total_diff"]) == (candidate, control, difference)
        assert card["forecast_quality"]["status"] == "NOT MEASURED"
        assert card["probabilistic_quality"]["status"] == "NOT MEASURED"
        assert card["operations"]["original_fit_runtime"] == "NOT MEASURED"
        assert card["operations"]["native_optimality"] == "NOT ASSESSED"
        assert card["checks"]["promotion"] == "NOT ASSESSED"
        assert card["verdict"]["promotion_eligible"] is False
        report = render_scorecard(card)
        assert f"| Candidate points | {candidate:.3f} |" in report
        assert f"| Control points | {control:.3f} |" in report
        assert f"| Difference | {difference:+.3f} |" in report
        assert f"[{outcomes['ci_low']:+.3f}, {outcomes['ci_high']:+.3f}]" in report
        assert "NOT MEASURED" in report and "Promotion: NOT ASSESSED" in report
        json.dumps(card, allow_nan=False)


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_scorecard_rejects_invalid_reporting_runtime(seconds):
    with pytest.raises(ValueError, match="reporting_seconds"):
        build_scorecard({}, {}, reporting_seconds=seconds)


def test_scorecard_rejects_metric_mismatch():
    with pytest.raises(ValueError, match="metric"):
        build_scorecard({"metric": "actual_net_points"}, {"metric": "actual_total_points"}, reporting_seconds=0)
