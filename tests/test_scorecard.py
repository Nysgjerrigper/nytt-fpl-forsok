"""Scorecards retain measured outcomes and explicitly unassessed evidence."""

import json

import pytest

from fpl import playground
from fpl.scorecard import build_scorecard, render_scorecard


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
