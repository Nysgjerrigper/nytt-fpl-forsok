"""Conservative decision scorecards from paired backtest comparison receipts.

This first adapter reports saved squad outcomes only. It cannot infer forecast
diagnostics, original fitting runtime, solver proof or promotion eligibility.
"""
from __future__ import annotations

import math
from typing import Any


def build_scorecard(spec: dict[str, Any], comparison: dict[str, Any], *,
                    reporting_seconds: float) -> dict[str, Any]:
    """Describe an observed paired difference, without turning it into promotion.

    Requires the comparison receipt produced by compare_backtests.compare.
    Timing measures this reporting invocation, never the original model fit.
    """
    if not math.isfinite(reporting_seconds) or reporting_seconds < 0:
        raise ValueError("reporting_seconds must be finite and nonnegative")
    if spec["metric"] != comparison["metric"]:
        raise ValueError("scorecard metric differs from comparison")
    exhibition = spec["purpose"] == "framework_validation"
    contracted = comparison["evidence_status"] == "CONTRACT_VALIDATED"
    status = "EXHIBITION" if exhibition else ("CONTRACT_VALIDATED" if contracted else "UNVERIFIED")
    return {
        "schema_version": 1, "metric_version": "paired_weekly_v1",
        "experiment": {"id": spec["id"], "purpose": spec["purpose"], "hypothesis": spec["hypothesis"],
                       "candidate": spec["candidate"]["name"], "control": spec["control"]["name"]},
        "status": status,
        "decision_outcomes": comparison,
        "forecast_quality": {"status": "NOT MEASURED", "reason": "No validated forecast panel supplied."},
        "probabilistic_quality": {"status": "NOT MEASURED", "reason": "No calibrated distribution supplied."},
        "operations": {"reporting_seconds": reporting_seconds, "original_fit_runtime": "NOT MEASURED",
                       "failure_rate": "NOT MEASURED", "native_optimality": "NOT ASSESSED",
                       "independent_legality": "NOT ASSESSED"},
        "checks": {"comparability": "PASS" if contracted else "NOT ASSESSED",
                   "causal_validity": "NOT ASSESSED", "promotion": "NOT ASSESSED"},
        "verdict": {"disposition": "framework_validation" if exhibition else "historical_reanalysis",
                    "promotion_eligible": False,
                    "reason": "Exhibition results cannot support model selection." if exhibition else
                    "Saved-outcome reanalysis only; full registered promotion evidence has not been assessed."},
    }


def render_scorecard(card: dict[str, Any]) -> str:
    """Render the same measured JSON values and explicit unknowns for a manager."""
    experiment, result = card["experiment"], card["decision_outcomes"]
    # Inline labels cannot introduce additional Markdown rows or formatting blocks.
    def label(value: Any) -> str:
        return str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")

    lines = ["# The Training Ground — scorecard", "",
             f"Experiment: **{label(experiment['id'])}**. Status: **{card['status']}**.", "",
             f"Hypothesis: {label(experiment['hypothesis'])}", "",
             f"Candidate: {label(experiment['candidate'])}; control: {label(experiment['control'])}.", "",
             f"Metric: `{result['metric']}`; gameweeks: {result['gameweeks']}.", "",
             "| Measured outcome | Value |", "| --- | ---: |",
             f"| Candidate points | {result['total_a']:.3f} |",
             f"| Control points | {result['total_b']:.3f} |",
             f"| Difference | {result['total_diff']:+.3f} |",
             f"| Paired 95% interval | [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}] |",
             f"| Positive bootstrap fraction | {result['bootstrap_positive_fraction']:.3f} |",
             f"| One-sided sign-test p-value | {result['sign_test']['p_a_better']:.5f} |", "",
             "The bootstrap fraction is not a probability of future superiority. "
             "The sign test assumes independent signs; block choice affects the interval.", "",
             f"Comparability: **{card['checks']['comparability']}**. "
             "Contract validation checks bytes and declared assumptions only.", "",
             "Forecast and probabilistic diagnostics: **NOT MEASURED**. "
             "Native optimality, independent legality and causal validity: **NOT ASSESSED**.", "",
             f"Reporting invocation: {card['operations']['reporting_seconds']:.3f} seconds. "
             "Original fit runtime and failure rate: **NOT MEASURED**.", "",
             f"**Promotion: NOT ASSESSED.** {card['verdict']['reason']}", "",
             "[Machine-readable scorecard](scorecard.json) · [Comparison receipt](comparison.json)", ""]
    return "\n".join(lines)
