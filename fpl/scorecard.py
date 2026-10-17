"""Conservative decision scorecards from paired backtest comparison receipts.

Forecast diagnostics require separately pinned, matching forecast panels. Neither
adapter infers original fitting runtime, solver proof or promotion eligibility.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


def build_scorecard(spec: dict[str, Any], comparison: dict[str, Any], *,
                    reporting_seconds: float,
                    forecast_quality: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe an observed paired difference, without turning it into promotion.

    Requires the comparison receipt produced by compare_backtests.compare.
    Timing measures this reporting invocation, never the original model fit.
    """
    if not math.isfinite(reporting_seconds) or reporting_seconds < 0:
        raise ValueError("reporting_seconds must be finite and nonnegative")
    if spec["metric"] != comparison["metric"]:
        raise ValueError("scorecard metric differs from comparison")
    if forecast_quality is not None:
        if ("forecasts" not in spec or forecast_quality.get("status") != "MEASURED" or
                forecast_quality.get("metric_version") != "forecast_player_round_v1" or
                forecast_quality.get("gameweeks") != comparison["gameweeks"] or
                forecast_quality.get("information_protocol") != spec["forecasts"]["information_protocol"] or
                forecast_quality.get("input_sha256") != {
                    side: spec["forecasts"][side]["sha256"] for side in ("candidate", "control")}):
            raise ValueError("forecast scorecard identity differs from declared comparison")
    elif "forecasts" in spec:
        raise ValueError("declared forecasts require validated diagnostics")
    exhibition = spec["purpose"] == "framework_validation"
    contracted = comparison["evidence_status"] == "CONTRACT_VALIDATED"
    status = "EXHIBITION" if exhibition else ("CONTRACT_VALIDATED" if contracted else "UNVERIFIED")
    return {
        "schema_version": 1, "metric_version": "paired_weekly_v1",
        "experiment": {"id": spec["id"], "purpose": spec["purpose"], "hypothesis": spec["hypothesis"],
                       "candidate": spec["candidate"]["name"], "control": spec["control"]["name"]},
        "status": status,
        "decision_outcomes": comparison,
        "forecast_quality": (deepcopy(forecast_quality) if forecast_quality is not None else
                             {"status": "NOT MEASURED", "reason": "No validated forecast panel supplied."}),
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
             ]
    forecast = card["forecast_quality"]
    if forecast["status"] == "MEASURED":
        lines += [f"Forecast diagnostics: **MEASURED** ({forecast['metric_version']}).",
                  f"Population: {label(forecast['population'])}.",
                  f"Protocol: {forecast['information_protocol']}; declared grain: {forecast['grain']}. "
                  "Candidate/control coverage matches exactly.",
                  f"Forecast-to-decision input linkage: **{forecast['decision_input_linkage']}**. "
                  "A hash-bound declaration alone does not prove how decisions were produced.", ""]
        for summary in forecast["summaries"]:
            lead = summary["lead"]
            lines += ["**Standard forecasts**" if lead is None else
                      f"**Origin forecasts: lead {lead} (target GW minus origin GW)**", "",
                      "| Diagnostic | Candidate | Control |", "| --- | ---: | ---: |"]
            for key, title in (("player_rounds", "Player-round forecasts"), ("fixture_rows", "Fixture rows"),
                               ("mae", "MAE"), ("rmse", "RMSE"), ("bias", "Bias (prediction minus actual)"),
                               ("total_calibration", "Aggregate calibration ratio"), ("spearman", "Within-group Spearman"),
                               ("top1_capture", "Top-ranked player capture ratio"),
                               ("spearman_used_groups", "Spearman groups used"),
                               ("spearman_skipped_groups", "Spearman groups skipped"),
                               ("tied_top_groups", "Groups with tied top forecasts")):
                values = [summary[side]["overall"][key] for side in ("candidate", "control")]
                display = ["NOT MEASURED / undefined" if value is None else f"{value:.4f}" for value in values]
                lines.append(f"| {title} | {display[0]} | {display[1]} |")
            lines += ["", "| Position | Candidate rows | Control rows | Candidate MAE | Control MAE |",
                      "| --- | ---: | ---: | ---: | ---: |"]
            for position in sorted(summary["candidate"]["by_position"]):
                candidate = summary["candidate"]["by_position"][position]
                control = summary["control"]["by_position"][position]
                lines.append(f"| {position} | {candidate['player_rounds']} | {control['player_rounds']} | "
                             f"{candidate['mae']:.4f} | {control['mae']:.4f} |")
            lines.append("")
        lines += ["MASE: **NOT MEASURED**; training-only scales and refit cutoffs were not supplied. "
                  "Original model coverage: **NOT ASSESSED**.",
                  "Top-ranked ties use the lowest stable player code. Capture is a ranking diagnostic, "
                  "not realized captaincy points. Zero-denominator ratios are undefined; negative outcomes "
                  "can produce ratios outside [0, 1]. Origin leads are reported separately, without "
                  "pooling repeated target outcomes.", ""]
    else:
        lines += ["Forecast diagnostics: **NOT MEASURED**.", ""]
    lines += ["Probabilistic diagnostics: **NOT MEASURED**. "
             "Native optimality, independent legality and causal validity: **NOT ASSESSED**.", "",
             f"Reporting invocation: {card['operations']['reporting_seconds']:.3f} seconds. "
             "Original fit runtime and failure rate: **NOT MEASURED**.", "",
             f"**Promotion: NOT ASSESSED.** {card['verdict']['reason']}", "",
             "[Machine-readable scorecard](scorecard.json) · [Comparison receipt](comparison.json)", ""]
    return "\n".join(lines)
