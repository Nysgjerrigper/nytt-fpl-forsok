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
                    forecast_quality: dict[str, Any] | None = None,
                    settlement_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe an observed paired difference, without turning it into promotion.

    Requires the comparison receipt produced by compare_backtests.compare.
    Timing measures this reporting invocation, never the original model fit.
    Optional settlement evidence must match both compared CSV and receipt hashes;
    recorded solver/input claims do not certify planning validity or promotion.
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
    if settlement_evidence is not None:
        if ("settlement_replays" not in spec or settlement_evidence.get("schema_version") != 1 or
                settlement_evidence.get("status") != "VERIFIED_SETTLEMENT_INPUTS" or
                settlement_evidence.get("gameweeks") != comparison["gameweeks"] or
                settlement_evidence.get("metric") != comparison["metric"] or
                settlement_evidence.get("promotion_eligible") is not False or
                not settlement_evidence.get("checks") or
                any(value != "NOT ASSESSED" for value in settlement_evidence["checks"].values())):
            raise ValueError("settlement scorecard identity differs from comparison")
        for side, key in (("candidate", "a"), ("control", "b")):
            evidence = settlement_evidence["sides"][side]
            if (evidence["comparison_input_sha256"] != comparison["input_sha256"][key] or
                    evidence["receipt_sha256"] != spec["settlement_replays"][side]["sha256"]):
                raise ValueError("settlement scorecard bytes differ from declared inputs")
    elif "settlement_replays" in spec:
        raise ValueError("declared settlements require validated evidence")
    exhibition = spec["purpose"] == "framework_validation"
    contracted = comparison["evidence_status"] == "CONTRACT_VALIDATED"
    status = "EXHIBITION" if exhibition else ("CONTRACT_VALIDATED" if contracted else "UNVERIFIED")
    card = {
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

    if settlement_evidence is not None:
        card["settlement_evidence"] = deepcopy(settlement_evidence)
    return card


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
    if "settlement_evidence" in card:
        evidence = card["settlement_evidence"]
        lines += ["## Saved decision and settlement evidence", "",
                  "Both strict replay bundles were hash-verified at report generation. "
                  "This does not validate their planning inputs, native certificates or full site rules.", "",
                  "| Measured settlement | Candidate | Control |", "| --- | ---: | ---: |"]
        for key, title in (("gross_points", "Gross points"), ("hit_points", "Transfer hit points"),
                           ("net_points", "Net points"), ("decision_rounds", "Recorded decision rounds"),
                           ("transfers_in", "Recorded incoming players"), ("transfers_out", "Recorded outgoing players")):
            values = [evidence["sides"][side][key] for side in ("candidate", "control")]
            lines.append(f"| {title} | {values[0]:g} | {values[1]:g} |")
        lines += ["", "Incoming players include initial squad construction when captured as transfers. "
                  "Decision rounds are not an attempted-solve denominator.", "",
                  "| Evidence gate | Assessment |", "| --- | --- |"]
        for key, assessment in sorted(evidence["checks"].items()):
            lines.append(f"| {label(key.replace('_', ' '))} | {label(assessment)} |")
        def recorded_display(record: dict[str, Any]) -> str:
            if record.get("status") != "RECORDED":
                return "NOT MEASURED"
            values = record.get("values", record.get("decisions"))
            if values is None:
                return label(record.get("snapshot", "NOT MEASURED"))
            if values and all(value == values[0] for value in values):
                return f"{label(values[0])} ({len(values)} rounds)"
            return ", ".join(label(value) for value in values)

        lines += ["", "| Recorded claim | Candidate | Control |", "| --- | --- | --- |"]
        for field, title in (("solver_status", "Solver status"), ("mip_gap", "Relative solver gap"),
                             ("solver_seconds", "Solver seconds per recorded window"),
                             ("price_policy", "Planning price policy"), ("pool_policy", "Player pool policy"),
                             ("solver_outcome_columns", "Solver outcome columns")):
            displays = [recorded_display(evidence["sides"][side]["recorded"].get(field, {}))
                        for side in ("candidate", "control")]
            lines.append(f"| {title} | {displays[0]} | {displays[1]} |")
        for side in ("candidate", "control"):
            value = evidence["sides"][side]
            lines += ["", f"**{side.capitalize()} recorded evidence**", "",
                      "Chip usage: " + (", ".join(f"{label(chip)}: {count}" for chip, count in
                          sorted(value["chip_usage"].items())) or "none recorded") + ".",
                      f"Receipt SHA256: `{value['receipt_sha256']}`.",
                      f"Snapshot SHA256: `{value['snapshot_sha256']}`."]
        lines += ["", "Recorded Optimal status and zero gap do not substitute for exported native certificates. "
                  "Settlement blank evidence does not validate zero-filled planning forecasts. "
                  "Original fit runtime, attempted solves and failure rate remain NOT MEASURED. "
                  "Source replay bundles are required to audit the captured receipts again.", ""]
    lines += ["Probabilistic diagnostics: **NOT MEASURED**. "
             "Native optimality, independent legality and causal validity: **NOT ASSESSED**.", "",
             f"Reporting invocation: {card['operations']['reporting_seconds']:.3f} seconds. "
             "Original fit runtime and failure rate: **NOT MEASURED**.", "",
             f"**Promotion: NOT ASSESSED.** {card['verdict']['reason']}", "",
             "[Machine-readable scorecard](scorecard.json) · [Comparison receipt](comparison.json)", ""]
    return "\n".join(lines)
