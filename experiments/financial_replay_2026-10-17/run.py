"""Registered no-solve financial accounting and fixed-decision price sensitivity."""
from io import BytesIO
import json
from pathlib import Path
import sys

import pandas as pd

from fpl import experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.financial_audit import audit_legacy_finances
from fpl.milp.replay import read_replay
from fpl.research_runner import _source_hashes, _runtime_versions, _verify_outputs, read_result


def price_evidence(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Reconstruct the evaluated legacy price assumptions, not deadline-known prices."""
    source = frame.copy()
    source["value"] = pd.to_numeric(source.value)
    source["raw_missing_default"] = source.value.isna()
    source["value"] = source.value.fillna(50.0)
    keys = ["player_id", "GW"] + (["origin_gw"] if "origin_gw" in source else [])
    disagreements = source.groupby(["player_id", "GW"]).value.nunique(dropna=False).gt(1)
    grouped = source.groupby(keys, as_index=False).agg(value=("value", "first"),
        raw_missing_default=("raw_missing_default", "first"))
    grouped = grouped.loc[grouped.GW.between(max(1, start - 1), end)]
    # groupby above reproduces the original sorted-key/first-origin policy.
    observed = grouped.drop_duplicates(["player_id", "GW"]).set_index(["player_id", "GW"])
    weeks = sorted(grouped.GW.unique())
    rows = []
    for player, group in grouped.groupby("player_id", sort=True):
        available = sorted(set(group.GW))
        for week in weeks:
            if week in available:
                source_week, basis = week, "OBSERVED_FIRST_COPY"
            elif any(value < week for value in available):
                source_week, basis = max(value for value in available if value < week), "FORWARD_FILL"
            else:
                source_week, basis = min(available), "BACKWARD_FILL"
            selected = observed.loc[(player, source_week)]
            rows.append({"player_id": int(player), "GW": int(week), "value": selected.value,
                "price_basis": basis, "source_GW": int(source_week),
                "raw_missing_default": bool(selected.raw_missing_default),
                "price_copies_disagree": bool(disagreements.loc[(player, source_week)])})
    return pd.DataFrame(rows)


def main(plan_path: Path, output: Path) -> None:
    plan = json.loads(plan_path.read_bytes())
    if (plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions()
            or plan["script_sha256"] != sha256_file(Path(__file__))):
        raise ValueError("registered source/runtime/script changed")
    original = Path(plan["original_bundle"])
    read_result(original)
    if sha256_file(original / "complete.json") != plan["original_receipt_sha256"]:
        raise ValueError("original pilot receipt changed")
    recovery = Path(plan["recovery_bundle"])
    for name, digest in plan["recovery_input_sha256"].items():
        if sha256_file(recovery / name) != digest:
            raise ValueError("source-derived recovery evidence changed")
    recovered = json.loads((recovery / "recovery.json").read_bytes())
    if any(plan[key] != recovered[key] for key in ("initial_state", "ft_cap", "ft_gain")):
        raise ValueError("audit state/rules differ from source-bound recovery")
    for case in plan["cases"]:
        for name, digest in case["input_sha256"].items():
            if sha256_file(original / name) != digest:
                raise ValueError("registered original input changed")
        settlement = Path(plan["settlement_bundle"]) / case["id"]
        if sha256_file(settlement / "complete.json") != plan["settlement_receipt_sha256"][case["id"]]:
            raise ValueError("registered settlement receipt changed")
        receipt = read_replay(settlement)
        if (receipt["input_sha256"]["decisions.json"] != case["input_sha256"][case["decisions"]]
                or receipt["input_sha256"]["predictions.csv"] != case["input_sha256"][case["predictions"]]
                or receipt["reference_reconciliation"] != "PASS"):
            raise ValueError("settlement evidence differs from financial audit inputs")
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "plan.json", plan)
    write_json_new(output / "started.json", {"status": "STARTED", "plan_sha256": sha256_file(plan_path),
                   "source_sha256": plan["source_sha256"], "promotion_eligible": False})
    results = {}
    for case in plan["cases"]:
        values = pd.read_csv(BytesIO((original / case["predictions"]).read_bytes()))
        prices = price_evidence(values, plan["window"]["start_gw"], plan["window"]["end_gw"])
        snapshot = json.loads((original / case["decisions"]).read_bytes())
        used = {(player, row["gameweek"]) for row in snapshot["decisions"]
                for player in set(row["squad"]) | set(row["transfers_out"])}
        used_prices = prices.loc[[pair in used for pair in zip(prices.player_id, prices.GW)]]
        for policy in ("current_price", "half_profit"):
            key = case["id"] + "-" + policy
            target = output / key
            target.mkdir()
            (target / "decisions.json").write_bytes((original / case["decisions"]).read_bytes())
            prices.to_csv(target / "prices.csv", index=False)
            write_json_new(target / "started.json", {"status": "STARTED", "plan_sha256": sha256_file(plan_path),
                "source_sha256": plan["source_sha256"], "promotion_eligible": False})
            try:
                audited = audit_legacy_finances(json.loads((target / "decisions.json").read_bytes()), prices,
                    initial_state=plan["initial_state"], ft_cap=plan["ft_cap"], ft_gain=plan["ft_gain"],
                    selling_policy=policy, compare_reported=policy == "current_price")
                audited.to_csv(target / "audit.csv", index=False)
                results[key] = {"status": "FINANCIAL_ARITHMETIC_PASS" if policy == "current_price" else "FIXED_DECISION_SENSITIVITY_COMPLETED",
                    "rounds": len(audited), "final_derived_bank": float(audited.derived_bank_end.iloc[-1]),
                    "final_reported_bank": float(audited.reported_budget_end.iloc[-1]),
                    "final_bank_difference": float(audited.bank_end_difference.iloc[-1]),
                    "used_price_cells": len(used_prices),
                    "used_price_evidence": used_prices.price_basis.value_counts().to_dict(),
                    "used_prices_sourced_from_default50": int(used_prices.raw_missing_default.sum()),
                    "used_prices_with_disagreeing_copies": int(used_prices.price_copies_disagree.sum()),
                    "promotion_eligible": False}
                receipt = dict(results[key], output_sha256={p.name: sha256_file(p) for p in sorted(target.iterdir())})
                write_json_new(target / "complete.json", receipt)
                _verify_outputs(target, receipt["output_sha256"], {"complete.json"})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                results[key] = {"status": "FAILED", "reason": str(exc), "promotion_eligible": False}
                write_json_new(target / "failed.json", results[key])
            experiment.log_result(key, {"plan_sha256": sha256_file(plan_path), "scope": "no_chip_financial_arithmetic"},
                results[key], results_path=output / "registry.csv")
    if plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions():
        raise ValueError("source/runtime changed during audit")
    read_result(original)
    for name, digest in plan["recovery_input_sha256"].items():
        if sha256_file(recovery / name) != digest:
            raise ValueError("recovery evidence changed during audit")
    result = {"status": "FINANCIAL_MATRIX_COMPLETED", "plan_sha256": sha256_file(plan_path), "model_fits": 0, "optimizer_solves": 0,
        "cases": results, "promotion_eligible": False, "initial_state_provenance": "SOURCE_DERIVED",
        "full_site_rule_legality": "NOT ASSESSED", "outcome_blind_execution": "NOT ASSESSED",
        "limitation": "No-chip historical accounting under reconstructed price assumptions. Half-profit is a fixed-decision sensitivity, not reoptimization or deadline-known price proof. Imputation and original access limits remain explicit."}
    lines = ["# Financial audit — preserved pilot", "",
        "Initial state is recovered from exact evaluated source bytes, not original explicit plan fields.", "",
        "| Case / policy | Status | Derived closing bank (£m) | Reported bank (£m) | Difference (£m) |",
        "| --- | --- | ---: | ---: | ---: |"]
    for name, row in results.items():
        if row["status"] == "FAILED":
            lines.append(f"| {name} | FAILED | — | — | — |")
        else:
            lines.append(f"| {name} | {row['status']} | {row['final_derived_bank'] / 10:.1f} | "
                         f"{row['final_reported_bank'] / 10:.1f} | {row['final_bank_difference'] / 10:+.1f} |")
    lines += ["", result["limitation"], "",
        "Current-price rows check the declared legacy arithmetic. Half-profit rows retain the same decisions and FT policy; they do not represent newly optimized performance.",
        "", "Full site-rule legality and outcome-blind execution: NOT ASSESSED. No ranking or promotion."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    result["output_sha256"] = {str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*")) if p.is_file()}
    write_json_new(output / "result.json", result)
    _verify_outputs(output, result["output_sha256"], {"result.json"})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
