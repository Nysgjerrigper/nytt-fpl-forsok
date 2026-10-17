"""Bounded opt-in comparisons over existing forecasting, MILP and scorecard APIs.

Plans register complete inputs before execution. Results are diagnostic historical
reanalyses, never fresh holdouts, independent legality certificates or promotions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from typing import Any

import pandas as pd

from fpl import config, experiment, features
from fpl.artifacts import sha256_file, write_json_new
from fpl.data.provenance import load_frozen_research_dataset
from fpl.forecast_scorecard import summarize_forecast_pair
from fpl.milp import optimize
from fpl.model import models, predict
from fpl.model.moe_tournament import protocol_cutoffs
from fpl.model.recipes import resolve_fit_recipe
from fpl.playground import read_run, run_experiment

_MODELS = {"catboost_mae", "catboost_rmse"}
_PROTOCOLS = ("standard", "origin")
_ARMS = ("control", "candidate")
_PLAN_KEYS = {"schema_version", "window", "seed", "dataset_sha256", "source_sha256", "runtime", "arms", "budget", "promotion_eligible"}


def _read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("JSON must be an object")
    json.dumps(value, allow_nan=False)
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(config.ROOT)): sha256_file(path)
            for path in sorted((config.ROOT / "fpl").rglob("*.py"))}


def _runtime_versions() -> dict[str, str]:
    packages = {name: name for name in ("catboost", "lightgbm", "xgboost", "numpy", "pandas", "scipy", "pulp", "highspy")}
    packages["sklearn"] = "scikit-learn"
    return {"python": platform.python_version(), **{label: version(package) for label, package in packages.items()}}


def _window(start_gw: int, end_gw: int, retrain_every: int, horizon: int) -> dict:
    start, end = _integer(start_gw, "start_gw"), _integer(end_gw, "end_gw")
    retrain, lookahead = _integer(retrain_every, "retrain_every"), _integer(horizon, "horizon")
    cutoffs = protocol_cutoffs()
    phase = None
    if cutoffs["selection_min_gw"] <= start <= end <= cutoffs["selection_max_gw"]:
        phase = "selection_diagnostic"
    elif cutoffs["backtest_min_gw"] <= start <= end <= cutoffs["backtest_max_gw"]:
        phase = "historical_evaluation"
    if phase is None:
        raise ValueError("window must lie wholly inside the registered selection or historical evaluation range")
    return {"start_gw": start, "end_gw": end, "retrain_every": retrain, "horizon": lookahead, "phase": phase}


def _load_frame(dataset_path: str | Path, window: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_frozen_research_dataset(dataset_path)
    raw = raw.loc[raw.GW_global <= window["end_gw"]].copy()
    raw.attrs.pop("frozen_dataframe_fingerprint", None)
    raw.attrs["research_max_gw"] = window["end_gw"]
    expected = set(range(window["start_gw"], window["end_gw"] + 1))
    if set(raw.loc[raw.GW_global >= window["start_gw"], "GW_global"]) != expected:
        raise ValueError("requested rounds require complete contiguous frozen-data coverage")
    history = raw.loc[raw.GW_global < window["start_gw"]]
    if history.empty or set(history.position.unique()) != set(config.ONFIELD_POSITIONS):
        raise ValueError("pre-window training history must cover all four positions")
    frame = features.build_feature_frame(raw)
    before = frame.loc[frame.GW_global < window["start_gw"]]
    if before.empty or set(before.position.unique()) != set(config.ONFIELD_POSITIONS):
        raise ValueError("feature training history must cover all four positions")
    return raw, frame


def _arm(model: str, canonical: list[str], drop: Any, common: dict, seed: int) -> dict:
    if model not in _MODELS:
        raise ValueError("runner supports only catboost_mae and catboost_rmse")
    if not isinstance(drop, (tuple, list)) or any(not isinstance(name, str) for name in drop) or len(set(drop)) != len(drop) or not set(drop) <= set(canonical):
        raise ValueError("drop features must be unique existing canonical feature names")
    selected = [name for name in canonical if name not in drop]
    if not selected:
        raise ValueError("an arm must retain at least one canonical feature")
    weights = {position: {model: 1.0} for position in config.ONFIELD_POSITIONS}
    params = {position: {model: deepcopy(common)} for position in config.ONFIELD_POSITIONS}
    return {"model": model, "features": selected, "params_by_pos": resolve_fit_recipe(weights, params, seed)}


def create_plan(dataset_path: str | Path, *, start_gw: int, end_gw: int,
                retrain_every: int = 4, horizon: int = 3, seed: int = 0,
                parameters: dict | None = None, control_model: str = "catboost_mae",
                candidate_model: str = "catboost_rmse", control_drop: tuple = (),
                candidate_drop: tuple = (), max_seconds: float | None = None,
                solve_seconds: float | None = None) -> dict:
    """Register a bounded exact-input contrast without fitting or solving models.

    Historical evaluation is already observed, not a fresh holdout. A contrast may
    change the registered model or its ordered feature subset, never both together.
    """
    window = _window(start_gw, end_gw, retrain_every, horizon)
    seed = _integer(seed, "seed", 0)
    maximum = _seconds(config.RESEARCH_RUN_MAX_SECONDS if max_seconds is None else max_seconds, "max_seconds")
    solve = _seconds(config.RESEARCH_SOLVE_SECONDS if solve_seconds is None else solve_seconds, "solve_seconds")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("shared parameters must be a JSON object")
    common = {**config.RESEARCH_COMPARISON_PARAMS, **({} if parameters is None else parameters)}
    _, frame = _load_frame(dataset_path, window)
    canonical = features.feature_columns(frame)
    arms = {"control": _arm(control_model, canonical, control_drop, common, seed),
            "candidate": _arm(candidate_model, canonical, candidate_drop, common, seed)}
    if control_model != candidate_model and arms["control"]["features"] != arms["candidate"]["features"]:
        raise ValueError("a contrast cannot change both model and feature list")
    rounds = window["end_gw"] - window["start_gw"] + 1
    return {"schema_version": 1, "window": window, "seed": seed,
            "dataset_sha256": sha256_file(dataset_path), "source_sha256": _source_hashes(),
            "runtime": _runtime_versions(), "arms": arms,
            "budget": {"max_seconds": maximum, "solve_seconds": solve,
                       "max_fit_calls": 16 * math.ceil(rounds / window["retrain_every"]), "max_solver_calls": 4 * rounds},
            "promotion_eligible": False}


def _validate_plan(plan: dict, dataset_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (set(plan) != _PLAN_KEYS or type(plan["schema_version"]) is not int or plan["schema_version"] != 1
            or plan["promotion_eligible"] is not False):
        raise ValueError("invalid plan schema or promotion claim")
    json.dumps(plan, allow_nan=False)
    window = plan["window"]
    if not isinstance(window, dict) or set(window) != {"start_gw", "end_gw", "retrain_every", "horizon", "phase"}:
        raise ValueError("invalid plan window")
    if window != _window(window["start_gw"], window["end_gw"], window["retrain_every"], window["horizon"]):
        raise ValueError("window phase disagrees with registered protocol")
    seed = _integer(plan["seed"], "seed", 0)
    budget = plan["budget"]
    if not isinstance(budget, dict) or set(budget) != {"max_seconds", "solve_seconds", "max_fit_calls", "max_solver_calls"}:
        raise ValueError("invalid compute budget")
    for name in ("max_seconds", "solve_seconds"):
        _seconds(budget[name], name)
    rounds = window["end_gw"] - window["start_gw"] + 1
    if (_integer(budget["max_fit_calls"], "max_fit_calls") != 16 * math.ceil(rounds / window["retrain_every"])
            or _integer(budget["max_solver_calls"], "max_solver_calls") != 4 * rounds):
        raise ValueError("fit/solve budgets disagree with declared schedule")
    if plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions():
        raise ValueError("source or runtime differs from registered plan")
    if sha256_file(dataset_path) != plan["dataset_sha256"]:
        raise ValueError("dataset bytes differ from registered plan")
    raw, frame = _load_frame(dataset_path, window)
    canonical = features.feature_columns(frame)
    arms = plan["arms"]
    if not isinstance(arms, dict) or set(arms) != set(_ARMS):
        raise ValueError("plan requires exactly control and candidate")
    for arm in arms.values():
        if not isinstance(arm, dict) or set(arm) != {"model", "features", "params_by_pos"} or arm["model"] not in _MODELS:
            raise ValueError("invalid registered arm")
        selected = arm["features"]
        if (not isinstance(selected, list) or not selected or any(not isinstance(name, str) for name in selected)
                or len(set(selected)) != len(selected) or selected != [name for name in canonical if name in selected]):
            raise ValueError("features must preserve a nonempty ordered canonical subset")
        weights = {position: {arm["model"]: 1.0} for position in config.ONFIELD_POSITIONS}
        if resolve_fit_recipe(weights, arm["params_by_pos"], seed) != arm["params_by_pos"]:
            raise ValueError("plan parameters must contain complete resolved defaults")
    if arms["control"]["model"] != arms["candidate"]["model"] and arms["control"]["features"] != arms["candidate"]["features"]:
        raise ValueError("a contrast cannot change both model and feature list")
    for position in config.ONFIELD_POSITIONS:
        settings = []
        for arm in arms.values():
            params = arm["params_by_pos"][position][arm["model"]]
            settings.append({key: value for key, value in params.items() if key != "loss_function"})
        if settings[0] != settings[1]:
            raise ValueError("control and candidate must use matched non-loss parameters")
    return raw, frame


def _failure(root: Path, reason: str, kind: str) -> None:
    if not (root / "failed.json").exists():
        write_json_new(root / "failed.json", {"status": "FAILED", "type": kind, "reason": reason, "promotion_eligible": False})


def run_plan(plan_path: str | Path, dataset_path: str | Path, output_dir: str | Path) -> Path:
    """Validate and capture inputs, then run one fixed worker under a wall-clock cap."""
    plan_path, dataset_path, root = Path(plan_path), Path(dataset_path), Path(output_dir).resolve()
    content = plan_path.read_bytes()
    plan = json.loads(content)
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    _validate_plan(plan, dataset_path)
    plan_sha256 = hashlib.sha256(content).hexdigest()
    if any(root.is_relative_to(folder.resolve()) for folder in (config.DATASETT_DIR, config.ROOT / "fpl")):
        raise ValueError("research output cannot target production code or datasets")
    root.mkdir(parents=True, exist_ok=False)
    try:
        with (root / "plan.json").open("xb") as stream:
            stream.write(content)
        shutil.copyfile(dataset_path, root / "dataset.csv")
        if sha256_file(root / "dataset.csv") != plan["dataset_sha256"]:
            raise ValueError("dataset changed during capture")
        write_json_new(root / "started.json", {"schema_version": 1, "plan_sha256": plan_sha256,
                                               "status": "STARTED", "fresh_holdout": False, "promotion_eligible": False})
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(config.ROOT)
        with (root / "worker.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-m", "fpl.research_runner", "_worker", str(root)],
                                    cwd=config.ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=plan["budget"]["max_seconds"], check=False)
        if result.returncode != 0:
            raise RuntimeError(f"research worker exited with status {result.returncode}")
        if (root / "failed.json").exists():
            raise RuntimeError("research worker preserved a failure")
        receipt = _read_json(root / "worker-complete.json")
        if (receipt.get("status") != "COMPLETED" or receipt.get("schema_version") != 1
                or receipt.get("promotion_eligible") is not False or receipt.get("fresh_holdout") is not False
                or receipt.get("plan_sha256") != plan_sha256
                or receipt.get("plan_sha256") != sha256_file(root / "plan.json")):
            raise ValueError("worker completion disagrees with captured plan")
        _verify_outputs(root, receipt.get("outputs"), {"worker-complete.json", "worker.log"})
        if plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions():
            raise RuntimeError("source or runtime changed before completion")
        # subprocess.run has reaped the child and the log stream is closed. Native
        # shutdown output is now included, rather than invalidating a child-made hash.
        receipt["outputs"] = {**receipt["outputs"],
                              "worker-complete.json": sha256_file(root / "worker-complete.json"),
                              "worker.log": sha256_file(root / "worker.log")}
        write_json_new(root / "complete.json", receipt)
        read_result(root)
    except Exception as exc:
        _failure(root, str(exc), type(exc).__name__)
        raise
    return root


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _worker(output_dir: str | Path) -> None:
    """Execute captured recipes with measured fit receipts and native-optimal solves."""
    root = Path(output_dir).resolve()
    original_fit = models.fit_model
    state = {"fit_calls": 0, "fit_seconds": 0.0, "solver_calls": 0, "solver_seconds": 0.0}
    started = time.monotonic()
    try:
        plan = _read_json(root / "plan.json")
        feature_start = time.monotonic()
        raw, frame = _validate_plan(plan, root / "dataset.csv")
        feature_seconds = time.monotonic() - feature_start
        window, budget = plan["window"], plan["budget"]
        weeks = list(range(window["start_gw"], window["end_gw"] + 1))
        def measured_fit(name, X, y, position=None, minutes=None, gw=None, *, params=None, seed=None):
            if state["fit_calls"] >= budget["max_fit_calls"]:
                raise RuntimeError("registered fit-call budget exhausted")
            state["fit_calls"] += 1
            sequence = state["fit_calls"]
            record = {"model": name, "position": position, "protocol": state["protocol"], "arm": state["arm"],
                      "features": list(X.columns), "params": params, "seed": seed,
                      "train_min_gw": int(gw.min()), "train_max_gw": int(gw.max()), "training_rows": len(X)}
            write_json_new(root / f"fit-start-{sequence}.json", record)
            fit_start = time.monotonic()
            fitted = original_fit(name, X, y, position=position, minutes=minutes, gw=gw, params=params, seed=seed)
            seconds = time.monotonic() - fit_start
            state["fit_seconds"] += seconds
            complete = {**record, "seconds": seconds}
            if hasattr(fitted, "get_all_params"):
                complete["fitted_params"] = fitted.get_all_params()
            write_json_new(root / f"fit-complete-{sequence}.json", complete)
            return fitted
        panels = {}
        models.fit_model = measured_fit
        try:
            for protocol in _PROTOCOLS:
                for side in _ARMS:
                    state["protocol"], state["arm"] = protocol, side
                    arm = plan["arms"][side]
                    common = {"retrain_every": window["retrain_every"], "weight_strategy": "single:" + arm["model"],
                              "params_by_pos": arm["params_by_pos"], "seed": plan["seed"]}
                    if protocol == "standard":
                        panel = predict.walk_forward_predictions(frame, arm["features"], window["start_gw"], window["end_gw"], **common)
                    else:
                        panel = predict.origin_based_predictions(frame, raw, arm["features"], window["start_gw"], window["end_gw"], window["horizon"], **common)
                    panels[(protocol, side)] = panel
                    _write_csv(root / f"{protocol}-{side}-predictions.csv", panel)
        finally:
            models.fit_model = original_fit
        population = "Common frozen forecast pool; original unavailable-player coverage is not assessed"
        for protocol in _PROTOCOLS:
            summary = summarize_forecast_pair(panels[(protocol, "candidate")], panels[(protocol, "control")],
                        information_protocol=protocol, grain="fixture", gameweeks=weeks, population=population)
            write_json_new(root / f"{protocol}-forecast-summary.json", summary)
        comparisons = {}
        for protocol in _PROTOCOLS:
            for side in _ARMS:
                if state["solver_calls"] + len(weeks) > budget["max_solver_calls"]:
                    raise RuntimeError("registered solver-call budget exhausted")
                state["solver_calls"] += len(weeks)
                solver_start = time.monotonic()
                args = optimize.parse_args([
                    "--predictions-csv", str(root / f"{protocol}-{side}-predictions.csv"),
                    "--start-gw", str(window["start_gw"]), "--max-gw", str(window["end_gw"]),
                    "--horizon", str(window["horizon"]), "--solver", "highs", "--gap-rel", "0",
                    "--scoring-mode", "corrected", "--time-limit", str(budget["solve_seconds"]),
                    "--output", str(root / f"{protocol}-{side}-squads.csv"),
                    "--decisions-json", str(root / f"{protocol}-{side}-decisions.json")])
                result = optimize.run(args)
                state["solver_seconds"] += time.monotonic() - solver_start
                if (len(result) != len(weeks) or result.gameweek.tolist() != weeks
                        or not result.native_optimality_assessment.eq("PROVEN_OPTIMAL").all()):
                    raise RuntimeError("optimizer did not return complete native-optimal gameweeks")
                decisions = _read_json(root / f"{protocol}-{side}-decisions.json")
                if (len(decisions["decisions"]) != len(weeks)
                        or decisions.get("native_optimality_assessment") != "PROVEN_OPTIMAL"
                        or any(row.get("native_optimality_assessment") != "PROVEN_OPTIMAL"
                               for row in decisions["decisions"])):
                    raise RuntimeError("decision receipt lacks complete native-optimal evidence")
            def pin(path):
                return {"path": str(path), "sha256": sha256_file(path)}
            spec = {"schema_version": 1, "id": f"training-{protocol}", "purpose": "historical_reanalysis",
                    "hypothesis": "Diagnostic historical model/feature contrast; no promotion or fresh-holdout claim.",
                    "metric": "actual_net_points", "bootstrap": {"n_boot": 10000, "block_len": 3, "seed": plan["seed"]},
                    **{side: {"name": plan["arms"][side]["model"], "input": pin(root / f"{protocol}-{side}-squads.csv")} for side in _ARMS},
                    "forecasts": {"information_protocol": protocol, "grain": "fixture", "population": population,
                                  **{side: pin(root / f"{protocol}-{side}-predictions.csv") for side in _ARMS}}}
            spec_path = root / f"{protocol}-comparison-spec.json"
            write_json_new(spec_path, spec)
            report = run_experiment(spec_path, runs_dir=root / "reports")
            card = read_run(report)
            comparisons[protocol] = card["decision_outcomes"]
            experiment.log_result(f"training-ground-{protocol}",
                {"scope": window["phase"], "protocol": protocol, "plan_sha256": sha256_file(root / "plan.json"),
                 "source_sha256": plan["source_sha256"], "fresh_holdout": False, "promotion_eligible": False},
                {"status": "COMPLETED", "net_points_candidate": comparisons[protocol]["total_a"],
                 "net_points_control": comparisons[protocol]["total_b"], "net_delta": comparisons[protocol]["total_diff"],
                 "ci_low": comparisons[protocol]["ci_low"], "ci_high": comparisons[protocol]["ci_high"],
                 "fit_seconds": state["fit_seconds"], "solver_seconds": state["solver_seconds"]},
                results_path=root / "registry.csv")
        if plan["source_sha256"] != _source_hashes() or plan["runtime"] != _runtime_versions():
            raise RuntimeError("source or runtime changed during execution")
        sys.stdout.flush()
        sys.stderr.flush()
        files = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*"))
                 if path.is_file() and path != root / "worker.log"}
        write_json_new(root / "worker-complete.json", {"schema_version": 1, "status": "COMPLETED",
            "plan_sha256": sha256_file(root / "plan.json"), "promotion_eligible": False, "fresh_holdout": False,
            "timings": {"feature_seconds": feature_seconds, "fit_seconds": state["fit_seconds"],
                        "solver_seconds": state["solver_seconds"], "total_seconds": time.monotonic() - started},
            "fit_calls": state["fit_calls"], "solver_calls": state["solver_calls"],
            "comparisons": comparisons, "outputs": files})
    except Exception as exc:
        _failure(root, str(exc), type(exc).__name__)
        raise
    finally:
        models.fit_model = original_fit


def _verify_outputs(root: Path, outputs: dict, excluded: set[str]) -> None:
    """Check exact file coverage and bytes, excluding only named lifecycle receipts."""
    if not isinstance(outputs, dict):
        raise ValueError("invalid research output manifest")
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and str(path.relative_to(root)) not in excluded}
    if actual != set(outputs):
        raise ValueError("research output file coverage changed")
    for name, digest in outputs.items():
        relative = Path(name)
        target = root / relative
        if relative.is_absolute() or ".." in relative.parts or not target.resolve().is_relative_to(root):
            raise ValueError("unsafe output path in completion receipt")
        if sha256_file(target) != digest:
            raise ValueError(f"research output changed: {name}")


def read_result(output_dir: str | Path) -> dict:
    """Verify exact captured file coverage and bytes before exposing a completion."""
    root = Path(output_dir).resolve()
    if (root / "failed.json").exists():
        raise ValueError("failed research runs cannot be reported as completed")
    receipt = _read_json(root / "complete.json")
    if (receipt.get("schema_version") != 1 or receipt.get("status") != "COMPLETED"
            or receipt.get("promotion_eligible") is not False or receipt.get("fresh_holdout") is not False):
        raise ValueError("invalid research completion receipt")
    _verify_outputs(root, receipt.get("outputs"), {"complete.json"})
    if receipt.get("plan_sha256") != sha256_file(root / "plan.json"):
        raise ValueError("completed plan hash differs")
    return receipt


def main(argv: list[str] | None = None) -> None:
    """Prepare registered plans, execute a bounded worker, or verify captured results."""
    parser = argparse.ArgumentParser(description="The Training Ground — bounded forecast experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for option in ("dataset", "output"):
        prepare.add_argument("--" + option, type=Path, required=True)
    for option in ("start-gw", "end-gw"):
        prepare.add_argument("--" + option, type=int, required=True)
    prepare.add_argument("--parameters", type=Path)
    prepare.add_argument("--control-model", default="catboost_mae")
    prepare.add_argument("--candidate-model", default="catboost_rmse")
    prepare.add_argument("--control-drop", nargs="*", default=[])
    prepare.add_argument("--candidate-drop", nargs="*", default=[])
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--retrain-every", type=int, default=4)
    prepare.add_argument("--horizon", type=int, default=3)
    prepare.add_argument("--max-seconds", type=float)
    prepare.add_argument("--solve-seconds", type=float)
    run = commands.add_parser("run")
    run.add_argument("plan", type=Path)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            values = vars(args).copy()
            for name in ("command", "output", "dataset"):
                values.pop(name)
            values["parameters"] = _read_json(args.parameters) if args.parameters else None
            write_json_new(args.output, create_plan(args.dataset, **values))
            print(args.output)
        elif args.command == "run":
            print(run_plan(args.plan, args.dataset, args.output))
        elif args.command == "report":
            print(json.dumps(read_result(args.directory), indent=2, allow_nan=False))
        else:
            _worker(args.directory)
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Training Ground stopped: {exc}\n")


if __name__ == "__main__":
    main()
