"""The Training Ground: offline saved-run comparisons and an evidence league.

No live fetch, model fitting, arbitrary executable adapters or production writes
occur here. Fitting/ablation orchestration will use the existing research APIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from fpl import config
from fpl.artifacts import sha256_file, write_json_new
from fpl.backtest_contract import validate_pair
from fpl.data.provenance import load_frozen_dataset_manifest
from fpl.milp.compare_backtests import compare, paired_gw_differences
from fpl.scorecard import build_scorecard, render_scorecard

_OUTPUTS = {"spec.json", "candidate.csv", "control.csv", "comparison.json", "scorecard.json", "REPORT.md"}
_OPTIONAL_OUTPUTS = {"candidate-contract.json", "control-contract.json"}


def _read_json(path: Path) -> tuple[dict, bytes]:
    content = path.read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("JSON must be an object")
    json.dumps(value, allow_nan=False)
    return value, content


def _slug(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is not None


def _pinned(record: dict, base: Path) -> tuple[Path, bytes]:
    if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
            or not isinstance(record["path"], str) or not record["path"]):
        raise ValueError("input requires path and sha256")
    path = (base / record["path"]).resolve()
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError(f"input bytes changed: {path}")
    return path, content


def _load_spec(path: str | Path) -> tuple[dict, bytes, dict]:
    location = Path(path).resolve()
    spec, content = _read_json(location)
    required = {"schema_version", "id", "purpose", "hypothesis", "metric", "candidate", "control", "bootstrap"}
    if (set(spec) != required or type(spec["schema_version"]) is not int or spec["schema_version"] != 1
            or not _slug(spec["id"])):
        raise ValueError("invalid experiment fields, schema or id")
    if spec["purpose"] not in ("framework_validation", "historical_reanalysis"):
        raise ValueError("initial playground supports framework_validation or historical_reanalysis only")
    if not isinstance(spec["hypothesis"], str) or not spec["hypothesis"].strip():
        raise ValueError("hypothesis is required")
    if spec["metric"] not in ("actual_total_points", "actual_net_points"):
        raise ValueError("explicit gross/net metric is required")
    bootstrap = spec["bootstrap"]
    if not isinstance(bootstrap, dict) or set(bootstrap) != {"n_boot", "block_len", "seed"}:
        raise ValueError("bootstrap requires n_boot, block_len and seed")
    for name, value in bootstrap.items():
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"invalid bootstrap {name}")
    if bootstrap["n_boot"] > config.PLAYGROUND_MAX_BOOTSTRAPS:
        raise ValueError("bootstrap draws exceed playground budget")
    inputs = {}
    frames = []
    for name in ("candidate", "control"):
        entry = spec[name]
        if (not isinstance(entry, dict) or not {"name", "input"} <= set(entry)
                or set(entry) - {"name", "input", "contract"}
                or not isinstance(entry["name"], str) or not entry["name"].strip()):
            raise ValueError(f"invalid {name} definition")
        inputs[name] = _pinned(entry["input"], location.parent)
        frames.append(pd.read_csv(BytesIO(inputs[name][1])))
        if "contract" in entry:
            inputs[f"{name}_contract"] = _pinned(entry["contract"], location.parent)
    paired_gw_differences(*frames, points_col=spec["metric"])
    cutoff = load_frozen_dataset_manifest()["max_gw_global"]
    if any((frame.gameweek > cutoff).any() for frame in frames):
        raise ValueError("prospective rounds cannot enter playground research/reanalysis")
    if ("candidate_contract" in inputs) != ("control_contract" in inputs):
        raise ValueError("both contracts are required")
    if "candidate_contract" in inputs:
        admitted = validate_pair(str(inputs["candidate_contract"][0]), str(inputs["control_contract"][0]),
                                [spec[name]["input"]["sha256"] for name in ("candidate", "control")],
                                frames, spec["metric"])
        expected = "framework_validation" if spec["purpose"] == "framework_validation" else "registered_research"
        if admitted["a"]["purpose"] != expected:
            raise ValueError("comparison purpose cannot be relabelled as historical research")
    return spec, content, inputs


def validate_spec(path: str | Path) -> dict:
    """Preflight exact input bytes, metric compatibility and declared compute limits."""
    return _load_spec(path)[0]


def _identity(content: bytes) -> dict:
    modules = ("playground.py", "scorecard.py", "artifacts.py", "backtest_contract.py",
               "milp/compare_backtests.py", "config.py", "data/provenance.py",
               "data/frozen_research_dataset_gw231.json")
    root = Path(__file__).parent
    return {"spec_sha256": hashlib.sha256(content).hexdigest(),
            "implementation": {name: sha256_file(root / name) for name in modules},
            "reporting_runtime": {"python": platform.python_version(),
                                  **{name: version(name) for name in ("numpy", "pandas", "scipy")}}}


def read_run(directory: str | Path) -> dict:
    """Verify every captured output before reporting a completed historical run."""
    root = Path(directory)
    receipt, _ = _read_json(root / "complete.json")
    outputs = receipt.get("outputs")
    if (receipt.get("schema_version") != 1 or not isinstance(outputs, dict)
            or not _OUTPUTS <= set(outputs) or set(outputs) - _OUTPUTS - _OPTIONAL_OUTPUTS):
        raise ValueError("invalid completion receipt")
    for name, digest in outputs.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"completed output changed: {name}")
    spec, content = _read_json(root / "spec.json")
    card, _ = _read_json(root / "scorecard.json")
    if receipt.get("identity", {}).get("spec_sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError("completed spec differs from run identity")
    if card.get("schema_version") != 1 or card["experiment"]["id"] != spec["id"]:
        raise ValueError("scorecard schema/identity mismatch")
    return card


def run_experiment(path: str | Path, *, attempt: str = "first",
                   runs_dir: str | Path | None = None) -> Path:
    """Capture one immutable reporting attempt, or verify and reuse its completion.

    Inputs are saved before comparison. Failed/incomplete attempts are preserved;
    a different explicit attempt name is required to retry. No research fits occur.
    """
    spec, content, inputs = _load_spec(path)
    if not _slug(attempt):
        raise ValueError("invalid attempt name")
    root = Path(config.PLAYGROUND_RUNS_DIR if runs_dir is None else runs_dir)
    resolved_root = root.resolve()
    protected = (config.DATASETT_DIR.resolve(), (config.ROOT / "fpl").resolve())
    if any(resolved_root.is_relative_to(folder) for folder in protected):
        raise ValueError("playground outputs cannot target production code or dataset directories")
    directory = root / spec["id"] / attempt
    if not directory.resolve().is_relative_to(resolved_root):
        raise ValueError("attempt directory escapes runs root")
    identity = _identity(content)
    if directory.exists():
        if not (directory / "complete.json").exists():
            raise FileExistsError("started/failed attempt preserved; choose a new --attempt")
        receipt, _ = _read_json(directory / "complete.json")
        if receipt.get("identity") != identity:
            raise ValueError("completed attempt has different inputs, implementation or runtime")
        read_run(directory)
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    write_json_new(directory / "started.json", {"identity": identity, "id": spec["id"],
                                               "attempt": attempt, "purpose": spec["purpose"]})
    started = time.monotonic()
    try:
        with (directory / "spec.json").open("xb") as stream:
            stream.write(content)
        for name in ("candidate", "control"):
            with (directory / f"{name}.csv").open("xb") as stream:
                stream.write(inputs[name][1])
        contracts = {}
        if "candidate_contract" in inputs:
            for name, key in (("candidate", "contract_a"), ("control", "contract_b")):
                source, raw = inputs[f"{name}_contract"]
                with (directory / f"{name}-contract.json").open("xb") as stream:
                    stream.write(raw)
                # Original location preserves relative artifact resolution.
                contracts[key] = str(source)
        comparison = compare(str(directory / "candidate.csv"), str(directory / "control.csv"),
                             points_col=spec["metric"], **spec["bootstrap"], **contracts)
        if contracts:
            for name, key in (("candidate", "a"), ("control", "b")):
                if comparison["contract"][key]["contract_sha256"] != spec[name]["contract"]["sha256"]:
                    raise ValueError("contract changed during comparison")
        write_json_new(directory / "comparison.json", comparison)
        card = build_scorecard(spec, comparison, reporting_seconds=time.monotonic() - started)
        write_json_new(directory / "scorecard.json", card)
        with (directory / "REPORT.md").open("x", encoding="utf-8") as stream:
            stream.write(render_scorecard(card))
        names = _OUTPUTS | (_OPTIONAL_OUTPUTS if contracts else set())
        write_json_new(directory / "complete.json", {"schema_version": 1, "identity": identity,
                       "outputs": {name: sha256_file(directory / name) for name in sorted(names)}})
    except Exception as exc:
        write_json_new(directory / "failed.json", {"type": type(exc).__name__, "reason": str(exc)})
        raise
    return directory


def league(runs_dir: str | Path | None = None) -> list[dict]:
    """Group completed results by matching control and comparison assumptions.

    Ranks are descriptive within a cohort, never promotion decisions. Exhibitions,
    unverified results and incomplete attempts remain visible without ranks.
    """
    root = Path(config.PLAYGROUND_RUNS_DIR if runs_dir is None else runs_dir)
    groups: dict[str, dict] = {}
    seen = set()
    for directory in sorted(root.glob("*/*")):
        if not directory.is_dir():
            continue
        if not (directory / "complete.json").exists():
            status = "FAILED" if (directory / "failed.json").exists() else "INCOMPLETE"
            key = f"unfinished:{directory.parent.name}/{directory.name}"
            groups[key] = {"competition": key, "control": None, "entries": [
                {"candidate": directory.parent.name, "status": status, "rank": None,
                 "run": str(directory), "net_or_gross_points": None, "delta": None, "ci": None}]}
            continue
        card = read_run(directory)
        result, experiment = card["decision_outcomes"], card["experiment"]
        # Identical inputs are one observation, not extra evidence from retries.
        contract_identity = None
        if result["contract"]:
            contract_identity = {side: {key: result["contract"][side][key]
                                        for key in ("context", "artifacts", "purpose", "intervention")}
                                 for side in ("a", "b")}
        observation = json.dumps({"inputs": result["input_sha256"], "metric": result["metric"],
                                  "contract": contract_identity, "purpose": experiment["purpose"]}, sort_keys=True)
        if observation in seen:
            continue
        seen.add(observation)
        rankable = card["status"] == "CONTRACT_VALIDATED"
        if rankable:
            control = result["contract"]["b"]
            cohort = {"metric": result["metric"], "control_csv": result["input_sha256"]["b"],
                      "context": control["context"], "artifacts": control["artifacts"],
                      "intervention": control["intervention"], "purpose": experiment["purpose"]}
            key = hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()
        else:
            key = f"unranked:{directory.parent.name}/{directory.name}"
        group = groups.setdefault(key, {"competition": key, "control": experiment["control"],
                                        "metric": result["metric"], "entries": []})
        group["entries"].append({"candidate": experiment["candidate"], "status": card["status"],
                                 "rank": None, "run": str(directory),
                                 "net_or_gross_points": result["total_a"], "delta": result["total_diff"],
                                 "ci": [result["ci_low"], result["ci_high"]]})
    for group in groups.values():
        entries = group["entries"]
        if all(row["status"] == "CONTRACT_VALIDATED" for row in entries):
            entries.sort(key=lambda row: (-row["net_or_gross_points"], row["candidate"]))
            previous, rank = None, 0
            for index, row in enumerate(entries, 1):
                if row["net_or_gross_points"] != previous:
                    rank = index
                row["rank"], previous = rank, row["net_or_gross_points"]
    return list(groups.values())


def init_demo(destination: str | Path) -> Path:
    """Create labelled synthetic backtests and their pinned exhibition specification."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    start, _ = config.season_window("2024-25")
    entries = {}
    for name, points, hits in (("candidate", [52, 62, 65], [1, 1, 0]),
                               ("control", [50, 60, 60], [0, 0, 0])):
        path = root / f"{name}.csv"
        pd.DataFrame({"gameweek": [start, start + 1, start + 2], "actual_total_points": points,
                      "alpha": hits, "scoring_protocol": "autosubs_v2"}).to_csv(path, index=False)
        entries[name] = {"name": f"synthetic-{name}", "input": {"path": path.name, "sha256": sha256_file(path)}}
    spec = {"schema_version": 1, "id": "synthetic-exhibition", "purpose": "framework_validation",
            "hypothesis": "Check reporting and hit accounting; synthetic scores are not FPL evidence.",
            "metric": "actual_net_points", **entries,
            "bootstrap": {"n_boot": 1000, "block_len": 2, "seed": 0}}
    write_json_new(root / "spec.json", spec)
    return root / "spec.json"


def render_league(competitions: list[dict]) -> str:
    """Show descriptive cohort standings alongside unranked and unfinished entries."""
    def label(value: Any) -> str:
        return str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")

    lines = ["# The Training Ground — league table", "",
             f"Configured Matchday strategy: `{config.PRODUCTION_WEIGHT_STRATEGY}`. "
             "Ranks describe saved results only; they do not approve promotion.", ""]
    if not competitions:
        lines += ["No captured experiments yet.", ""]
    for group in competitions:
        lines += [f"## Competition: {label(group['competition'][:20])}", "",
                  f"Control: {label(group.get('control') or 'unavailable')}; "
                  f"metric: `{group.get('metric', 'unavailable')}`.", "",
                  "| Rank | Candidate | Status | Points | Difference | Paired 95% interval |",
                  "| ---: | --- | --- | ---: | ---: | --- |"]
        for row in group["entries"]:
            points = "—" if row["net_or_gross_points"] is None else f"{row['net_or_gross_points']:.3f}"
            delta = "—" if row["delta"] is None else f"{row['delta']:+.3f}"
            interval = "—" if row["ci"] is None else f"[{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]"
            rank = "—" if row["rank"] is None else str(row["rank"])
            lines.append(f"| {rank} | {label(row['candidate'])} | {row['status']} | {points} | {delta} | {interval} |")
        lines.append("")
    lines += ["Exhibitions and unverified/incomplete runs are unranked. "
              "Contract-validated ranks still do not establish causal validity, legality, native optimality "
              "or a fresh holdout. Repeated attempts are not independent evidence.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Expose validate/run/report/league and an explicitly synthetic exhibition."""
    parser = argparse.ArgumentParser(description="The Training Ground — offline evidence playground")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("init-demo", help="Create synthetic exhibition inputs in a new directory")
    demo.add_argument("destination", type=Path)
    validate = commands.add_parser("validate", help="Check a saved-run comparison spec without writing")
    validate.add_argument("spec", type=Path)
    run = commands.add_parser("run", help="Run or verify one immutable reporting attempt")
    run.add_argument("spec", type=Path)
    run.add_argument("--attempt", default="first")
    run.add_argument("--runs-dir", type=Path, default=config.PLAYGROUND_RUNS_DIR)
    report = commands.add_parser("report", help="Display a verified completed scorecard")
    report.add_argument("directory", type=Path)
    table = commands.add_parser("league", help="Show separate evidence competitions; no automatic promotion")
    table.add_argument("--runs-dir", type=Path, default=config.PLAYGROUND_RUNS_DIR)
    table.add_argument("--json", action="store_true", help="Emit structured competitions for a future dashboard")
    args = parser.parse_args(argv)
    try:
        if args.command == "init-demo":
            print(init_demo(args.destination))
        elif args.command == "validate":
            spec = validate_spec(args.spec)
            print(f"PASS: {spec['id']} ({spec['purpose']}); no model promotion assessed")
        elif args.command == "run":
            print(run_experiment(args.spec, attempt=args.attempt, runs_dir=args.runs_dir))
        elif args.command == "report":
            print(render_scorecard(read_run(args.directory)))
        else:
            competitions = league(args.runs_dir)
            print(json.dumps(competitions, indent=2, allow_nan=False) if args.json else render_league(competitions))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Training Ground stopped: {exc}\n")


if __name__ == "__main__":
    main()
