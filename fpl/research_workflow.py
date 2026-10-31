"""Hypothesis registration and conservative iteration budgets over Training Ground."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from fpl import config, research_runner as runner
from fpl.artifacts import sha256_file, write_json_new

_TEXT = ("question", "hypothesis", "success_criteria", "stopping_rule", "limitations")
_KEYS = {*_TEXT, "max_worker_seconds", "plans"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def _positive(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("worker budgets must be finite positive numbers")
    return float(value)


def _validate(spec: dict, plans: list[dict]) -> None:
    if set(spec) != _KEYS or any(not isinstance(spec[key], str) or not spec[key].strip() for key in _TEXT):
        raise ValueError("registration requires exact fields and nonempty research statements")
    maximum = _positive(spec["max_worker_seconds"])
    if not isinstance(spec["plans"], list) or not plans:
        raise ValueError("registration requires at least one plan")
    identities = []
    for plan in plans:
        if set(plan) != runner._PLAN_KEYS or plan["promotion_eligible"] is not False:
            raise ValueError("invalid runner plan")
        window = plan["window"]
        canonical = runner._window(window["start_gw"], window["end_gw"], window["retrain_every"], window["horizon"])
        if window != canonical or canonical["phase"] != "selection_diagnostic":
            raise ValueError("iterative research admits selection diagnostics only")
        identities.append([window, plan["dataset_sha256"], plan["seed"], plan["arms"]["control"],
                           plan["source_sha256"], plan["runtime"]])
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("iterations require identical control, window, seed and provenance")
    if sum(_positive(plan["budget"]["max_seconds"]) for plan in plans) > maximum:
        raise ValueError("allocated worker limits exceed aggregate budget")


def register(spec_path: str | Path, output_dir: str | Path) -> Path:
    """Capture predeclared plans and semantic criteria without fitting or solving.

    Aggregate compute is conservatively reserved as the sum of worker time caps;
    registration/validation/reporting overhead is outside that worker budget.
    """
    source, root = Path(spec_path).resolve(), Path(output_dir).resolve()
    spec = _json(source)
    if set(spec) != _KEYS or not isinstance(spec["plans"], list) or any(not isinstance(p, str) for p in spec["plans"]):
        raise ValueError("invalid registration fields or plan paths")
    paths = [(source.parent / name).resolve() for name in spec["plans"]]
    contents = [path.read_bytes() for path in paths]
    plans = [json.loads(content) for content in contents]
    _validate(spec, plans)
    if any(root.is_relative_to(folder.resolve()) for folder in (config.DATASETT_DIR, config.ROOT / "fpl")):
        raise ValueError("research output cannot target production code or datasets")
    root.mkdir(parents=True, exist_ok=False)
    pins = []
    for index, content in enumerate(contents, 1):
        path = root / f"plan-{index}.json"
        with path.open("xb") as stream:
            stream.write(content)
        pins.append({"path": path.name, "sha256": sha256_file(path)})
    write_json_new(root / "registration.json", {**spec, "plans": pins})
    write_json_new(root / "registered.json", {"registration_sha256": sha256_file(root / "registration.json")})
    return root


def _load(root: Path) -> dict:
    if _json(root / "registered.json")["registration_sha256"] != sha256_file(root / "registration.json"):
        raise ValueError("registration bytes changed")
    spec = _json(root / "registration.json")
    plans = []
    for index, pin in enumerate(spec["plans"], 1):
        path = root / f"plan-{index}.json"
        if pin["path"] != path.name or pin["sha256"] != sha256_file(path):
            raise ValueError("registered plan bytes changed")
        plans.append(_json(path))
    _validate(spec, plans)
    return spec


def _claims(root: Path) -> list[Path]:
    paths = list(root.glob("launch-*.json"))
    expected = [root / f"launch-{i}.json" for i in range(1, len(paths) + 1)]
    if set(paths) != set(expected):
        raise ValueError("launch chronology is not contiguous")
    return expected


def _completed(root: Path, index: int, pin: dict, previous: str | None) -> dict:
    claim = _json(root / f"launch-{index}.json")
    if claim["plan_sha256"] != pin["sha256"] or claim["previous_complete_sha256"] != previous:
        raise ValueError("launch chronology or registered plan binding changed")
    result = runner.read_result(root / f"trial-{index}")
    if result["plan_sha256"] != pin["sha256"]:
        raise ValueError("completed trial differs from registered plan")
    return result


def run_next(directory: str | Path, dataset_path: str | Path, reason: str) -> Path:
    """Launch the next predeclared plan after verified prior evidence and a reason.

    Failed or incomplete launches consume their reservation and block continuation.
    The exclusive lock remains after process termination; it is never auto-cleared.
    """
    root = Path(directory).resolve()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a scientific reason is required before each launch")
    spec = _load(root)
    lock = root / "launch.lock"
    write_json_new(lock, {"reason": reason})
    try:
        claims, previous = _claims(root), None
        for index, _ in enumerate(claims, 1):
            _completed(root, index, spec["plans"][index - 1], previous)
            previous = sha256_file(root / f"trial-{index}" / "complete.json")
        index = len(claims) + 1
        if index > len(spec["plans"]):
            raise ValueError("registered iteration budget exhausted")
        pin = spec["plans"][index - 1]
        write_json_new(root / f"launch-{index}.json", {"reason": reason, "plan_sha256": pin["sha256"],
                                                     "previous_complete_sha256": previous})
        return runner.run_plan(root / pin["path"], dataset_path, root / f"trial-{index}")
    finally:
        lock.unlink()


def report(directory: str | Path) -> str:
    """Render compact saved metrics only after existing runner hash verification."""
    root = Path(directory).resolve()
    spec = _load(root)
    claims = _claims(root)
    first = _json(root / "plan-1.json")
    lines = ["# FPL research report", *[f"{key.replace('_', ' ').title()}: {spec[key]}" for key in _TEXT],
             f"Configuration: global GW {first['window']['start_gw']}-{first['window']['end_gw']}; "
             f"horizon {first['window']['horizon']}; seed {first['seed']}; control {first['arms']['control']['model']}. "
             "Exact recipes, ordered features and provenance are pinned in plan-N.json.",
             f"Worker budget: {spec['max_worker_seconds']} seconds; {len(claims)}/{len(spec['plans'])} launches reserved.",
             "Selection diagnostics; not a fresh holdout. Recommendation: investigate further. No automatic promotion."]
    previous = None
    for index, _ in enumerate(claims, 1):
        trial = root / f"trial-{index}"
        if not (trial / "complete.json").exists() or (trial / "failed.json").exists():
            status = "FAILED" if (trial / "failed.json").exists() else "INCOMPLETE"
            lines.append(f"Trial {index}: {status}; no verified metrics; continuation blocked.")
            break
        result = _completed(root, index, spec["plans"][index - 1], previous)
        previous = sha256_file(trial / "complete.json")
        plan = _json(root / f"plan-{index}.json")
        lines.append(f"Trial {index}: COMPLETED (verified captured bytes); candidate {plan['arms']['candidate']['model']}.")
        control_features = plan['arms']['control']['features']
        candidate_features = plan['arms']['candidate']['features']
        dropped = [name for name in control_features if name not in candidate_features]
        added = [name for name in candidate_features if name not in control_features]
        lines.append(f"Feature intervention: dropped {dropped}; added {added} (relative to control).")
        for protocol in ("standard", "origin"):
            comparison = result["comparisons"][protocol]
            lines.append(f"{protocol}: candidate/control net points {comparison['total_a']}/{comparison['total_b']}; "
                         f"delta {comparison['total_diff']}, paired CI [{comparison['ci_low']}, {comparison['ci_high']}].")
            summary = _json(trial / f"{protocol}-forecast-summary.json")
            for item in summary["summaries"]:
                for side in ("control", "candidate"):
                    stats = item[side]["overall"]
                    values = ", ".join(f"{key}={stats.get(key)}" for key in
                                       ("mae", "rmse", "bias", "total_calibration", "spearman", "top1_capture"))
                    lines.append(f"{protocol} lead={item['lead']} {side}: {values}.")
        lines.append("Evidence remains UNVERIFIED/unranked; short-window intervals do not establish generalization.")
    return "\n\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Register, advance one bounded iteration, or print a verified compact report."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("spec", type=Path)
    registration.add_argument("--output", type=Path, required=True)
    advance = commands.add_parser("next")
    advance.add_argument("directory", type=Path)
    advance.add_argument("--dataset", type=Path, required=True)
    advance.add_argument("--reason", required=True)
    reporting = commands.add_parser("report")
    reporting.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            print(register(args.spec, args.output))
        elif args.command == "next":
            print(run_next(args.directory, args.dataset, args.reason))
        else:
            print(report(args.directory), end="")
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, IndexError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Research workflow stopped: {exc}\n")


if __name__ == "__main__":
    main()
