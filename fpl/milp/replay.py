"""Strict saved-decision settlement, without fitting, solving or inferring blanks.

Static roster checks and shared corrected settlement are separate from financial
state/rule legality, native solver verification and outcome-blind execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np
import pandas as pd

from fpl import config, experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.compare_backtests import scored_points
from fpl.milp.solio import score_decisions
from fpl.research_runner import _source_hashes, _runtime_versions, _verify_outputs


def _integer(value, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not 0 < value < 2**63):
        raise ValueError(f"{name} requires positive integer stable identities")
    return int(value)


def _ids(value, count: int, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} requires an ordered list of {count} stable IDs")
    result = [_integer(item, name) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate stable IDs")
    return result


def replay_snapshot(snapshot: dict, predictions: pd.DataFrame) -> pd.DataFrame:
    """Check static rosters and resettle against complete selected-squad outcomes.

    Team/position identity must be explicit and consistent in the current round.
    Missing player rows cannot establish blanks. No bank/transfer/chip-inventory
    legality or original native optimality is certified by this function.
    """
    required = {"schema_version", "settlement_completed", "decisions"}
    if (not isinstance(snapshot, dict) or not required <= set(snapshot)
            or type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1
            or snapshot["settlement_completed"] is not False
            or not isinstance(snapshot["decisions"], list) or not snapshot["decisions"]):
        raise ValueError("replay requires a nonempty schema-1 pre-settlement snapshot")
    json.dumps(snapshot, allow_nan=False)
    if (predictions.empty or not predictions.columns.is_unique
            or not {"player_id", "GW", "position", "team"} <= set(predictions)):
        raise ValueError("replay requires explicit current-round player/team/position outcomes")
    for column in ("player_id", "GW"):
        values = predictions[column]
        if (not values.map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all()
                or not np.isfinite(values.to_numpy(dtype=float)).all()
                or not ((values > 0) & (values < 2**63) & (values % 1 == 0)).all()):
            raise ValueError(f"invalid stable forecast identity: {column}")
    if "GW_global" in predictions and not predictions.GW_global.eq(predictions.GW).all():
        raise ValueError("replay GW and GW_global must use the same global mapping")
    records, weeks = [], []
    for source in snapshot["decisions"]:
        if not isinstance(source, dict) or any(key.startswith("actual_") or key == "gross_actual_total_points" for key in source):
            raise ValueError("snapshot rows must precede realized settlement outputs")
        row = source.copy()
        gw = _integer(row.get("gameweek"), "gameweek")
        if gw in weeks:
            raise ValueError("snapshot has duplicate gameweeks")
        weeks.append(gw)
        squad = _ids(row.get("squad"), 15, "squad")
        lineup = _ids(row.get("lineup"), 11, "lineup")
        bench = _ids(row.get("bench_order"), 4, "bench_order")
        captain = _ids(row.get("captain"), 1, "captain")
        vice = _ids(row.get("vice_captain"), 1, "vice_captain")
        if (set(lineup) & set(bench) or set(lineup + bench) != set(squad)
                or captain == vice or not set(captain + vice) <= set(lineup)):
            raise ValueError(f"invalid XI/bench/captain partition in GW{gw}")
        chip = row.get("chip_played")
        if chip not in (None, "WC", "FH", "BB", "TC") and not (isinstance(chip, str) and chip.startswith("TC_") and len(chip) > 3):
            raise ValueError("unknown chip label in replay snapshot")
        current = predictions.loc[predictions.GW == gw]
        missing = set(squad) - set(current.player_id)
        if missing:
            raise ValueError(f"missing selected-player outcomes in GW{gw}: {sorted(missing)}; blanks require explicit evidence")
        selected = current.loc[current.player_id.isin(squad)].copy()
        selected["team"] = selected.team.replace(config.TEAM_NAME_CORRECTIONS)
        if (not selected.position.isin(config.ONFIELD_POSITIONS).all()
                or not selected.team.map(lambda name: isinstance(name, str) and bool(name.strip())).all()
                or selected.groupby("player_id")[["position", "team"]].nunique().gt(1).any().any()):
            raise ValueError(f"ambiguous current-round position/team identity in GW{gw}")
        metadata = selected.drop_duplicates("player_id").set_index("player_id")
        positions = metadata.position.to_dict()
        if Counter(positions[player] for player in squad) != {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}:
            raise ValueError(f"illegal squad position counts in GW{gw}")
        if metadata.team.value_counts().gt(3).any():
            raise ValueError(f"illegal club limit in GW{gw}")
        counts = Counter(positions[player] for player in lineup)
        if any(not config.LINEUP_MIN_COUNTS[pos] <= counts[pos] <= config.LINEUP_MAX_COUNTS[pos]
               for pos in config.ONFIELD_POSITIONS):
            raise ValueError(f"illegal XI formation in GW{gw}")
        if positions[bench[0]] != "GK" or any(positions[player] == "GK" for player in bench[1:]):
            raise ValueError(f"reserve GK must retain its separate bench slot in GW{gw}")
        records.append(row)
    if weeks != sorted(weeks):
        raise ValueError("snapshot gameweeks must retain chronological order")
    # Shared scorer admits fixture identity, outcomes, hits and repeated origins.
    return score_decisions(pd.DataFrame(records), predictions, require_complete_outcomes=True)


def run_replay(snapshot_path: str | Path, predictions_path: str | Path,
               output_dir: str | Path, *, reference_path: str | Path | None = None) -> Path:
    """Capture immutable inputs and report settlement evidence, including failures."""
    output = Path(output_dir).resolve()
    if any(output.is_relative_to(folder.resolve()) for folder in (config.DATASETT_DIR, config.ROOT / "fpl")):
        raise ValueError("replay output cannot target production code or datasets")
    inputs = {"decisions.json": Path(snapshot_path).read_bytes(), "predictions.csv": Path(predictions_path).read_bytes()}
    if reference_path is not None:
        inputs["reference.csv"] = Path(reference_path).read_bytes()
    sources, runtime = _source_hashes(), _runtime_versions()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        for name, content in inputs.items():
            with (output / name).open("xb") as stream:
                stream.write(content)
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in inputs.items()}
        write_json_new(output / "started.json", {"status": "STARTED", "input_sha256": hashes,
                       "source_sha256": sources, "runtime": runtime, "promotion_eligible": False})
        snapshot = json.loads(inputs["decisions.json"])
        replayed = replay_snapshot(snapshot, pd.read_csv(BytesIO(inputs["predictions.csv"])))
        with (output / "replayed.csv").open("x", encoding="utf-8", newline="") as stream:
            replayed.to_csv(stream, index=False)
        reconciliation = "NOT ASSESSED"
        if "reference.csv" in inputs:
            reference = pd.read_csv(BytesIO(inputs["reference.csv"]))
            for metric in ("actual_total_points", "actual_net_points"):
                actual = scored_points(replayed, points_col=metric)
                expected = scored_points(reference, points_col=metric)
                if not actual.index.equals(expected.index) or not np.allclose(actual, expected, rtol=0, atol=1e-9):
                    raise ValueError(f"reference reconciliation failed for {metric}")
            if "scoring_protocol" not in reference or not reference.scoring_protocol.eq("autosubs_v2").all():
                raise ValueError("reference requires corrected scoring protocol")
            reconciliation = "PASS"
        if sources != _source_hashes() or runtime != _runtime_versions():
            raise ValueError("source or runtime changed during replay")
        result = {"schema_version": 1, "status": "SETTLEMENT_REPLAY_COMPLETED", "promotion_eligible": False,
                  "static_roster_structure": "PASS", "selected_squad_outcomes": "COMPLETE",
                  "reference_reconciliation": reconciliation, "independent_financial_rule_legality": "NOT ASSESSED",
                  "native_optimality": "NOT ASSESSED", "outcome_blind_execution": "NOT ASSESSED",
                  "gameweeks": replayed.gameweek.tolist(),
                  "gross_points": float(replayed.actual_total_points.sum()),
                  "net_points": float(replayed.actual_net_points.sum()),
                  "replay_seconds": time.monotonic() - started, "input_sha256": hashes,
                  "source_sha256": sources, "runtime": runtime}
        experiment.log_result("saved-decision-settlement-replay", {"scope": "settlement_replay", "input_sha256": hashes},
                              {key: result[key] for key in ("status", "gross_points", "net_points", "replay_seconds")},
                              results_path=output / "registry.csv")
        result["output_sha256"] = {str(path.relative_to(output)): sha256_file(path)
                                    for path in sorted(output.rglob("*")) if path.is_file()}
        write_json_new(output / "complete.json", result)
        read_replay(output)
    except Exception as exc:
        write_json_new(output / "failed.json", {"status": "FAILED", "reason": str(exc),
                       "type": type(exc).__name__, "promotion_eligible": False})
        raise
    return output


def read_replay(output_dir: str | Path) -> dict:
    """Verify exact saved input/output coverage before reporting a replay receipt."""
    root = Path(output_dir).resolve()
    if (root / "failed.json").exists():
        raise ValueError("failed replay cannot be reported as completed")
    result = json.loads((root / "complete.json").read_bytes())
    json.dumps(result, allow_nan=False)
    if (not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("status") != "SETTLEMENT_REPLAY_COMPLETED"
            or result.get("promotion_eligible") is not False):
        raise ValueError("invalid saved replay receipt")
    _verify_outputs(root, result.get("output_sha256"), {"complete.json"})
    inputs = result.get("input_sha256")
    if not isinstance(inputs, dict) or set(inputs) not in (
            {"decisions.json", "predictions.csv"}, {"decisions.json", "predictions.csv", "reference.csv"}):
        raise ValueError("invalid captured replay input manifest")
    for name, digest in result["input_sha256"].items():
        if not isinstance(digest, str) or len(digest) != 64 or result["output_sha256"].get(name) != digest:
            raise ValueError("captured replay input identity changed")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Strict saved-decision settlement replay")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("decisions", type=Path)
    run.add_argument("predictions", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reference", type=Path)
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            print(json.dumps(read_replay(args.directory), indent=2))
        else:
            print(run_replay(args.decisions, args.predictions, args.output, reference_path=args.reference))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Settlement replay stopped: {exc}\n")


if __name__ == "__main__":
    main()
