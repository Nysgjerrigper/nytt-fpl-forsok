"""Strict saved-decision settlement, without fitting, solving or inferring blanks.

Static roster checks and shared corrected settlement are separate from financial
state/rule legality, native solver verification and outcome-blind execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import gzip
import re
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


def replay_snapshot(snapshot: dict, predictions: pd.DataFrame, *,
                    blank_outcomes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Check static rosters and resettle against complete selected-squad outcomes.

    Team/position identity must be explicit and consistent in the current round.
    Missing player rows cannot establish blanks. Separately admitted blank rows
    require caller-verified evidence; run_replay captures and checks that evidence.
    No bank/transfer/chip-inventory
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
        if blank_outcomes is not None:
            current = pd.concat([current, blank_outcomes.loc[blank_outcomes.GW == gw]], ignore_index=True)
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
    return score_decisions(pd.DataFrame(records), predictions, require_complete_outcomes=True,
                           explicit_blank_outcomes=blank_outcomes)


def _admit_blank_evidence(snapshot: dict, predictions: pd.DataFrame, inputs: dict[str, bytes]) -> pd.DataFrame:
    """Validate captured hashes and predeadline register timing before blank admission.

    Hashes bind declared sources; they do not authenticate an archive publisher.
    Registration is capture-time identity, not exact-deadline identity proof.
    """
    from fpl.milp.blank_outcomes import explicit_blank_outcomes

    evidence = json.loads(inputs["blank-evidence.json"])
    names = {"fixtures.csv", "teams.csv", "bootstrap-static.json.gz", "manifest.json"}
    if (not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int
            or evidence["schema_version"] != 1 or set(evidence.get("input_sha256", {})) != names
            or any(evidence["input_sha256"][name] != hashlib.sha256(inputs[name]).hexdigest() for name in names)
            or not isinstance(evidence.get("fixture_source_commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", evidence["fixture_source_commit"]) is None):
        raise ValueError("invalid blank evidence hashes or pinned fixture source")
    manifest = json.loads(inputs["manifest.json"])
    payload = gzip.decompress(inputs["bootstrap-static.json.gz"])
    entries = [entry for entry in manifest["files"] if entry["name"] == "bootstrap-static.json.gz"]
    if (len(entries) != 1 or entries[0]["endpoint"] != "/bootstrap-static/"
            or entries[0]["sha256"] != hashlib.sha256(payload).hexdigest()):
        raise ValueError("historical register payload differs from capture manifest")
    register = json.loads(payload)
    event = evidence["event"]
    deadlines = [row["deadline_time"] for row in register["events"] if row["id"] == event]
    if len(deadlines) != 1 or manifest["event"] != event or manifest["backfill"]["season"] != evidence["season"]:
        raise ValueError("register capture event/season differs from blank evidence")
    capture, deadline = pd.Timestamp(manifest["captured_at"]), pd.Timestamp(deadlines[0])
    if (capture.tzinfo is None or deadline.tzinfo is None or capture > deadline
            or (deadline - capture).total_seconds() > config.REPLAY_REGISTER_MAX_AGE_HOURS * 3600
            or pd.Timestamp(manifest["event_deadline"]) != deadline):
        raise ValueError("blank register requires a matching recent predeadline capture")
    _, blanks = explicit_blank_outcomes(snapshot, predictions,
        fixtures=pd.read_csv(BytesIO(inputs["fixtures.csv"])), teams=pd.read_csv(BytesIO(inputs["teams.csv"])),
        register=register, season=evidence["season"], first_global_gw=evidence["first_global_gw"], event=event)
    return blanks


_BLANK_ASSETS = ("fixtures.csv", "teams.csv", "bootstrap-static.json.gz", "manifest.json")


def _collection_entries(evidence: dict) -> list[dict] | None:
    """Recognize the opt-in collection without changing single-event admission."""
    if not isinstance(evidence, dict) or "events" not in evidence:
        return None
    if (set(evidence) != {"schema_version", "events"}
            or type(evidence["schema_version"]) is not int or evidence["schema_version"] != 1
            or not isinstance(evidence["events"], list) or not evidence["events"]):
        raise ValueError("invalid blank evidence collection")
    for entry in evidence["events"]:
        if (not isinstance(entry, dict) or set(entry) != {"path", "sha256"}
                or not isinstance(entry["path"], str) or not entry["path"]
                or "\\" in entry["path"] or Path(entry["path"]).is_absolute()
                or any(part in (".", "..") for part in entry["path"].split("/"))
                or Path(entry["path"]).as_posix() != entry["path"]
                or not isinstance(entry["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
            raise ValueError("invalid pinned blank descriptor path or SHA256")
    return evidence["events"]


def _capture_blank_inputs(path: Path) -> dict[str, bytes]:
    inputs = {"blank-evidence.json": path.read_bytes()}
    entries = _collection_entries(json.loads(inputs["blank-evidence.json"]))
    if entries is None:
        inputs.update({name: (path.parent / name).read_bytes() for name in _BLANK_ASSETS})
        return inputs
    root = path.parent.resolve()
    for index, entry in enumerate(entries, 1):
        descriptor = (root / entry["path"]).resolve()
        if not descriptor.is_relative_to(root):
            raise ValueError("blank descriptor escapes collection directory")
        prefix = f"blank-evidence-events/{index:04d}/"
        for name in ("blank-evidence.json", *_BLANK_ASSETS):
            source = descriptor if name == "blank-evidence.json" else descriptor.parent / name
            if not source.resolve().is_relative_to(root):
                raise ValueError("blank evidence asset escapes collection directory")
            inputs[prefix + name] = source.read_bytes()
    _blank_groups(inputs)  # Pin descriptors/assets and reject event collisions before output creation.
    return inputs


def _blank_groups(inputs: dict[str, bytes]) -> list[dict[str, bytes]]:
    """Derive exact evidence coverage solely from immutable captured bytes."""
    if "blank-evidence.json" not in inputs:
        if inputs:
            raise ValueError("unexpected captured blank evidence")
        return []
    entries = _collection_entries(json.loads(inputs["blank-evidence.json"]))
    if entries is None:
        if set(inputs) != {"blank-evidence.json", *_BLANK_ASSETS}:
            raise ValueError("invalid captured single-event evidence coverage")
        return [inputs]
    groups, expected, events, weeks = [], {"blank-evidence.json"}, set(), set()
    for index, entry in enumerate(entries, 1):
        prefix = f"blank-evidence-events/{index:04d}/"
        names = ("blank-evidence.json", *_BLANK_ASSETS)
        expected.update(prefix + name for name in names)
        if any(prefix + name not in inputs for name in names):
            raise ValueError("incomplete captured blank evidence event")
        group = {name: inputs[prefix + name] for name in names}
        if hashlib.sha256(group["blank-evidence.json"]).hexdigest() != entry["sha256"]:
            raise ValueError("blank descriptor SHA256 mismatch")
        descriptor = json.loads(group["blank-evidence.json"])
        if not isinstance(descriptor, dict) or "events" in descriptor:
            raise ValueError("collection entries must be single-event descriptors")
        season, event = descriptor.get("season"), _integer(descriptor.get("event"), "blank event")
        first = _integer(descriptor.get("first_global_gw"), "blank first_global_gw")
        if not isinstance(season, str) or event > config.GWS_PER_SEASON:
            raise ValueError("invalid collection event identity")
        identity, gw = (season, event), first + event - 1
        if identity in events or gw in weeks:
            raise ValueError("duplicate blank evidence event")
        events.add(identity)
        weeks.add(gw)
        pins = descriptor.get("input_sha256")
        if (not isinstance(pins, dict) or set(pins) != set(_BLANK_ASSETS)
                or any(pins[name] != hashlib.sha256(group[name]).hexdigest() for name in _BLANK_ASSETS)):
            raise ValueError("blank evidence asset SHA256 mismatch")
        groups.append(group)
    if set(inputs) != expected:
        raise ValueError("extra captured blank evidence files")
    return groups


def _admit_blank_collection(snapshot: dict, predictions: pd.DataFrame,
                            inputs: dict[str, bytes]) -> pd.DataFrame:
    groups = _blank_groups(inputs)
    blanks = pd.concat([_admit_blank_evidence(snapshot, predictions, group) for group in groups],
                       ignore_index=True)
    if blanks.duplicated(["player_id", "GW"]).any():
        raise ValueError("duplicate blank player/gameweek across evidence events")
    return blanks


def run_replay(snapshot_path: str | Path, predictions_path: str | Path,
               output_dir: str | Path, *, reference_path: str | Path | None = None,
               blank_evidence_path: str | Path | None = None) -> Path:
    """Capture immutable inputs and settle, accepting single-event or pinned collection evidence.

    Collection entries are relative descriptor paths with SHA256 pins; saved
    reporting uses captured bytes and never needs their original source paths.
    """
    output = Path(output_dir).resolve()
    if any(output.is_relative_to(folder.resolve()) for folder in (config.DATASETT_DIR, config.ROOT / "fpl")):
        raise ValueError("replay output cannot target production code or datasets")
    inputs = {"decisions.json": Path(snapshot_path).read_bytes(), "predictions.csv": Path(predictions_path).read_bytes()}
    if reference_path is not None:
        inputs["reference.csv"] = Path(reference_path).read_bytes()
    if blank_evidence_path is not None:
        inputs.update(_capture_blank_inputs(Path(blank_evidence_path)))
    sources, runtime = _source_hashes(), _runtime_versions()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        for name, content in inputs.items():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            with (output / name).open("xb") as stream:
                stream.write(content)
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in inputs.items()}
        write_json_new(output / "started.json", {"status": "STARTED", "input_sha256": hashes,
                       "source_sha256": sources, "runtime": runtime, "promotion_eligible": False})
        snapshot = json.loads(inputs["decisions.json"])
        predictions = pd.read_csv(BytesIO(inputs["predictions.csv"]))
        evidence_inputs = {name: value for name, value in inputs.items()
                           if name not in {"decisions.json", "predictions.csv", "reference.csv"}}
        blanks = _admit_blank_collection(snapshot, predictions, evidence_inputs) if evidence_inputs else None
        if blanks is not None:
            blanks.to_csv(output / "explicit-blank-outcomes.csv", index=False)
        replayed = replay_snapshot(snapshot, predictions, blank_outcomes=blanks)
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
        result["explicit_blank_outcomes"] = 0 if blanks is None else len(blanks)
        result["registration_identity"] = "CAPTURE_TIME_ONLY" if blanks is not None else "NOT ASSESSED"
        result["outcome_evidence_basis"] = "OBSERVED_AND_EXPLICIT_RETROSPECTIVE_BLANKS" if blanks is not None else "OBSERVED_ROWS"
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
    base = {"decisions.json", "predictions.csv"}
    if not isinstance(inputs, dict) or not base <= set(inputs):
        raise ValueError("invalid captured replay input manifest")
    evidence_names = set(inputs) - base - {"reference.csv"}
    captured_evidence_names = {name for name in result["output_sha256"]
                               if name == "blank-evidence.json" or name in _BLANK_ASSETS
                               or name.startswith("blank-evidence-events/")}
    if captured_evidence_names != evidence_names:
        raise ValueError("captured evidence output/input coverage differs")
    for name, digest in result["input_sha256"].items():
        if not isinstance(digest, str) or len(digest) != 64 or result["output_sha256"].get(name) != digest:
            raise ValueError("captured replay input identity changed")
    evidence_inputs = {name: (root / name).read_bytes() for name in evidence_names}
    _blank_groups(evidence_inputs)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Strict saved-decision settlement replay")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("decisions", type=Path)
    run.add_argument("predictions", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reference", type=Path)
    run.add_argument("--blank-evidence", type=Path)
    report = commands.add_parser("report")
    report.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            print(json.dumps(read_replay(args.directory), indent=2))
        else:
            print(run_replay(args.decisions, args.predictions, args.output, reference_path=args.reference,
                             blank_evidence_path=args.blank_evidence))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Settlement replay stopped: {exc}\n")


if __name__ == "__main__":
    main()
