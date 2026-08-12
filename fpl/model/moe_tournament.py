"""Reproducible, pre-registered position-expert tournament plan.

The module writes commands and a manifest rather than hiding long model/MILP runs
inside Python.  This makes every candidate, seed, cutoff and output inspectable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fpl import config
from fpl.model.expert_policy import POSITIONS, parse_expert_map
from fpl.model.metrics import mae, naive_lag1_scale, mase
from fpl.model import models
from fpl.model.mid_gate import select_mid_gate

SEEDS = (0, 1, 2)


def spent_window() -> tuple[int, int]:
    """Return the permanently spent confirmation window for current configuration."""
    return config.season_window("2025-26")


def artifact_hash(payload: object) -> str:
    """Stable SHA-256 identity for an artifact's complete JSON payload."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_tuned_artifact_manifest(path: str | Path, *, experts: tuple[str, ...], seed: int) -> dict[tuple[str, str], str]:
    """Verify a self-hashed selection parameter manifest and return artifact paths."""
    payload = json.loads(Path(path).read_text())
    asserted = payload.pop("sha256", None)
    if asserted != artifact_hash(payload):
        raise ValueError("tuned-artifact manifest hash mismatch")
    seeds = payload.get("seeds", {})
    if set(seeds) != {str(seed)}:
        raise ValueError("tuned-artifact manifest must contain exactly the requested seed")
    result = {}
    if set(seeds[str(seed)]) != set(POSITIONS):
        raise ValueError("tuned-artifact manifest has missing or extra position entries")
    for position in POSITIONS:
        entries = seeds[str(seed)].get(position, {})
        if set(entries) != set(experts):
            raise ValueError("tuned-artifact manifest has missing or extra position/expert entries")
        for expert, record in entries.items():
            if set(record) != {"path", "sha256"}:
                raise ValueError("tuned artifact record must bind only path and sha256")
            artifact = Path(record["path"])
            if not artifact.exists() or hashlib.sha256(artifact.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("tuned artifact bytes do not match manifest")
            result[(position, expert)] = str(artifact)
    return result


def protocol_cutoffs(selection_season: str = "2024-25", selection_weeks: int = 16) -> dict[str, int]:
    """Derive causal research windows from configured season ordinals.

    The selection block immediately precedes the requested season; no caller may
    silently inject raw global GW constants.  The 2025-26 confirmation season is
    permanently spent and rejected by lineage validation below.
    """
    start = config.season_start_gw(selection_season)
    if not 1 <= selection_weeks < config.GWS_PER_SEASON:
        raise ValueError("selection_weeks must be between 1 and GWS_PER_SEASON - 1")
    result = {"discovery_max_gw": start - selection_weeks - 1,
            "selection_min_gw": start - selection_weeks,
            "selection_max_gw": start - 1,
            "backtest_min_gw": start,
            "backtest_max_gw": start + 30}
    for lower, upper in ((result["selection_min_gw"], result["selection_max_gw"]),
                         (result["backtest_min_gw"], result["backtest_max_gw"])):
        spent_min, spent_max = spent_window()
        if lower <= spent_max and upper >= spent_min:
            raise ValueError("derived protocol overlaps permanently spent GW191-221")
    return result


def validate_lineage(frame: pd.DataFrame, cutoffs: dict[str, int]) -> None:
    """Reject OOF evidence without causal, unspent, row-level provenance."""
    required = {"position", "expert", "GW_global", "prediction", "actual_total_points",
                "train_max_gw", "seed", "params_hash", "stage", "mase_scale"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF lineage missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OOF lineage cannot be empty")
    if not frame["GW_global"].between(cutoffs["selection_min_gw"], cutoffs["selection_max_gw"]).all():
        raise ValueError("OOF rows lie outside the registered selection window")
    if not (frame["train_max_gw"] < frame["GW_global"]).all():
        raise ValueError("OOF lineage has a non-causal training cutoff")
    spent_min, spent_max = spent_window()
    if frame["GW_global"].between(spent_min, spent_max).any() or frame["train_max_gw"].between(spent_min, spent_max).any():
        raise ValueError("OOF lineage touches permanently spent GW191-221")
    if not (frame["stage"] == "selection").all() or frame["params_hash"].isna().any():
        raise ValueError("OOF lineage must record selection-stage parameter hashes")


def generate_selection_oof(df: pd.DataFrame, feature_cols: list[str], experts: tuple[str, ...],
                           artifact_dir: str | Path, *, seed: int = 0,
                           cutoffs: dict[str, int] | None = None,
                           tuned_artifacts: dict[tuple[str, str], str] | None = None) -> tuple[pd.DataFrame, Path]:
    """Generate deterministic per-row causal OOF forecasts for every position/expert.

    Each prediction refits on rows strictly before its target GW and records that
    exact cutoff and a hash of the resolved expert defaults.  Outputs are namespaced
    by ``selection/seed-<n>`` so stability evidence cannot be overwritten.
    """
    cutoffs = protocol_cutoffs() if cutoffs is None else dict(cutoffs)
    root = Path(artifact_dir) / "selection" / f"seed-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    records: list[pd.DataFrame] = []
    for position in POSITIONS:
        pos = df[df["position"] == position]
        for expert in experts:
            if expert not in models.REGISTERED_MODEL_NAMES:
                raise ValueError(f"unregistered expert {expert!r}")
            if tuned_artifacts is None or (position, expert) not in tuned_artifacts:
                raise ValueError("tournament OOF requires a validated tuned artifact per position/expert")
            from fpl.model import tuning
            params = tuning.load_validated_params(
                tuned_artifacts[(position, expert)], position=position, model_name=expert,
                seed=seed, stage="selection", max_train_gw=cutoffs["discovery_max_gw"]
            )
            params_hash = artifact_hash({"expert": expert, "position": position, "seed": seed, "params": params})
            for gw in range(cutoffs["selection_min_gw"], cutoffs["selection_max_gw"] + 1):
                train = pos[pos["GW_global"] < gw]
                test = pos[pos["GW_global"] == gw]
                if train.empty or test.empty:
                    continue
                fitted = models.fit_model(expert, train[feature_cols], train["total_points"],
                                          position=position, minutes=train.get("minutes"), gw=train.get("GW_global"),
                                          params=params, seed=seed)
                part = test[["player_id", "position", "GW_global", "total_points", "mins60_rate_roll5"]].copy()
                part["expert"] = expert
                part["prediction"] = np.asarray(fitted.predict(test[feature_cols]), dtype=float)
                part = part.rename(columns={"total_points": "actual_total_points"})
                part["train_max_gw"] = gw - 1
                part["mase_scale"] = naive_lag1_scale(train)
                part["seed"] = seed
                part["params_hash"] = params_hash
                part["stage"] = "selection"
                records.append(part)
    oof = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    validate_lineage(oof, cutoffs)
    path = root / "oof.csv"
    oof.to_csv(path, index=False)
    meta = {"cutoffs": cutoffs, "seed": seed, "experts": list(experts), "oof_sha256": artifact_hash(oof.to_dict("records"))}
    (root / "oof_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return oof, path


def select_from_oof(oof: pd.DataFrame, artifact_dir: str | Path, *, cutoffs: dict[str, int] | None = None) -> Path:
    """Rank causal OOF evidence and freeze a complete champion map with hashes."""
    cutoffs = protocol_cutoffs() if cutoffs is None else dict(cutoffs)
    validate_lineage(oof, cutoffs)
    rows = []
    champions = {}
    for position in POSITIONS:
        for expert, group in oof[oof["position"] == position].groupby("expert", sort=True):
            scale = float(group["mase_scale"].mean())
            score = mase(group["actual_total_points"], group["prediction"], scale)
            rows.append({"position": position, "expert": expert, "mase": score, "mae": mae(group["actual_total_points"], group["prediction"]), "rows": len(group), "eligible": bool(np.isfinite(score)), "lineage_hash": artifact_hash(group.to_dict("records"))})
    ranking = pd.DataFrame(rows).sort_values(["position", "mase", "expert"], kind="stable")
    for position in POSITIONS:
        eligible = ranking[(ranking.position == position) & ranking.eligible]
        if eligible.empty:
            raise ValueError(f"no eligible OOF expert for {position}")
        champions[position] = str(eligible.iloc[0].expert)
    root = Path(artifact_dir) / "selection"
    root.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(root / "expert_ranking.csv", index=False)
    payload = {"cutoffs": cutoffs, "champion_map": champions,
               "oof_sha256": artifact_hash(oof.to_dict("records")),
               "ranking_sha256": artifact_hash(ranking.to_dict("records")), "spent_window": list(spent_window())}
    payload["artifact_sha256"] = artifact_hash(payload)
    path = root / "frozen_selection.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def finalize_seed_artifacts(frozen_selection: str | Path, artifact_dir: str | Path,
                            seed_artifacts: dict[int, dict[str, str]],
                            control_artifacts: dict[int, dict[str, str]]) -> Path:
    """Freeze the only admissible multi-seed finalist/MILP artifact interface.

    Each seed must provide standard and origin MILP CSV paths.  Their content hashes,
    rather than user-supplied point differences, become the promotion input.
    """
    frozen = json.loads(Path(frozen_selection).read_text())
    asserted = frozen.get("artifact_sha256")
    recomputed = artifact_hash({key: value for key, value in frozen.items() if key != "artifact_sha256"})
    if asserted != recomputed or set(frozen.get("champion_map", {})) != set(POSITIONS):
        raise ValueError("frozen selection lacks a complete hash-bound champion map")
    if set(seed_artifacts) != set(SEEDS):
        raise ValueError(f"finalists require exactly registered seeds {SEEDS}")
    if set(control_artifacts) != set(SEEDS):
        raise ValueError(f"controls require exactly registered seeds {SEEDS}")
    def freeze_group(group):
        records = {}
        for seed, paths in group.items():
            if set(paths) != {"standard", "origin"}:
                raise ValueError("each seed requires exactly standard and origin artifacts")
            records[str(seed)] = {}
            for protocol, value in paths.items():
                content = Path(value).read_bytes()
                csv = pd.read_csv(value)
                expected = set(range(protocol_cutoffs()["backtest_min_gw"], protocol_cutoffs()["backtest_max_gw"] + 1))
                gw_col = "gameweek" if "gameweek" in csv else "GW"
                observed = set(csv[gw_col]) if gw_col in csv else set()
                spent_min, spent_max = spent_window()
                if gw_col not in csv or observed != expected or csv[gw_col].duplicated().any() or csv[gw_col].between(spent_min, spent_max).any():
                    raise ValueError("MILP artifact has invalid or spent gameweek lineage")
                records[str(seed)][protocol] = {"path": str(value), "sha256": hashlib.sha256(content).hexdigest()}
        return records
    records = freeze_group(seed_artifacts)
    controls = freeze_group(control_artifacts)
    payload = {"selection_sha256": frozen["artifact_sha256"], "seeds": records, "controls": controls,
               "candidate": "position_map", "allowed_family": ["position_map", "mid_gate"]}
    payload["artifact_sha256"] = artifact_hash(payload)
    path = Path(artifact_dir) / "finalists.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def seed_differences_from_artifacts(finalists: str | Path) -> list[float]:
    """Derive, never accept, seed standard-point differences from recorded CSVs."""
    payload = json.loads(Path(finalists).read_text())
    if set(map(int, payload.get("seeds", {}))) != set(SEEDS) or set(map(int, payload.get("controls", {}))) != set(SEEDS):
        raise ValueError("seed stability needs exactly the registered seed family")
    diffs = []
    for seed in SEEDS:
        record = payload["seeds"][str(seed)]["standard"]
        candidate_path = Path(record["path"])
        if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("candidate seed artifact bytes changed after freeze")
        candidate = pd.read_csv(candidate_path)
        control_record = payload["controls"][str(seed)]["standard"]
        control_path = Path(control_record["path"])
        if hashlib.sha256(control_path.read_bytes()).hexdigest() != control_record["sha256"]:
            raise ValueError("control seed artifact bytes changed after freeze")
        control = pd.read_csv(control_path)
        if "actual_total_points" not in candidate or "actual_total_points" not in control:
            raise ValueError("seed MILP artifacts require actual_total_points")
        diffs.append(float(candidate.actual_total_points.sum() - control.actual_total_points.sum()))
    return diffs


def select_mid_gate_from_oof(training_frame: pd.DataFrame, oof: pd.DataFrame, *, champion: str,
                             candidates: tuple[str, ...], cutoffs: dict[str, int] | None = None):
    """Build a MID gate only from validated, causal selection OOF lineage."""
    cutoffs = protocol_cutoffs() if cutoffs is None else dict(cutoffs)
    validate_lineage(oof, cutoffs)
    mid = oof[oof.position == "MID"].copy()
    if set(mid.expert) != set(candidates) or champion not in candidates:
        raise ValueError("MID gate candidates must exactly match registered OOF experts")
    key = ["player_id", "GW_global", "actual_total_points", "mins60_rate_roll5"]
    counts = mid.groupby(key, dropna=False).expert.nunique()
    if not (counts == len(candidates)).all():
        raise ValueError("MID OOF lineage is incomplete or in-sample for one or more rows")
    wide = mid.pivot(index=key, columns="expert", values="prediction").reset_index()
    source = training_frame[training_frame.position == "MID"]
    if source.empty or int(source.GW_global.max()) > cutoffs["discovery_max_gw"]:
        raise ValueError("MID gate training frame exceeds registered discovery cutoff")
    for expert in candidates:
        wide[f"pred_{expert}"] = wide[expert]
    joined = wide
    if joined["mins60_rate_roll5"].isna().all():
        raise ValueError("MID OOF rows lack deadline-known routing features")
    joined = joined.rename(columns={"actual_total_points": "total_points"})
    return select_mid_gate(source, joined, {name: f"pred_{name}" for name in candidates}, champion=champion,
                           training_max_gw=cutoffs["discovery_max_gw"],
                           provenance={"oof_sha256": artifact_hash(mid.to_dict("records")), "cutoffs": cutoffs})


@dataclass(frozen=True)
class TournamentPlan:
    experts: tuple[str, ...]
    artifact_dir: str
    horizon: int = 3

    @property
    def tuned_artifact_manifest(self) -> Path:
        """Deterministic required manifest location for the selection command."""
        return Path(self.artifact_dir) / "selection" / "tuned_artifacts_seed-0.json"

    def commands(self):
        root = Path(self.artifact_dir)
        commands = []
        for position in POSITIONS:
            for expert in self.experts:
                commands.append(["python", "-m", "fpl.model.tuning", "--position", position,
                                 "--model", expert, "--stage", "selection", "--seed", "0",
                                 "--train-max-gw", str(protocol_cutoffs()["discovery_max_gw"])])
        commands.append(["python", "-m", "fpl.model.moe_tournament", "build-tuned-manifest",
                         "--artifact-dir", str(root), "--experts", ",".join(self.experts),
                         "--seed", "0", "--output", str(self.tuned_artifact_manifest)])
        # The selection report is deliberately separate: its only legal holdout is 137--152.
        commands.append(["python", "-m", "fpl.model.moe_tournament", "select",
                         "--experts", ",".join(self.experts),
                         "--selection-min-gw", str(protocol_cutoffs()["selection_min_gw"]), "--selection-max-gw", str(protocol_cutoffs()["selection_max_gw"]),
                         "--artifact-dir", str(root), "--tuned-artifact-manifest",
                         str(self.tuned_artifact_manifest)])
        return commands

    def prediction_commands(self, champion_map, mid_gate_config=None):
        cutoffs = protocol_cutoffs()
        mapping = ",".join(f"{p}={champion_map[p]}" for p in POSITIONS)
        variants = [("control", None), ("position_map", None)]
        if mid_gate_config:
            variants.append(("mid_gate", mid_gate_config))
        result = []
        for name, gate in variants:
            expert = "GK=catboost,DEF=catboost,MID=catboost,FWD=catboost" if name == "control" else mapping
            for protocol in ("standard", "origin"):
                prediction = Path(self.artifact_dir) / "predictions" / f"{name}_{protocol}.csv"
                cmd = ["python", "-m", "fpl.model.predict", "--start-gw", str(cutoffs["backtest_min_gw"]),
                       "--end-gw", str(cutoffs["backtest_max_gw"]), "--horizon", str(self.horizon),
                       "--expert-map", expert, "--output", str(prediction)]
                if protocol == "origin": cmd.append("--origin-based")
                if gate: cmd += ["--mid-gate-config", str(gate)]
                milp = ["python", "-m", "fpl.milp.optimize", "--predictions-csv", str(prediction),
                        "--start-gw", str(cutoffs["backtest_min_gw"]), "--max-gw", str(cutoffs["backtest_max_gw"]),
                        "--horizon", str(self.horizon), "--wc1-gw", "0", "--wc2-gw", "0", "--tc-gw", "0",
                        "--fh-gw", "0", "--bb-gw", "0", "--output",
                        str(Path(self.artifact_dir) / "milp" / f"{name}_{protocol}.csv")]
                result.extend([cmd, milp])
        return result


def write_plan(plan: TournamentPlan, champion_map=None, mid_gate_config=None):
    root = Path(plan.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    live_protocol = protocol_cutoffs()
    payload = {"protocol": {"discovery_max_gw": live_protocol["discovery_max_gw"],
                            "selection": [live_protocol["selection_min_gw"], live_protocol["selection_max_gw"]],
                            "backtest": [live_protocol["backtest_min_gw"], live_protocol["backtest_max_gw"]], "seeds": list(SEEDS)},
               "plan": asdict(plan), "commands": plan.commands() +
               (plan.prediction_commands(champion_map, mid_gate_config) if champion_map else [])}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    path = root / "tournament_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or plan the causal position-MoE tournament")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--experts", required=False,
                        help="Comma-separated research expert names; defaults to all registered research experts")
    parser.add_argument("--champion-map", type=parse_expert_map)
    parser.add_argument("--mid-gate-config")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("select", nargs="?", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tuned-artifact-manifest", help="Self-hashed selection tuned-artifact manifest; required for select.")
    parser.add_argument("--output", help="Output path for build-tuned-manifest.")
    parser.add_argument("--selection-min-gw", type=int,
                        help="Must equal the runtime season-derived selection start.")
    parser.add_argument("--selection-max-gw", type=int,
                        help="Must equal the runtime season-derived selection end.")
    args = parser.parse_args(argv)
    from fpl.model.models import EXPERT_SPECS
    experts = tuple(args.experts.split(",")) if args.experts else tuple(
        name for name, spec in EXPERT_SPECS.items() if spec.research_only)
    if args.select == "build-tuned-manifest":
        output = Path(args.output) if args.output else TournamentPlan(experts, args.artifact_dir).tuned_artifact_manifest
        unknown = set(experts) - set(models.REGISTERED_MODEL_NAMES)
        if unknown:
            parser.error(f"unregistered experts: {sorted(unknown)}")
        cutoff = protocol_cutoffs()["discovery_max_gw"]
        # Scan only the selection namespace for this seed/cutoff. Production and
        # other research runs (different seed/stage/cutoff) are deliberately ignored.
        observed = {}
        for artifact in config.MODELS_DIR.glob("tuned_params_*.json"):
            try:
                payload = json.loads(artifact.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"malformed tuned artifact {artifact}: {exc}")
            metadata = payload.get("_meta")
            if not isinstance(metadata, dict):
                parser.error(f"tuned artifact has missing or invalid _meta: {artifact}")
            identity = (metadata.get("position"), metadata.get("model"), metadata.get("seed"),
                        metadata.get("stage"), metadata.get("train_max_gw"))
            if (identity[0] not in POSITIONS or identity[1] not in models.REGISTERED_MODEL_NAMES
                    or not isinstance(identity[2], int) or identity[3] not in {"discovery", "selection", "finalist"}
                    or not isinstance(identity[4], int)):
                parser.error(f"tuned artifact has invalid namespace metadata: {artifact}")
            expected_name = f"tuned_params_{identity[0]}_{identity[1]}.json"
            if artifact.name != expected_name:
                parser.error(f"tuned artifact filename disagrees with metadata: {artifact}")
            if (metadata.get("seed"), metadata.get("stage"), metadata.get("train_max_gw")) == (args.seed, "selection", cutoff):
                key = (metadata.get("position"), metadata.get("model"))
                if None in key or key in observed:
                    parser.error("invalid or duplicate tuned artifact in selection namespace")
                observed[key] = artifact
        expected_keys = {(position, expert) for position in POSITIONS for expert in experts}
        if set(observed) != expected_keys:
            parser.error("selection tuned-artifact namespace has missing or extra position/expert entries")
        entries = {str(args.seed): {}}
        for position in POSITIONS:
            entries[str(args.seed)][position] = {}
            for expert in experts:
                artifact = observed[(position, expert)]
                from fpl.model import tuning
                try:
                    params = tuning.load_validated_params(artifact, position=position, model_name=expert,
                                                          seed=args.seed, stage="selection", max_train_gw=cutoff)
                    if models.EXPERT_SPECS[expert].search_space is not None and not params:
                        raise ValueError("selection tuned artifact has an empty required parameter payload")
                except ValueError as exc:
                    parser.error(str(exc))
                entries[str(args.seed)][position][expert] = {"path": str(artifact),
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
        payload = {"seeds": entries}
        payload["sha256"] = artifact_hash(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Wrote tuned artifact manifest to {output}")
        return
    if args.select is not None:
        if args.select != "select":
            parser.error("subcommand must be select or build-tuned-manifest")
        from fpl import features
        if not args.tuned_artifact_manifest:
            parser.error("select requires --tuned-artifact-manifest")
        raw = pd.read_csv(config.MASTER_DATASET_PATH, low_memory=False)
        frame = features.build_feature_frame(raw)
        cutoffs = protocol_cutoffs()
        # Explicit user bounds can only restate the registered season-derived window.
        supplied = (args.selection_min_gw, args.selection_max_gw)
        expected = (cutoffs["selection_min_gw"], cutoffs["selection_max_gw"])
        if supplied != (None, None) and supplied != expected:
            parser.error("selection bounds must equal the configured season-derived protocol")
        tuned = load_tuned_artifact_manifest(args.tuned_artifact_manifest, experts=experts, seed=args.seed)
        oof, oof_path = generate_selection_oof(frame, features.feature_columns(frame), experts,
                                               args.artifact_dir, seed=args.seed, cutoffs=cutoffs,
                                               tuned_artifacts=tuned)
        frozen = select_from_oof(oof, args.artifact_dir, cutoffs=cutoffs)
        print(f"Wrote causal OOF lineage to {oof_path}")
        print(f"Wrote frozen champion selection to {frozen}")
        return
    path = write_plan(TournamentPlan(experts, args.artifact_dir, args.horizon),
                      args.champion_map, args.mid_gate_config)
    print(f"Wrote reproducible tournament manifest to {path}")


if __name__ == "__main__":
    main()
