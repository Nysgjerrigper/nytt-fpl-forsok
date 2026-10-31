"""Repeat the saved archive-admission audit offline, without models or outcomes.

Local ignored payloads and indexes are required. This checks the preserved proof
chain; it does not fetch or renew independent Internet Archive evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fpl import config
from fpl.data.deadline_snapshots import SnapshotUnavailableError, load_deadline_rosters
from fpl.data.provenance import FROZEN_MANIFEST_PATH


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(audit_dir: Path) -> dict:
    """Reconstruct each saved admission, preserving all used input/source hashes."""
    audit_dir = audit_dir.resolve()
    source_hashes = {str(path.relative_to(REPO)): _sha(path) for path in sorted((REPO / "fpl").rglob("*.py"))}
    source_hashes[str(Path(__file__).resolve().relative_to(REPO))] = _sha(Path(__file__))
    inputs: dict[str, str] = {}

    def bind(path: Path, expected: str | None = None) -> None:
        path = path.resolve()
        digest = _sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Input SHA256 mismatch: {path}")
        label = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        if label in inputs and inputs[label] != digest:
            raise ValueError(f"Input changed during verification: {path}")
        inputs[label] = digest

    def local(relative: str) -> Path:
        path = (audit_dir / relative).resolve()
        if not path.is_relative_to(audit_dir):
            raise ValueError(f"Audit path escapes its directory: {relative}")
        return path

    bind(FROZEN_MANIFEST_PATH)
    saved_path = audit_dir / "admission_results.json"
    bind(saved_path)
    saved = json.loads(saved_path.read_bytes())
    if saved["status"] != "complete":
        raise ValueError("Saved admission must be complete")
    proofs = {}
    verified_counts = {}
    capture_ages = {}
    for name, expected in saved["proof_sha256"].items():
        path = local(name)
        bind(path, expected)
        proof = json.loads(path.read_bytes())
        index_name = "discovery_wayback_index.json" if name.startswith("discovery_") else "wayback_index.json"
        index_path = local(f"inventory/{index_name}")
        bind(index_path, proof["index"]["sha256"])
        proofs[name] = (path, index_path, expected)
        verified_counts[name] = len(proof["captures"])
        for capture in proof["captures"]:
            bind(local(capture["original_payload"]), capture["original_payload_sha256"])
            bind(local(capture["manifest"]), capture["manifest_sha256"])
            origin = config.season_start_gw(capture["season"]) + capture["event"] - 1
            if origin in capture_ages:
                raise ValueError(f"Duplicate proof capture for origin {origin}")
            captured = datetime.fromisoformat(capture["captured_at"].replace("Z", "+00:00"))
            deadline = datetime.fromisoformat(capture["deadline"].replace("Z", "+00:00"))
            capture_ages[origin] = (deadline - captured).total_seconds() / 3600
    if verified_counts != {"discovery_wayback_verification.json": 96, "wayback_verification.json": 47}:
        raise ValueError(f"Verified capture counts differ: {verified_counts}")
    # Bind the pinned downloads too, including repository compressed payloads and
    # manifests. The loader independently verifies original CDX SHA1/decoded SHA256.
    for name in ("download_manifest.json", "discovery_download_manifest.json"):
        manifest_path = local(name)
        bind(manifest_path)
        for record in json.loads(manifest_path.read_bytes())["files"]:
            bind(local(record["local_path"]), record["sha256"])

    reconstructed = []
    for origin in range(1, 184):
        name = "discovery_wayback_verification.json" if origin <= 136 else "wayback_verification.json"
        path, index_path, expected = proofs[name]
        try:
            roster = load_deadline_rosters(path, index_path, expected, [origin])
        except SnapshotUnavailableError as exc:
            row = {"origin": origin, "status": "missing", "reason": exc.reason}
        else:
            ages = roster.snapshot_age_hours.unique()
            if len(ages) != 1:
                raise ValueError(f"Origin {origin} lacks a single capture age")
            row = {"origin": origin, "status": "admitted", "players": len(roster),
                   "age_hours": float(ages[0]), "excluded_managers": roster.attrs["excluded_nonplayer_count"]}
        reconstructed.append(row)
    if reconstructed != saved["origins"]:
        mismatch = [row["origin"] for row, previous in zip(reconstructed, saved["origins"]) if row != previous]
        raise ValueError(f"Saved admissions differ (counts/reasons/ages/manager exclusions): {mismatch}")
    total = sum(row["status"] == "admitted" for row in reconstructed)
    if total != saved["admitted_count"] or total != 122:
        raise ValueError("Total admitted count differs")
    windows = {}
    for label, start, end, admitted_count in (("discovery", 1, 136, 78), ("selection", 137, 152, 16), ("reference", 153, 183, 28)):
        rows = reconstructed[start - 1:end]
        summary = {"origins": len(rows), "admitted": sum(row["status"] == "admitted" for row in rows),
                   "stale": sum(row.get("reason", "").startswith("capture age ") for row in rows),
                   "no_verified_capture": sum(row.get("reason") == "no verified capture" for row in rows),
                   "excluded_managers": sum(row.get("excluded_managers", 0) for row in rows)}
        if summary["admitted"] != admitted_count:
            raise ValueError(f"Unexpected {label} coverage: {summary}")
        windows[label] = summary
    for origin, row in enumerate(reconstructed, 1):
        if row["status"] == "admitted" and row["excluded_managers"] != (20 if origin >= 175 else 0):
            raise ValueError(f"Unexpected manager exclusions at origin {origin}")
    for label, digest in inputs.items():
        path = Path(label) if Path(label).is_absolute() else REPO / label
        if _sha(path) != digest:
            raise ValueError(f"Input drift during verification: {label}")
    for label, digest in source_hashes.items():
        if _sha(REPO / label) != digest:
            raise ValueError(f"Source drift during verification: {label}")
    return {"status": "PASS", "scope": "offline archive admission; no models or outcomes",
            "max_age_hours": config.DEADLINE_SNAPSHOT_MAX_AGE_HOURS, "admitted_count": total,
            "verified_capture_counts": verified_counts, "windows": windows,
            "stale_capture_age_hours": {str(row["origin"]): capture_ages[row["origin"]] for row in reconstructed
                                        if row.get("reason", "").startswith("capture age ")},
            "origins": reconstructed, "input_sha256": inputs, "source_sha256": source_hashes}


def main() -> None:
    """Write a deterministic local report only after exact reconstruction succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.audit_dir)
    output = args.output or args.audit_dir / "admission_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {result['admitted_count']}/183 origins admitted; report {output}")


if __name__ == "__main__":
    main()
