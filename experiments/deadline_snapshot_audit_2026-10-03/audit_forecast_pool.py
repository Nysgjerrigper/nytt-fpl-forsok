"""Offline roster coverage audit; no outcome scores or optimizer evaluation."""
import argparse
import json
from pathlib import Path

from fpl.data.deadline_snapshots import load_deadline_rosters, SnapshotUnavailableError
from fpl.data.provenance import load_frozen_research_dataset
from fpl.model import research_package
from fpl import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.FROZEN_RESEARCH_DATASET_PATH)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    proof = root / "wayback_verification.json"
    digest = research_package.file_hash(proof)
    raw = load_frozen_research_dataset(args.dataset)
    raw = raw.loc[raw.GW_global <= 183].copy()
    rows = []
    for origin in range(137, 184):
        try:
            roster = load_deadline_rosters(proof, root / "inventory/wayback_index.json", digest, [origin])
        except SnapshotUnavailableError as exc:
            rows.append({"origin": origin, "status": "missing", "reason": exc.reason})
            continue
        past = raw.loc[raw.GW_global < origin]
        # Exactly the current live snapshot's active-code criterion, without
        # constructing any feature frame or reading any current/future outcome.
        latest = past.groupby("player_id").GW_global.max()
        active = set(latest.index[latest > past.GW_global.max() - config.GWS_PER_SEASON])
        registered = set(roster.player_id)
        absent = roster.loc[~roster.player_id.isin(active)]
        no_history = absent.loc[~absent.player_id.isin(latest.index)]
        rows.append({"origin": origin, "status": "admitted", "registered_players": len(roster),
                     "active_historical_codes": len(active),
                     "registered_absent_from_snapshot": len(absent),
                     "registered_without_any_prior_row": len(no_history),
                     "snapshot_codes_absent_from_capture": len(active - registered),
                     "missing_code_examples": sorted(absent.player_id.tolist())[:10],
                     "missing_status_counts": absent.status.value_counts().to_dict(),
                     "excluded_managers": roster.attrs["excluded_nonplayer_count"]})
    if research_package.file_hash(proof) != digest:
        raise ValueError("archive proof changed during roster audit")
    result = {"status": "complete", "proof_sha256": digest,
              "dataset_sha256": raw.attrs["frozen_dataset_sha256"],
              "code_sha256": research_package.source_hashes(), "origins": rows,
              "limitations": ["Crawl-time roster, not exact deadline registration.",
                              "Codes absent from a capture can be historical departures; this does not establish live eligibility.",
                              "Counts repeat players across rounds and are not independent observations.",
                              "No new forecast, realized-point or cold-prior calibration result."]}
    (root / "forecast_pool_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for start, end in [(137, 152), (153, 183)]:
        phase = [row for row in rows if start <= row["origin"] <= end and row["status"] == "admitted"]
        counts = [row["registered_absent_from_snapshot"] for row in phase]
        print(start, end, "admitted", len(phase), "registered missing range", min(counts), max(counts))


if __name__ == "__main__":
    main()
