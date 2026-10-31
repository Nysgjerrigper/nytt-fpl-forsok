"""Independently reconstruct paired evidence without repeating fits or solves."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl import config
from fpl.data.origin_prices import build_origin_price_catalog
from fpl.data.provenance import load_frozen_research_dataset
from fpl.model import research_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    registration_path = directory / "registration.json"
    registration = json.loads(registration_path.read_bytes())
    assert summary["status"] in {"complete", "selection_rejected"}
    assert not summary["promotion_eligible"] and not registration["promotion_eligible"]
    assert summary["registration_sha256"] == research_package.file_hash(registration_path)
    for relative, digest in registration["code_sha256"].items():
        assert research_package.file_hash(directory / registration["source_archive"] / relative) == digest
    for path, digest in {**registration["input_sha256"], **summary["immutable_output_sha256"]}.items():
        assert research_package.file_hash(Path(path)) == digest, path
    raw = load_frozen_research_dataset(registration["dataset"])
    assert raw.attrs["frozen_dataset_sha256"] == registration["dataset_sha256"]
    coverage = pd.read_csv(directory / "coverage_mask.csv")
    for phase, (start, end) in registration["windows"].items():
        catalog = pd.read_csv(directory / f"{phase}_origin_prices.csv")
        phase_mask = coverage.loc[coverage.phase == phase]
        target_ids = phase_mask.loc[phase_mask.GW_global.between(start, end), "player_id"].unique()
        rebuilt = build_origin_price_catalog(raw.loc[raw.GW_global <= end], target_ids, start, end)
        pd.testing.assert_frame_equal(catalog, rebuilt, check_dtype=False)
        stats = registration["coverage_counts"][phase]
        assert len(phase_mask) == stats["fixture_rows"]
        assert int(phase_mask.availability_known.sum()) == stats["covered_fixture_rows"]
        expected = raw.loc[raw.GW_global.between(1, end)]
        keys = [column for column in ("player_id", "GW_global", "fixture") if column in phase_mask]
        # Feature construction sorts rows. Compare keyed fixture multisets to
        # establish that missing archive coverage did not delete any outcome row.
        left = phase_mask[keys].sort_values(keys).reset_index(drop=True)
        right = expected[keys].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    selected = summary["selection"]
    advanced = selected["mae_availability"]["total_points"] > selected["mae_full"]["total_points"]
    assert advanced == summary["advanced"]
    lock = json.loads((directory / "selection_lock.json").read_bytes())
    assert lock["advanced"] == advanced
    assert lock["selection_summary_sha256"] == research_package.file_hash(directory / "selection_summary.json")
    assert lock["parameters"] == registration["parameters"]
    assert lock["feature_variants"] == registration["feature_variants"]
    if not advanced:
        assert not summary["evaluation"] and not (directory / "reference").exists()
    results = [("selection", "origin", arm, value) for arm, value in selected.items()]
    for arm, modes in summary["evaluation"].items():
        results.extend(("reference", mode, arm, value) for mode, value in modes.items())
    optimal, budget_checks = 0, 0
    scores = {}
    for phase, mode, arm, result in results:
        location = directory / phase
        start, end = registration["windows"][phase]
        predictions = pd.read_csv(location / result["predictions"])
        research_package.validate_forecasts(predictions, start, end, mode == "origin")
        squads = pd.read_csv(location / result["squads"])
        assert squads.gameweek.tolist() == list(range(start, end + 1))
        assert squads.scoring_protocol.eq("autosubs_v2").all()
        assert squads.points_accounting.eq("net_after_transfer_hits").all()
        assert squads.chip_played.isna().all()
        assert np.allclose(squads.actual_total_points, squads.gross_actual_total_points - 4 * squads.alpha)
        assert squads.actual_total_points.sum() == result["total_points"]
        prefix = location / f"{arm}_{mode}"
        admission = json.loads(prefix.with_suffix(".admission.json").read_bytes())
        assert admission["feature_columns"] == registration["feature_variants"][arm]
        assert admission["parameters"] == registration["parameters"]["catboost_mae"]
        count = prefix.with_suffix(".log").read_text().count("Status: Optimal (")
        assert count == end - start + 1
        optimal += count
        if mode == "origin":
            assert admission["origin_price_policy"] == "origin-current-proxy"
            assert admission["dated_origin_snapshot"] == (arm == "mae_availability")
            assert admission["extra_player_features"] == (registration["extra_player_features"] if arm == "mae_availability" else [])
            keys = ["player_id", "GW", "origin_gw"]
            grouped = predictions.groupby(keys, as_index=False).agg(value=("value", "first"), name=("name", "first"))
            restricted = grouped.loc[grouped.GW.between(start - 1, end)].sort_values(keys)
            metadata = restricted.drop_duplicates("player_id").set_index("player_id")
            assert metadata.name.is_unique, "Cannot reconstruct ambiguous exported squad names"
            reverse = {str(name).strip(): int(code) for code, name in metadata.name.items()}
            deduplicated = restricted.drop_duplicates(["player_id", "GW"])
            prices = deduplicated.pivot(index="player_id", columns="GW", values="value").sort_index(axis=1)
            prices = prices.ffill(axis=1).bfill(axis=1).fillna(50.)
            catalog = pd.read_csv(directory / f"{phase}_origin_prices.csv")
            prior_ids = catalog.groupby("origin_gw").player_id.agg(set).to_dict()
            previous_codes = set()
            previous_bank = 1000.
            for row in squads.to_dict("records"):
                origin = int(row["gameweek"])
                codes = [reverse[name.strip()] for name in row["squad"].split(", ")]
                assert len(codes) == len(set(codes)) == 15
                assert set(codes).issubset(prior_ids[origin])
                assert np.isclose(row["budget_start"], previous_bank)
                starting_value = prices.loc[list(previous_codes), origin].sum() if previous_codes else 0.
                assert np.isclose(prices.loc[codes, origin].sum() + row["budget_end"], previous_bank + starting_value)
                previous_codes = set(codes)
                previous_bank = row["budget_end"]
                budget_checks += 1
        scores[f"{phase}/{arm}/{mode}"] = result["total_points"]
    for phase, modes in [("selection", ["origin"]), ("reference", ["standard", "origin"] if advanced else [])]:
        for mode in modes:
            arms = selected if phase == "selection" else {arm: entries[mode] for arm, entries in summary["evaluation"].items()}
            control, candidate = arms["mae_full"], arms["mae_availability"]
            location = directory / phase
            research_package.validate_paired_forecasts(location / candidate["predictions"], location / control["predictions"])
            rebuilt = research_package.compare_backtests.compare(str(location / candidate["squads"]), str(location / control["squads"]), **registration["comparison"])
            assert rebuilt == summary["comparisons"][f"{phase}_{mode}"]
    assert optimal == (156 if advanced else 32)
    assert config.PRODUCTION_WEIGHT_STRATEGY == "single:catboost"
    report = {"status": "verified", "promotion_eligible": False,
              "summary_sha256": research_package.file_hash(summary_path),
              "registration_sha256": research_package.file_hash(registration_path),
              "optimal_solves": optimal, "independently_reconstructed_current_budgets": budget_checks,
              "immutable_outputs": len(summary["immutable_output_sha256"]), "results": scores,
              "checks": ["source and input/output byte locks", "unchanged phase outcome fixture coverage",
                         "strict selection gate and immutable finalist lock", "identical paired forecast metadata",
                         "prior-price catalogs rebuilt", "selected eligibility and reconstructed current-price budgets",
                         "optimal status, chips disabled, net gross hit accounting", "all bootstrap and sign-test reports"]}
    (directory / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
