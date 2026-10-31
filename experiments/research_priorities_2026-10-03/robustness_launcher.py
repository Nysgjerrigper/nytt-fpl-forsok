"""Predeclare seeds 1/2 with the same seed-zero tuned studies; run serially once."""
from __future__ import annotations

import json
from pathlib import Path
import time

from fpl import config, experiment
from fpl.model import research_package as package


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    parent = Path(__file__).resolve().parent
    protocol_path = parent / "robustness_protocol.json"
    summary_path = parent / "robustness_summary.json"
    seeds = (1, 2)
    if protocol_path.exists() or any((parent / f"selection_seed{seed}").exists() for seed in seeds):
        raise ValueError("preserve existing registered robustness runs; do not overwrite or retry")
    original = json.loads((parent / "selection_seed0/registration.json").read_text())
    dataset = Path(original["dataset"])
    raw, _, columns = package.load_research_features(package.protocol_cutoffs()["selection_max_gw"], dataset)
    seed_zero, tuned_hashes = package.resolve_parameters("selection", 0)
    if (seed_zero != original["parameters"]
            or package.feature_variants(columns) != original["feature_variants"]
            or raw.attrs["frozen_dataset_sha256"] != original["dataset_sha256"]
            or package.protocol_cutoffs() != original["cutoffs"]):
        raise ValueError("robustness repetitions differ from original feature/settings/data protocol")
    resolved = {str(seed): package.resolve_parameters("selection", seed)[0] for seed in seeds}
    protocol = {
        "status": "registered", "issue": 13, "seeds": list(seeds), "parameters": resolved,
        "hypothesis": "Registered tree decision comparisons may vary with refit randomness.",
        "dataset": str(dataset.resolve()), "dataset_sha256": raw.attrs["frozen_dataset_sha256"],
        "code_sha256": package.source_hashes(), "launcher_sha256": package.file_hash(Path(__file__)),
        "tuned_artifact_sha256": tuned_hashes, "feature_variants": original["feature_variants"],
        "cutoffs": original["cutoffs"], "retrain_every": 4, "horizon": 3,
        "comparison": {"n_boot": 10000, "block_len": 3, "seed": "matching refit seed"},
        "selection_rule": "Each seed independently chooses its MAE ablation on earlier selection origin net points; full wins ties. RMSE is predeclared.",
        "promotion_eligible": False, "scoring_protocol": "autosubs_v2",
        "points_accounting": "net_after_transfer_hits", "gap_rel": 0, "chips": "disabled",
        "limitations": ["Preserved seed-zero tuning studies are reused; these are refits, not new tuning.",
                        "Historical selection/reference windows were previously examined.",
                        "Ablation finalists can differ across seeds; do not pool different treatments.",
                        "Seeds share outcomes; seed averages are descriptive, not independent replications.",
                        "Current sources include engineering hardening after seed zero; numerical settings/features match.",
                        "New pinned controls differ from the historical 2088/1901 replay; no promotion claim."],
    }
    write(protocol_path, protocol)
    protocol_sha = package.file_hash(protocol_path)
    experiment.log_result("research_package_seed_robustness_registration",
                          {"protocol": str(protocol_path), **protocol},
                          {"status": "REGISTERED", "promotion_eligible": False})
    started = time.perf_counter()
    summary = {"status": "running", "promotion_eligible": False, "protocol_sha256": protocol_sha, "seeds": {}}

    def check() -> None:
        if (package.source_hashes() != protocol["code_sha256"]
                or package.file_hash(protocol_path) != protocol_sha
                or package.file_hash(dataset) != protocol["dataset_sha256"]
                or package.file_hash(Path(__file__)) != protocol["launcher_sha256"]):
            raise ValueError("registered robustness inputs changed; stop without retry")
        for relative, digest in tuned_hashes.items():
            if package.file_hash(config.ROOT / relative) != digest:
                raise ValueError("pinned tuning changed during robustness repetitions")
        for result in summary["seeds"].values():
            if package.file_hash(Path(result["output_dir"]) / "summary.json") != result["summary_sha256"]:
                raise ValueError("completed seed summary changed during robustness repetitions")

    try:
        for seed in seeds:
            check()
            output = parent / f"selection_seed{seed}"
            result = package.run_package(dataset, output, parameter_mode="selection", seed=seed)
            summary["seeds"][str(seed)] = {"output_dir": str(output), "status": result["status"],
                                          "selected_ablation": result["selected_ablation"],
                                          "comparisons": result["comparisons"],
                                          "summary_sha256": package.file_hash(output / "summary.json")}
            check()
            write(summary_path, summary)
        check()
        summary["status"] = "complete"
    except BaseException as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        write(summary_path, summary)
        experiment.log_result("research_package_seed_robustness_checkpoint",
                              {"protocol": str(protocol_path), "summary_sha256": package.file_hash(summary_path)},
                              {"status": summary["status"].upper(), "promotion_eligible": False})


if __name__ == "__main__":
    main()
