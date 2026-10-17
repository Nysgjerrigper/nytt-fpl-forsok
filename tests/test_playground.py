"""Training Ground receipts, failure recovery and comparable league observations."""

import json

import pandas as pd
import pytest

from fpl import playground
from fpl.artifacts import sha256_file


def _edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    return value


def _entries(root):
    return [entry for group in playground.league(root) for entry in group["entries"]]


def _historical(tmp_path, name="historical", *, purpose="registered_research", points=52,
                metric="actual_net_points", origin="origin", control_points=50, weeks=(153, 154, 155)):
    """Construct independent literal contracts and exact artifact bytes."""
    directory = tmp_path / name
    directory.mkdir()
    sides = {}
    for side, score in (("candidate", points), ("control", control_points)):
        csv = directory / f"{side}.csv"
        pd.DataFrame({"gameweek": weeks, "actual_total_points": [score] * len(weeks),
                      "alpha": [0] * len(weeks), "scoring_protocol": "autosubs_v2"}).to_csv(csv, index=False)
        artifacts = {}
        for artifact in ("dataset", "outcomes", "player_pool", "predictions", "parameters", "features",
                         "model_source", "optimizer_source", "scorer_source"):
            path = directory / f"{side}-{artifact}.txt"
            path.write_text(f"frozen {artifact}")
            artifacts[artifact] = {"path": path.name, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": 1, "purpose": purpose, "run_id": side,
            "intervention": "forecast", "backtest_sha256": sha256_file(csv), "metric": metric,
            "context": {
                "season": "2024-25", "start_season": "2020-21", "gameweeks": list(weeks),
                "information_protocol": origin, "horizon": 3, "scoring_protocol": "autosubs_v2",
                "price_policy": "static", "sell_price_policy": "buy_price",
                "initial_state": {"mode": "fresh", "budget": 1000}, "rule_set": "legacy",
                "chip_inventory": {"enabled": []}, "runtime": {"solver": "highs", "gap_rel": 0},
            }, "artifacts": artifacts,
        }
        contract = directory / f"{side}-contract.json"
        contract.write_text(json.dumps(manifest))
        sides[side] = {"name": f"{name}-{side}",
                       "input": {"path": csv.name, "sha256": sha256_file(csv)},
                       "contract": {"path": contract.name, "sha256": sha256_file(contract)}}
    spec = {"schema_version": 1, "id": name, "purpose": "historical_reanalysis",
            "hypothesis": "Saved outcomes test contract grouping, not forecast superiority.",
            "metric": metric, **sides, "bootstrap": {"n_boot": 20, "block_len": 2, "seed": 0}}
    path = directory / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_demo_preflight_does_not_create_run_outputs(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = playground.validate_spec(spec)
    assert result["purpose"] == "framework_validation"
    assert "synthetic" in result["hypothesis"]
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert pd.read_csv(spec.parent / "candidate.csv").gameweek.tolist() == [153, 154, 155]


def test_completed_attempt_is_immutable_verified_and_reused(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    directory = playground.run_experiment(spec, runs_dir=root)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert playground.run_experiment(spec, runs_dir=root) == directory
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert {"started.json", "complete.json", "candidate.csv", "control.csv", "scorecard.json", "REPORT.md"} <= set(before)
    assert not (directory / "failed.json").exists()
    card = playground.read_run(directory)
    assert card["status"] == "EXHIBITION"
    assert card["decision_outcomes"]["total_a"] == 171
    assert card["decision_outcomes"]["total_b"] == 170
    assert card["decision_outcomes"]["total_diff"] == 1
    entry = _entries(root)[0]
    assert entry["rank"] is None and entry["status"] == "EXHIBITION"


@pytest.mark.parametrize("output", ["candidate.csv", "control.csv", "comparison.json", "scorecard.json", "REPORT.md", "spec.json"])
def test_corrupted_completed_output_is_rejected_on_read_and_reuse(tmp_path, output):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    directory = playground.run_experiment(spec, runs_dir=root)
    with (directory / output).open("a") as stream:
        stream.write("\nchanged")
    with pytest.raises(ValueError, match="changed"):
        playground.read_run(directory)
    with pytest.raises(ValueError, match="changed"):
        playground.run_experiment(spec, runs_dir=root)


def test_changed_source_bytes_cannot_use_old_pin(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    csv = spec.parent / "candidate.csv"
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes changed"):
        playground.validate_spec(spec)
    with pytest.raises(ValueError, match="bytes changed"):
        playground.run_experiment(spec, runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_repinning_changed_inputs_cannot_reuse_completed_attempt(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    playground.run_experiment(spec, runs_dir=root)
    csv = spec.parent / "candidate.csv"
    csv.write_text(csv.read_text() + "\n")
    _edit(spec, lambda value: value["candidate"]["input"].__setitem__("sha256", sha256_file(csv)))
    with pytest.raises(ValueError, match="different inputs"):
        playground.run_experiment(spec, runs_dir=root)


def test_failure_is_preserved_and_explicit_new_attempt_can_retry(tmp_path, monkeypatch):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    original = playground.compare
    def fail(*args, **kwargs):
        raise ValueError("deliberate comparison failure after preflight")
    monkeypatch.setattr(playground, "compare", fail)
    with pytest.raises(ValueError, match="deliberate"):
        playground.run_experiment(spec, runs_dir=root)
    directory = root / "synthetic-exhibition" / "first"
    assert json.loads((directory / "failed.json").read_text())["type"] == "ValueError"
    assert (directory / "started.json").exists()
    assert not (directory / "complete.json").exists()
    assert _entries(root)[0]["status"] == "FAILED"
    with pytest.raises(FileExistsError, match="new --attempt"):
        playground.run_experiment(spec, runs_dir=root)
    monkeypatch.setattr(playground, "compare", original)
    retry = playground.run_experiment(spec, attempt="retry", runs_dir=root)
    assert retry != directory and (retry / "complete.json").exists()
    assert (directory / "failed.json").exists()


def test_incomplete_attempt_is_visible_and_cannot_be_reused(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    directory = root / "synthetic-exhibition" / "first"
    directory.mkdir(parents=True)
    (directory / "started.json").write_text("{}")
    assert _entries(root)[0]["status"] == "INCOMPLETE"
    assert _entries(root)[0]["rank"] is None
    with pytest.raises(FileExistsError):
        playground.run_experiment(spec, runs_dir=root)


def test_retries_are_one_observation_not_additional_evidence(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    playground.run_experiment(spec, runs_dir=root)
    playground.run_experiment(spec, attempt="retry", runs_dir=root)
    assert len(_entries(root)) == 1


def test_unverified_historical_scores_are_unranked(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    _edit(spec, lambda value: value.__setitem__("purpose", "historical_reanalysis"))
    root = tmp_path / "runs"
    directory = playground.run_experiment(spec, runs_dir=root)
    assert playground.read_run(directory)["status"] == "UNVERIFIED"
    assert _entries(root)[0]["rank"] is None


def test_framework_manifest_cannot_be_relabelled_historical_research(tmp_path):
    spec = _historical(tmp_path, purpose="framework_validation")
    with pytest.raises(ValueError, match="purpose|framework|research"):
        playground.validate_spec(spec)


def test_contract_validated_historical_entries_rank_and_ties_share_rank(tmp_path):
    root = tmp_path / "runs"
    for name, score in (("leader", 54), ("tied", 54), ("third", 52)):
        spec = _historical(tmp_path, name, points=score)
        # A different source identity makes the tied result a distinct challenger.
        if name == "tied":
            artifact = spec.parent / "candidate-predictions.txt"
            artifact.write_text("different tied forecast")
            contract = spec.parent / "candidate-contract.json"
            _edit(contract, lambda c: c["artifacts"]["predictions"].__setitem__("sha256", sha256_file(artifact)))
            _edit(spec, lambda s: s["candidate"]["contract"].__setitem__("sha256", sha256_file(contract)))
        playground.run_experiment(spec, runs_dir=root)
    groups = playground.league(root)
    assert len(groups) == 1
    assert [row["rank"] for row in groups[0]["entries"]] == [1, 1, 3]
    assert all(row["status"] == "CONTRACT_VALIDATED" for row in groups[0]["entries"])


@pytest.mark.parametrize("change", ["metric", "origin", "control", "weeks"])
def test_incompatible_comparisons_have_separate_competitions(tmp_path, change):
    root = tmp_path / "runs"
    playground.run_experiment(_historical(tmp_path, "first"), runs_dir=root)
    kwargs = {"metric": "actual_total_points"} if change == "metric" else (
        {"origin": "standard"} if change == "origin" else (
            {"control_points": 49} if change == "control" else {"weeks": (153, 154)}))
    playground.run_experiment(_historical(tmp_path, "second", **kwargs), runs_dir=root)
    assert len(playground.league(root)) == 2


def test_cli_validate_report_league_and_clean_errors(tmp_path, capsys):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    playground.main(["validate", str(spec)])
    assert "PASS" in capsys.readouterr().out
    playground.main(["run", str(spec), "--runs-dir", str(root)])
    directory = root / "synthetic-exhibition" / "first"
    assert str(directory) in capsys.readouterr().out
    playground.main(["report", str(directory)])
    assert "EXHIBITION" in capsys.readouterr().out
    playground.main(["league", "--runs-dir", str(root)])
    assert "EXHIBITION" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        playground.main(["validate", str(tmp_path / "missing.json")])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "Training Ground stopped:" in stderr and "Traceback" not in stderr


def test_league_markdown_shows_exhibition_without_rank_and_json_cli(tmp_path, capsys):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    playground.run_experiment(spec, runs_dir=root)
    markdown = playground.render_league(playground.league(root))
    assert "| — | synthetic-candidate | EXHIBITION | 171.000 | +1.000 |" in markdown
    assert "Repeated attempts are not independent evidence" in markdown
    playground.main(["league", "--runs-dir", str(root), "--json"])
    groups = json.loads(capsys.readouterr().out)
    assert groups[0]["entries"][0]["rank"] is None
    assert "No captured experiments yet" in playground.render_league([])


def test_rounds_beyond_frozen_manifest_cutoff_are_rejected_before_run_creation(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    cutoff = playground.load_frozen_dataset_manifest()["max_gw_global"]
    for side in ("candidate", "control"):
        source = spec.parent / f"{side}.csv"
        frame = pd.read_csv(source)
        frame["gameweek"] = [cutoff - 1, cutoff, cutoff + 1]
        frame.to_csv(source, index=False)
        _edit(spec, lambda value, side=side, source=source:
              value[side]["input"].__setitem__("sha256", sha256_file(source)))
    with pytest.raises(ValueError, match="prospective"):
        playground.validate_spec(spec)
    runs = tmp_path / "runs"
    with pytest.raises(ValueError, match="prospective"):
        playground.run_experiment(spec, runs_dir=runs)
    assert not runs.exists()


def test_historical_explicit_attempt_retries_are_one_ranked_observation(tmp_path):
    spec = _historical(tmp_path)
    root = tmp_path / "runs"
    first = playground.run_experiment(spec, runs_dir=root)
    retry = playground.run_experiment(spec, attempt="explicit-retry", runs_dir=root)
    assert first != retry
    assert (first / "complete.json").exists() and (retry / "complete.json").exists()
    entries = _entries(root)
    assert len(entries) == 1
    assert entries[0]["status"] == "CONTRACT_VALIDATED"
    assert entries[0]["rank"] == 1


def test_report_uses_captured_outputs_after_original_inputs_are_removed(tmp_path, capsys):
    spec = _historical(tmp_path)
    root = tmp_path / "runs"
    directory = playground.run_experiment(spec, runs_dir=root)
    card = playground.read_run(directory)
    # Retain the spec to establish that preflight fails on missing source bytes,
    # rather than merely failing to locate the experiment definition.
    for source in spec.parent.iterdir():
        if source != spec:
            source.unlink()
    assert playground.read_run(directory) == card
    playground.main(["report", str(directory)])
    assert "CONTRACT_VALIDATED" in capsys.readouterr().out
    assert len(_entries(root)) == 1
    with pytest.raises(FileNotFoundError):
        playground.validate_spec(spec)
    with pytest.raises(FileNotFoundError):
        playground.run_experiment(spec, runs_dir=root)


def test_manifest_labels_and_whitespace_do_not_create_extra_evidence(tmp_path):
    spec = _historical(tmp_path)
    root = tmp_path / "runs"
    playground.run_experiment(spec, runs_dir=root)
    for side in ("candidate", "control"):
        manifest = spec.parent / f"{side}-contract.json"
        value = json.loads(manifest.read_text())
        value["run_id"] += "-relabeled"
        manifest.write_text(json.dumps(value, indent=4) + "\n\n")
        _edit(spec, lambda value, side=side, manifest=manifest:
              value[side]["contract"].__setitem__("sha256", sha256_file(manifest)))
    playground.run_experiment(spec, attempt="relabeled", runs_dir=root)
    groups = playground.league(root)
    assert len(groups) == 1
    assert len(groups[0]["entries"]) == 1


@pytest.mark.parametrize("protected_target", ["dataset", "source"])
def test_outputs_cannot_target_protected_data_or_source_directories(tmp_path, monkeypatch, protected_target):
    spec = playground.init_demo(tmp_path / "demo")
    repository = tmp_path / "repository"
    dataset = repository / "Datasett"
    monkeypatch.setattr(playground.config, "ROOT", repository)
    monkeypatch.setattr(playground.config, "DATASETT_DIR", dataset)
    destination = (dataset if protected_target == "dataset" else repository / "fpl") / "forbidden-runs"
    with pytest.raises(ValueError, match="production code or dataset"):
        playground.run_experiment(spec, runs_dir=destination)
    assert not destination.exists()
    assert not repository.exists()


@pytest.mark.parametrize("attempt", ["../escape", "../../escape", "/absolute", ".", "..", ""])
def test_attempt_slug_rejects_path_traversal_before_output_creation(tmp_path, attempt):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="attempt"):
        playground.run_experiment(spec, attempt=attempt, runs_dir=root)
    assert not root.exists()


def test_symlinked_experiment_parent_cannot_escape_runs_root(tmp_path):
    spec = playground.init_demo(tmp_path / "demo")
    root = tmp_path / "runs"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "synthetic-exhibition").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes runs root"):
        playground.run_experiment(spec, runs_dir=root)
    assert not list(outside.iterdir())
