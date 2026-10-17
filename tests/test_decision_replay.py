"""Hand-calculated settlement and saved-input admission; no optimizer is run."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from fpl.artifacts import sha256_file
from fpl.milp import replay, solio


def outcomes():
    return pd.DataFrame({"player_id": list(range(1, 16)), "GW": 141, "fixture": 1001,
                         "position": ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                         "team": [f"club{p % 5}" for p in range(15)],
                         "actual_total_points": 1., "minutes": 90.})


def snapshot(chip=None, alpha=0):
    return {"schema_version": 1, "settlement_completed": False, "decisions": [{
        "gameweek": 141, "squad": list(range(1, 16)),
        "lineup": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
        "bench_order": [2, 6, 7, 12], "captain": [8], "vice_captain": [9],
        "chip_played": chip, "alpha": alpha}]}


@pytest.mark.parametrize("chip,gross", [(None, 12), ("TC", 13), ("TC_player", 13), ("BB", 16), ("FH", 12)])
def test_normalized_chips_and_explicit_hits(chip, gross):
    scored = replay.replay_snapshot(snapshot(chip, alpha=1), outcomes())
    assert scored.actual_total_points.tolist() == [gross]
    assert scored.actual_net_points.tolist() == [gross - 4]
    assert scored.outcome_coverage_assessment.tolist() == ["COMPLETE_SELECTED_SQUAD"]
    assert scored.missing_outcome_players.tolist() == [0]


def test_triple_bonus_follows_vice_after_captain_dnp_and_priority_substitution():
    values = outcomes()
    values.loc[values.player_id == 8, ["actual_total_points", "minutes"]] = [0, 0]
    values.loc[values.player_id == 9, "actual_total_points"] = 5
    # Ten appearing starters score14; first outfield sub scores1; vice adds10.
    result = replay.replay_snapshot(snapshot("TC"), values)
    assert result.actual_total_points.iloc[0] == 25


def test_negative_realized_points_are_preserved():
    result = replay.replay_snapshot(snapshot(alpha=1), outcomes().assign(actual_total_points=-1.))
    assert result.actual_total_points.iloc[0] == -12
    assert result.actual_net_points.iloc[0] == -16


def test_dgw_sums_fixtures_but_not_repeated_origin_outcomes():
    values = pd.concat([outcomes().assign(fixture=f) for f in (1001, 1002)], ignore_index=True)
    panel = pd.concat([values.assign(origin_gw=o) for o in (140, 141)], ignore_index=True)
    result = replay.replay_snapshot(snapshot("TC"), panel)
    assert result.actual_total_points.iloc[0] == 26
    panel.loc[0, "actual_total_points"] = 9
    with pytest.raises(ValueError, match="disagree across"):
        replay.replay_snapshot(snapshot(), panel)


def test_missing_selected_bench_player_is_unknown_not_implicit_blank():
    values = outcomes().loc[lambda frame: frame.player_id != 12]
    with pytest.raises(ValueError, match="missing selected-player.*12"):
        replay.replay_snapshot(snapshot(), values)
    decisions = pd.DataFrame(snapshot()["decisions"])
    compatibility = solio.score_decisions(decisions, values)
    assert compatibility.actual_total_points.iloc[0] == 12
    assert compatibility.outcome_coverage_assessment.iloc[0] == "LEGACY_ZERO_FILL"
    assert compatibility.missing_outcome_players.iloc[0] == 1
    with pytest.raises(ValueError, match="Missing outcomes.*12"):
        solio.score_decisions(decisions, values, require_complete_outcomes=True)


@pytest.mark.parametrize("change", ["duplicate_id", "display_names", "overlapping_bench", "captain_outside",
    "same_vice", "wrong_formation", "unknown_chip", "post_settlement", "duplicate_gw", "scalar_captain"])
def test_invalid_stable_snapshot_structure_rejected(change):
    state = snapshot()
    row = state["decisions"][0]
    if change == "duplicate_id": row["squad"][0] = 2
    elif change == "display_names": row["squad"][0] = "Player One"
    elif change == "overlapping_bench": row["bench_order"][1] = 3
    elif change == "captain_outside": row["captain"] = [12]
    elif change == "same_vice": row["vice_captain"] = [8]
    elif change == "wrong_formation":
        row["lineup"] = [1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]
        row["bench_order"] = [2, 5, 6, 7]
    elif change == "unknown_chip": row["chip_played"] = "FALLBACK_NO_TRANSFERS"
    elif change == "post_settlement": row["actual_total_points"] = 12
    elif change == "duplicate_gw": state["decisions"].append(deepcopy(row))
    else: row["captain"] = 8
    with pytest.raises(ValueError):
        replay.replay_snapshot(state, outcomes())


@pytest.mark.parametrize("change", ["club_limit", "position", "club_ambiguity", "id_bool", "gw_mapping", "duplicate_fixture"])
def test_outcome_identities_and_static_roster_admission(change):
    values = outcomes()
    if change == "club_limit": values["team"] = "Same Club"
    elif change == "position": values.loc[0, "position"] = "DEF"
    elif change == "club_ambiguity":
        values = pd.concat([values, values.iloc[[0]].assign(fixture=1002, team="Other Club")], ignore_index=True)
    elif change == "id_bool":
        values["player_id"] = values.player_id.astype(object)
        values.loc[0, "player_id"] = True
    elif change == "gw_mapping": values["GW_global"] = 1
    else: values = pd.concat([values, values.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        replay.replay_snapshot(snapshot(), values)


@pytest.mark.parametrize("column,value", [("actual_total_points", True), ("actual_total_points", np.inf),
    ("actual_total_points", "1"), ("minutes", -1), ("minutes", np.nan)])
def test_invalid_realized_values_fail_before_scoring(column, value):
    values = outcomes().astype({column: object})
    values.loc[0, column] = value
    with pytest.raises(ValueError):
        replay.replay_snapshot(snapshot(), values)


@pytest.mark.parametrize("alpha", [True, -1, 1.5, np.nan])
def test_invalid_hit_counts_rejected(alpha):
    with pytest.raises(ValueError):
        replay.replay_snapshot(snapshot(alpha=alpha), outcomes())


def _inputs(tmp_path):
    state = tmp_path / "snapshot.json"
    predictions = tmp_path / "forecasts.csv"
    state.write_text(json.dumps(snapshot()))
    outcomes().to_csv(predictions, index=False)
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"gameweek": [141], "alpha": [0], "actual_total_points": [12],
                  "actual_net_points": [12], "scoring_protocol": ["autosubs_v2"]}).to_csv(reference, index=False)
    return state, predictions, reference


def test_saved_replay_reconciles_and_preserves_exact_captured_bytes(tmp_path):
    state, predictions, reference = _inputs(tmp_path)
    output = tmp_path / "replayed"
    assert replay.run_replay(state, predictions, output, reference_path=reference) == output
    receipt = replay.read_replay(output)
    assert receipt["reference_reconciliation"] == "PASS"
    assert receipt["gross_points"] == receipt["net_points"] == 12
    assert receipt["independent_financial_rule_legality"] == receipt["native_optimality"] == "NOT ASSESSED"
    assert receipt["promotion_eligible"] is False
    assert (output / "decisions.json").read_bytes() == state.read_bytes()
    assert receipt["input_sha256"]["predictions.csv"] == sha256_file(predictions)
    with pytest.raises(FileExistsError):
        replay.run_replay(state, predictions, output)


@pytest.mark.parametrize("change", ["extra", "changed", "missing", "failed"])
def test_corrupted_saved_replay_cannot_be_reported(tmp_path, change):
    state, predictions, _ = _inputs(tmp_path)
    output = replay.run_replay(state, predictions, tmp_path / "replayed")
    if change == "extra": (output / "extra.txt").write_text("extra")
    elif change == "changed": (output / "replayed.csv").write_text("changed")
    elif change == "missing": (output / "predictions.csv").unlink()
    else: (output / "failed.json").write_text('{}')
    with pytest.raises(ValueError):
        replay.read_replay(output)


@pytest.mark.parametrize("failure", ["missing_outcomes", "reference_points", "reference_protocol"])
def test_failure_preserves_inputs_without_completed_qualification(tmp_path, failure):
    state, predictions, reference = _inputs(tmp_path)
    if failure == "missing_outcomes": outcomes().iloc[:-1].to_csv(predictions, index=False)
    elif failure == "reference_points":
        frame = pd.read_csv(reference).assign(actual_total_points=13, actual_net_points=13)
        frame.to_csv(reference, index=False)
    else:
        frame = pd.read_csv(reference).assign(scoring_protocol="legacy_no_autosubs")
        frame.to_csv(reference, index=False)
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        replay.run_replay(state, predictions, output, reference_path=reference)
    assert (output / "failed.json").exists() and not (output / "complete.json").exists()
    assert (output / "decisions.json").read_bytes() == state.read_bytes()


def blank_row(player=12):
    return pd.DataFrame({"player_id": [player], "GW": [141], "position": ["MID"],
        "team": ["club1"], "actual_total_points": [0.], "minutes": [0.],
        "outcome_basis": ["EXPLICIT_RETROSPECTIVE_BLANK"]})


def test_separate_blank_outcomes_have_no_fake_fixture_or_forecast():
    original = outcomes().loc[lambda frame: frame.player_id != 12]
    before = original.copy(deep=True)
    result = replay.replay_snapshot(snapshot(), original, blank_outcomes=blank_row())
    assert result.actual_total_points.tolist() == [12]
    assert result.explicit_blank_players.tolist() == [1]
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize("failure", ["nonzero", "duplicate", "observed", "wrong_basis", "boolean"])
def test_blank_arithmetic_rejects_invalid_or_overlapping_outcomes(failure):
    original = outcomes().loc[lambda frame: frame.player_id != 12]
    blanks = blank_row()
    if failure == "nonzero": blanks["actual_total_points"] = 1.
    elif failure == "duplicate": blanks = pd.concat([blanks, blanks])
    elif failure == "observed": blanks = blank_row(8)
    elif failure == "wrong_basis": blanks["outcome_basis"] = "guessed"
    else: blanks["minutes"] = True
    with pytest.raises(ValueError):
        solio.score_decisions(pd.DataFrame(snapshot()["decisions"]), original,
                              require_complete_outcomes=True, explicit_blank_outcomes=blanks)


def capture_inputs():
    import gzip
    import hashlib
    register = {"events": [{"id": 27, "deadline_time": "2024-03-02T13:30:00Z"}]}
    payload = json.dumps(register).encode()
    manifest = {"captured_at": "2024-03-02T04:00:00Z", "event": 27,
        "event_deadline": "2024-03-02T13:30:00Z", "backfill": {"season": "2023-24"},
        "files": [{"name": "bootstrap-static.json.gz", "endpoint": "/bootstrap-static/",
                   "sha256": hashlib.sha256(payload).hexdigest()}]}
    inputs = {"fixtures.csv": b"id\n1\n", "teams.csv": b"id\n1\n",
              "bootstrap-static.json.gz": gzip.compress(payload), "manifest.json": json.dumps(manifest).encode()}
    evidence = {"schema_version": 1, "season": "2023-24", "first_global_gw": 115, "event": 27,
                "fixture_source_commit": "a" * 40,
                "input_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()}}
    inputs["blank-evidence.json"] = json.dumps(evidence).encode()
    return inputs


@pytest.mark.parametrize("failure", ["changed_asset", "changed_payload_manifest", "late", "old", "naive", "wrong_event", "wrong_season", "unpinned"])
def test_capture_evidence_rejects_changed_hashes_and_invalid_timing(failure):
    import hashlib
    inputs = capture_inputs()
    evidence = json.loads(inputs["blank-evidence.json"])
    if failure == "changed_asset": inputs["fixtures.csv"] += b"2\n"
    elif failure == "unpinned": evidence["fixture_source_commit"] = "main"
    else:
        manifest = json.loads(inputs["manifest.json"])
        if failure == "changed_payload_manifest": manifest["files"][0]["sha256"] = "0" * 64
        elif failure == "late": manifest["captured_at"] = "2024-03-02T14:00:00Z"
        elif failure == "old": manifest["captured_at"] = "2024-02-29T04:00:00Z"
        elif failure == "naive": manifest["captured_at"] = "2024-03-02T04:00:00"
        elif failure == "wrong_event": manifest["event"] = 28
        else: manifest["backfill"]["season"] = "2024-25"
        inputs["manifest.json"] = json.dumps(manifest).encode()
        evidence["input_sha256"]["manifest.json"] = hashlib.sha256(inputs["manifest.json"]).hexdigest()
    inputs["blank-evidence.json"] = json.dumps(evidence).encode()
    with pytest.raises(ValueError):
        replay._admit_blank_evidence(snapshot(), outcomes(), inputs)


def test_saved_blank_capture_preserves_original_bytes_and_audit(tmp_path, monkeypatch):
    from fpl.milp import blank_outcomes
    state, predictions, reference = _inputs(tmp_path)
    outcomes().loc[lambda frame: frame.player_id != 12].to_csv(predictions, index=False)
    # Isolate capture/receipt integration; full fixture-ledger admission is tested
    # independently with complete seasons in test_blank_outcomes.py.
    monkeypatch.setattr(blank_outcomes, "explicit_blank_outcomes",
                        lambda state, frame, **kwargs: (frame.copy(), blank_row()))
    for name, content in capture_inputs().items():
        (tmp_path / name).write_bytes(content)
    target = replay.run_replay(state, predictions, tmp_path / "blank-replay", reference_path=reference,
                               blank_evidence_path=tmp_path / "blank-evidence.json")
    receipt = replay.read_replay(target)
    assert receipt["explicit_blank_outcomes"] == 1
    assert receipt["registration_identity"] == "CAPTURE_TIME_ONLY"
    assert receipt["reference_reconciliation"] == "PASS"
    assert (target / "predictions.csv").read_bytes() == predictions.read_bytes()
    assert pd.read_csv(target / "explicit-blank-outcomes.csv").outcome_basis.tolist() == ["EXPLICIT_RETROSPECTIVE_BLANK"]
    assert receipt["independent_financial_rule_legality"] == "NOT ASSESSED"
    (target / "fixtures.csv").write_bytes(b"changed")
    with pytest.raises(ValueError): replay.read_replay(target)
