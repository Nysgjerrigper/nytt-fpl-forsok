"""End-to-end admission tests for independently constructed comparison contracts."""

import hashlib
import json

import pandas as pd
import pytest

from fpl.milp.compare_backtests import compare


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair(tmp_path, intervention="forecast"):
    """Build explicit manifests without depending on implementation schema constants."""
    contracts, backtests = [], []
    for label in ("a", "b"):
        csv = tmp_path / f"{label}.csv"
        pd.DataFrame({
            "gameweek": [153, 154, 155], "actual_total_points": [50, 55, 60],
            "alpha": [0, 1, 0], "actual_net_points": [50, 51, 60],
            "scoring_protocol": ["autosubs_v2"] * 3,
        }).to_csv(csv, index=False)
        artifacts = {}
        for name in ("dataset", "outcomes", "player_pool", "predictions", "parameters",
                     "features", "model_source", "optimizer_source", "scorer_source"):
            artifact = tmp_path / f"{label}-{name}.txt"
            artifact.write_text(f"frozen {name}\n")
            artifacts[name] = {"path": artifact.name, "sha256": _hash(artifact)}
        contract = {
            "schema_version": 1, "purpose": "framework_validation", "run_id": label,
            "intervention": intervention, "backtest_sha256": _hash(csv),
            "metric": "actual_net_points",
            "context": {
                "season": "2024-25", "start_season": "2020-21", "gameweeks": [153, 154, 155],
                "information_protocol": "origin", "horizon": 3, "scoring_protocol": "autosubs_v2",
                "price_policy": "static", "sell_price_policy": "buy_price",
                "initial_state": {"mode": "fresh", "squad": [], "budget": 1000, "free_transfers": 1},
                "transfer_hit_points": 4,
                "rule_set": "legacy", "chip_inventory": {"enabled": []},
                "runtime": {"solver": "highs", "gap_rel": 0},
            },
            "artifacts": artifacts,
        }
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(contract))
        contracts.append(path)
        backtests.append(csv)
    return backtests, contracts


def _edit(path, change):
    contract = json.loads(path.read_text())
    change(contract)
    path.write_text(json.dumps(contract))


def _compare(backtests, contracts):
    return compare(*backtests, n_boot=20, points_col="actual_net_points",
                   contract_a=contracts[0], contract_b=contracts[1])


def test_matching_contracts_validate_comparability_without_claiming_promotion(tmp_path):
    backtests, contracts = _pair(tmp_path)
    result = _compare(backtests, contracts)
    assert result["evidence_status"] == "CONTRACT_VALIDATED"
    assert result["contract"]["promotion_assessment"] == "NOT ASSESSED"
    assert result["total_diff"] == 0
    assert result["contract"]["a"]["contract_sha256"] == _hash(contracts[0])


def test_changed_backtest_bytes_invalidate_contract(tmp_path):
    backtests, contracts = _pair(tmp_path)
    with backtests[0].open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        _compare(backtests, contracts)


def test_changed_referenced_artifact_bytes_invalidate_contract(tmp_path):
    backtests, contracts = _pair(tmp_path)
    (tmp_path / "a-dataset.txt").write_text("changed data")
    with pytest.raises(ValueError, match="hash"):
        _compare(backtests, contracts)


@pytest.mark.parametrize("name,value", [
    ("horizon", 2), ("runtime", {"solver": "cbc", "gap_rel": 0}),
    ("initial_state", {"mode": "fresh", "squad": [], "budget": 999, "free_transfers": 1}),
    ("scoring_protocol", "legacy_no_autosubs"),
])
def test_mismatched_context_rejected(tmp_path, name, value):
    backtests, contracts = _pair(tmp_path)
    _edit(contracts[0], lambda spec: spec["context"].__setitem__(name, value))
    with pytest.raises(ValueError):
        _compare(backtests, contracts)


def test_mismatched_declared_purpose_rejected(tmp_path):
    backtests, contracts = _pair(tmp_path)
    _edit(contracts[0], lambda spec: spec.__setitem__("purpose", "registered_research"))
    with pytest.raises(ValueError, match="purpose"):
        _compare(backtests, contracts)


def _change_predictions(tmp_path, contract):
    artifact = tmp_path / "a-predictions.txt"
    artifact.write_text("new challenger predictions")
    _edit(contract, lambda spec: spec["artifacts"]["predictions"].__setitem__("sha256", _hash(artifact)))


def test_optimizer_intervention_requires_identical_predictions(tmp_path):
    backtests, contracts = _pair(tmp_path, intervention="optimizer")
    _change_predictions(tmp_path, contracts[0])
    with pytest.raises(ValueError, match="predictions"):
        _compare(backtests, contracts)


def test_forecast_intervention_allows_changed_predictions(tmp_path):
    backtests, contracts = _pair(tmp_path, intervention="forecast")
    _change_predictions(tmp_path, contracts[0])
    assert _compare(backtests, contracts)["evidence_status"] == "CONTRACT_VALIDATED"


@pytest.mark.parametrize("field", ["schema_version", "purpose", "run_id", "intervention",
                                    "backtest_sha256", "metric", "context", "artifacts"])
def test_required_top_level_fields_cannot_be_omitted(tmp_path, field):
    backtests, contracts = _pair(tmp_path)
    _edit(contracts[0], lambda spec: spec.pop(field))
    with pytest.raises(ValueError):
        _compare(backtests, contracts)


@pytest.mark.parametrize("field", ["season", "start_season", "gameweeks", "information_protocol", "horizon",
                                    "scoring_protocol", "price_policy", "sell_price_policy", "initial_state",
                                    "rule_set", "chip_inventory", "runtime", "transfer_hit_points"])
def test_required_context_fields_cannot_be_omitted(tmp_path, field):
    backtests, contracts = _pair(tmp_path)
    _edit(contracts[0], lambda spec: spec["context"].pop(field))
    with pytest.raises(ValueError):
        _compare(backtests, contracts)


@pytest.mark.parametrize("field", ["dataset", "outcomes", "player_pool", "predictions", "parameters",
                                    "features", "model_source", "optimizer_source", "scorer_source"])
def test_required_artifact_bindings_cannot_be_omitted(tmp_path, field):
    backtests, contracts = _pair(tmp_path)
    _edit(contracts[0], lambda spec: spec["artifacts"].pop(field))
    with pytest.raises(ValueError):
        _compare(backtests, contracts)


def test_one_contract_cannot_establish_pair_comparability(tmp_path):
    backtests, contracts = _pair(tmp_path)
    with pytest.raises(ValueError, match="both"):
        compare(*backtests, contract_a=contracts[0])


@pytest.mark.parametrize("field", ["mode", "squad", "budget", "free_transfers"])
def test_matching_initial_states_cannot_omit_required_fields(tmp_path, field):
    backtests, contracts = _pair(tmp_path)
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"]["initial_state"].pop(field))
    with pytest.raises(ValueError, match="initial_state"):
        _compare(backtests, contracts)


@pytest.mark.parametrize("field,value", [
    ("mode", "unknown"), ("mode", None), ("mode", []),
    ("squad", None), ("squad", {}), ("squad", [True]), ("squad", [1.0]),
    ("squad", ["1"]), ("squad", [0]), ("squad", [-1]), ("squad", [1, 1]),
    ("squad", [1]), ("squad", list(range(1, 16))),
    ("budget", True), ("budget", False), ("budget", -1), ("budget", float("nan")),
    ("budget", float("inf")), ("budget", "1000"), ("budget", None),
    ("free_transfers", True), ("free_transfers", 1.0), ("free_transfers", -1),
    ("free_transfers", "1"), ("free_transfers", None),
])
def test_matching_malformed_initial_states_are_rejected(tmp_path, field, value):
    backtests, contracts = _pair(tmp_path)
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"]["initial_state"].__setitem__(field, value))
    with pytest.raises(ValueError):
        _compare(backtests, contracts)


@pytest.mark.parametrize("squad", [[], list(range(1, 15)), list(range(1, 17)), [1] * 15])
def test_continuing_state_requires_fifteen_distinct_ids(tmp_path, squad):
    backtests, contracts = _pair(tmp_path)
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].__setitem__("initial_state", {
            "mode": "continuing", "squad": squad, "budget": 25, "free_transfers": 2}))
    with pytest.raises(ValueError, match="initial_state"):
        _compare(backtests, contracts)


def test_complete_continuing_state_preserves_explicit_extras_without_rule_claims(tmp_path):
    backtests, contracts = _pair(tmp_path)
    state = {"mode": "continuing", "squad": list(range(1, 16)), "budget": 123.45,
             "free_transfers": 8, "sell_prices": {"1": 50}, "owner_verified": False}
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].__setitem__("initial_state", state))
    result = _compare(backtests, contracts)
    assert result["evidence_status"] == "CONTRACT_VALIDATED"
    assert result["contract"]["a"]["context"]["initial_state"] == state
    assert result["contract"]["promotion_assessment"] == "NOT ASSESSED"


def test_complete_fresh_state_allows_explicit_nonstandard_budget_and_ft(tmp_path):
    backtests, contracts = _pair(tmp_path)
    state = {"mode": "fresh", "squad": [], "budget": 999.5, "free_transfers": 0}
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].__setitem__("initial_state", state))
    assert _compare(backtests, contracts)["contract"]["b"]["context"]["initial_state"] == state


def test_extra_initial_state_fields_remain_comparison_assumptions(tmp_path):
    backtests, contracts = _pair(tmp_path)
    for index, contract in enumerate(contracts):
        _edit(contract, lambda spec: spec["context"]["initial_state"].__setitem__("owner_verified", index == 0))
    with pytest.raises(ValueError, match="context differs"):
        _compare(backtests, contracts)


@pytest.mark.parametrize("cost", [0, -4, 8, 4.0, True, False, "4", None])
def test_transfer_hit_cost_must_be_the_supported_literal_integer(tmp_path, cost):
    backtests, contracts = _pair(tmp_path)
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].__setitem__("transfer_hit_points", cost))
    with pytest.raises(ValueError, match="transfer_hit_points"):
        _compare(backtests, contracts)


def test_missing_transfer_hit_cost_rejects_matching_old_contracts_without_rewriting(tmp_path):
    backtests, contracts = _pair(tmp_path)
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].pop("transfer_hit_points"))
    original = [contract.read_bytes() for contract in contracts]
    with pytest.raises(ValueError):
        _compare(backtests, contracts)
    assert [contract.read_bytes() for contract in contracts] == original


@pytest.mark.parametrize("state", [None, [], {}, {"arbitrary": "state"}])
def test_matching_nonstates_cannot_establish_comparability(tmp_path, state):
    backtests, contracts = _pair(tmp_path)
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].__setitem__("initial_state", state))
    with pytest.raises(ValueError, match="initial_state"):
        _compare(backtests, contracts)


@pytest.mark.parametrize("invalid_id", [True, 1.0, "1", 0, -1])
def test_complete_sized_continuing_squad_requires_strict_positive_integer_ids(tmp_path, invalid_id):
    backtests, contracts = _pair(tmp_path)
    state = {"mode": "continuing", "squad": [invalid_id, *range(2, 16)], "budget": 0, "free_transfers": 0}
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].__setitem__("initial_state", state))
    with pytest.raises(ValueError, match="initial_state"):
        _compare(backtests, contracts)


@pytest.mark.parametrize("season", ["2024-99", "2024"])
def test_matching_malformed_season_labels_do_not_establish_comparability(tmp_path, season):
    backtests, contracts = _pair(tmp_path)
    for contract in contracts:
        _edit(contract, lambda spec: spec["context"].__setitem__("season", season))
    with pytest.raises(ValueError, match="consecutive YYYY-YY"):
        _compare(backtests, contracts)
