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
                "initial_state": {"mode": "fresh", "budget": 1000},
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
    ("initial_state", {"mode": "fresh", "budget": 999}),
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
                                    "rule_set", "chip_inventory", "runtime"])
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
