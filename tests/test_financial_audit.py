"""Hand-calculated independent finance checks; no fitting, scoring or solving."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from fpl.milp.financial_audit import audit_legacy_finances


def _example():
    original = list(range(1, 16))
    swapped = list(range(2, 17))
    decisions = [
        {"gameweek": 153, "squad": swapped, "transfers_in": [16], "transfers_out": [1],
         "alpha": 1, "q_start": 0, "q_end": 1, "budget_start": 10, "budget_end": 14, "chip_played": None},
        {"gameweek": 154, "squad": original, "transfers_in": [1], "transfers_out": [16],
         "alpha": 0, "q_start": 1, "q_end": 1, "budget_start": 14, "budget_end": 4, "chip_played": None},
        {"gameweek": 155, "squad": swapped, "transfers_in": [16], "transfers_out": [1],
         "alpha": 0, "q_start": 1, "q_end": 1, "budget_start": 4, "budget_end": 18, "chip_played": None},
    ]
    prices = pd.DataFrame([{"player_id": player, "GW": week,
        "value": ({153: 54, 154: 60, 155: 64}[week] if player == 1 else 50)}
        for week in (153, 154, 155) for player in range(1, 17)])
    state = {"mode": "continuing", "squad": original, "budget": 10, "free_transfers": 0,
             "purchase_prices": {str(player): 50 for player in original}}
    return {"schema_version": 1, "settlement_completed": False, "decisions": decisions}, prices, state


def _audit(example, **kwargs):
    snapshot, prices, state = example
    return audit_legacy_finances(snapshot, prices, initial_state=state, ft_cap=2, ft_gain=1, **kwargs)


def test_current_price_sales_and_rebuy_follow_exact_two_round_arithmetic_without_mutation():
    example = _example()
    before = deepcopy(example)
    result = _audit(example)
    assert result.sales_proceeds.tolist() == [54, 50, 64]
    assert result.purchases_cost.tolist() == [50, 60, 50]
    assert result.derived_bank_start.tolist() == [10, 14, 4]
    assert result.derived_bank_end.tolist() == [14, 4, 18]
    assert result.bank_end_difference.tolist() == [0, 0, 0]
    assert result.alpha.tolist() == [1, 0, 0]
    assert result.q_start.tolist() == [0, 1, 1]
    assert result.q_end.tolist() == [1, 1, 1]
    assert result.arithmetic_assessment.eq("PASS").all()
    assert result.scope.eq("LEGACY_NO_CHIP_FINANCIAL_ARITHMETIC").all()
    assert example[0] == before[0] and example[2] == before[2]
    pd.testing.assert_frame_equal(example[1], before[1])


def test_half_profit_50_to_54_sells_at52_and_rebuy_resets_basis_to60():
    result = _audit(_example(), selling_policy="half_profit", compare_reported=False)
    assert result.sales_proceeds.tolist() == [52, 50, 62]
    assert result.derived_bank_end.tolist() == [12, 2, 14]
    assert result.reported_budget_end.tolist() == [14, 4, 18]
    assert result.bank_end_difference.tolist() == [-2, -2, -4]
    assert result.scope.eq("FIXED_DECISION_SELLING_POLICY_SENSITIVITY").all()
    assert result.permanent_squad_start.iloc[2] == list(range(1, 16))


@pytest.mark.parametrize("policy", ["current_price", "half_profit"])
def test_price_loss_50_to47_sells_at47_for_both_policies(policy):
    snapshot, prices, state = _example()
    snapshot["decisions"] = snapshot["decisions"][:1]
    snapshot["decisions"][0]["budget_end"] = 7
    prices.loc[(prices.player_id == 1) & (prices.GW == 153), "value"] = 47
    result = _audit((snapshot, prices, state), selling_policy=policy)
    assert result.sales_proceeds.tolist() == [47]
    assert result.derived_bank_end.tolist() == [7]


def test_fresh_build_then_continuing_transfer_in_integer_tenths():
    squad = list(range(1, 16))
    snapshot = {"schema_version": 1, "settlement_completed": False, "decisions": [
        {"gameweek": 153, "squad": squad, "transfers_in": [], "transfers_out": [], "alpha": 0,
         "q_start": 0, "q_end": 1, "budget_start": 1000, "budget_end": 250, "chip_played": None,
         "state_mode": "fresh_build", "permanent_squad_start": [], "permanent_squad_end": squad,
         "permanent_bank_start": 1000, "permanent_bank_end": 250, "temporary_selection_bank_end": None},
        {"gameweek": 154, "squad": list(range(2, 17)), "transfers_in": [16], "transfers_out": [1], "alpha": 0,
         "q_start": 1, "q_end": 1, "budget_start": 250, "budget_end": 254, "chip_played": None}]}
    prices = pd.DataFrame([{"player_id": player, "GW": week, "value": 54 if (player, week) == (1, 154) else 50}
                           for week in (153, 154) for player in range(1, 17)])
    result = audit_legacy_finances(snapshot, prices, initial_state={"mode": "fresh", "squad": [], "budget": 1000., "free_transfers": 0}, ft_cap=2, ft_gain=1)
    assert result["mode"].tolist() == ["fresh", "continuing"]
    assert result.purchases_cost.tolist() == [750, 50]
    assert result.sales_proceeds.tolist() == [0, 54]
    assert result.derived_bank_end.tolist() == [250, 254]
    assert result.permanent_squad_start.iloc[0] == []


@pytest.mark.parametrize("field,value", [("budget_start", 11), ("budget_end", 13), ("q_start", 1),
                                        ("q_end", 2), ("alpha", 0), ("alpha", 1.5),
                                        ("transfers_in", []), ("transfers_out", []),
                                        ("transfers_in", [16, 16]), ("transfers_out", [1, 16])])
def test_wrong_reported_finance_or_transfer_state_rejected(field, value):
    example = _example()
    example[0]["decisions"][0][field] = value
    with pytest.raises(ValueError):
        _audit(example)


@pytest.mark.parametrize("field,value", [("state_mode", "fresh_build"), ("permanent_squad_start", list(range(2, 17))),
                                        ("permanent_squad_end", list(range(1, 16))),
                                        ("permanent_bank_start", 9), ("permanent_bank_end", 15)])
def test_optional_exported_state_must_match_reconstruction(field, value):
    example = _example()
    example[0]["decisions"][0][field] = value
    with pytest.raises(ValueError, match="optional"):
        _audit(example)


def test_sensitivity_skips_bank_equality_but_retains_holdings_checks():
    example = _example()
    example[0]["decisions"][0]["permanent_bank_end"] = 999
    assert _audit(example, compare_reported=False).bank_end_difference.eq(0).all()
    example[0]["decisions"][0]["permanent_squad_end"] = list(range(1, 16))
    with pytest.raises(ValueError, match="optional"):
        _audit(example, compare_reported=False)


@pytest.mark.parametrize("field", ["alpha", "q_start", "q_end"])
def test_sensitivity_does_not_disable_hit_or_ft_validation(field):
    example = _example()
    example[0]["decisions"][0][field] += 1
    with pytest.raises(ValueError):
        _audit(example, selling_policy="half_profit", compare_reported=False)


@pytest.mark.parametrize("change", ["missing_selected", "missing_sold", "duplicate"])
def test_prices_must_cover_selected_and_sold_ids_exactly(change):
    snapshot, prices, state = _example()
    if change == "duplicate":
        prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    else:
        player = 2 if change == "missing_selected" else 1
        prices = prices.loc[~((prices.GW == 153) & (prices.player_id == player))]
    with pytest.raises(ValueError, match="prices"):
        _audit((snapshot, prices, state))


@pytest.mark.parametrize("value", [True, "50", np.nan, np.inf, 0, -1, 50.5])
def test_price_tenths_require_positive_finite_nonboolean_integer_values(value):
    snapshot, prices, state = _example()
    prices["value"] = prices.value.astype(object)
    prices.loc[0, "value"] = value
    with pytest.raises(ValueError):
        _audit((snapshot, prices, state))


@pytest.mark.parametrize("chip", ["WC", "FH", "BB", "TC", "TC_name"])
def test_chip_decisions_are_unsupported_instead_of_certified(chip):
    example = _example()
    example[0]["decisions"][0]["chip_played"] = chip
    with pytest.raises(ValueError, match="no-chip"):
        _audit(example)


def test_negative_reconstructed_funds_fail_even_in_sensitivity_mode():
    snapshot, prices, state = _example()
    prices.loc[(prices.player_id == 16) & (prices.GW == 153), "value"] = 70
    with pytest.raises(ValueError, match="unaffordable"):
        _audit((snapshot, prices, state), compare_reported=False)


@pytest.mark.parametrize("change", ["missing", "partial", "unknown", "negative"])
def test_half_profit_requires_complete_initial_purchase_basis(change):
    example = _example()
    if change == "missing":
        example[2].pop("purchase_prices")
    elif change == "partial":
        example[2]["purchase_prices"].pop("1")
    elif change == "unknown":
        example[2]["purchase_prices"]["99"] = 50
    else:
        example[2]["purchase_prices"]["1"] = -1
    with pytest.raises(ValueError, match="purchase"):
        _audit(example, selling_policy="half_profit", compare_reported=False)


def test_current_price_does_not_guess_initial_purchase_basis():
    example = _example()
    example[2].pop("purchase_prices")
    assert _audit(example).derived_bank_end.tolist() == [14, 4, 18]


@pytest.mark.parametrize("option,value", [("ft_cap", 0), ("ft_cap", True), ("ft_gain", 0), ("ft_gain", 1.5)])
def test_ft_parameters_are_explicit_positive_integers(option, value):
    snapshot, prices, state = _example()
    values = {"ft_cap": 2, "ft_gain": 1, option: value}
    with pytest.raises(ValueError):
        audit_legacy_finances(snapshot, prices, initial_state=state, **values)


def test_lifted_initial_free_transfer_cap_cannot_be_silently_lowered():
    example = _example()
    example[2]["free_transfers"] = 5
    with pytest.raises(ValueError, match="honor"):
        _audit(example)


@pytest.mark.parametrize("change", ["gap", "duplicate", "outcomes", "settled"])
def test_snapshot_requires_contiguous_pre_settlement_decisions(change):
    example = _example()
    if change == "gap":
        example[0]["decisions"][1]["gameweek"] = 157
    elif change == "duplicate":
        example[0]["decisions"][1]["gameweek"] = 153
    elif change == "outcomes":
        example[0]["decisions"][0]["actual_total_points"] = 80
    else:
        example[0]["settlement_completed"] = True
    with pytest.raises(ValueError):
        _audit(example)
