"""Independent validation of the external optimizer's decision boundary."""

from copy import deepcopy

import pandas as pd
import pytest

from fpl.milp.solio_output import normalize_solution


def inputs():
    ids = list(range(1, 17))
    meta = pd.DataFrame({"element_type": [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3 + [2],
                         "name": [str((p - 1) % 5) for p in range(1, 16)] + ["2"],
                         "1_Pts": ids, "2_Pts": ids, "3_Pts": ids}, index=ids)
    return {"merged_data": meta, "element_to_stable": {p: p + 1000 for p in ids},
            "buy_price": {p: 5.0 for p in ids}, "sell_price": {p: 4.8 for p in range(1, 16)},
            "initial_squad": list(range(1, 16)), "itb": 2.0, "ft": 2,
            "next_gw": 1, "season_offset": 228}


def week_rows(week=1, squad=None, chip="", ins=(), outs=()):
    squad = set(range(1, 16)) if squad is None else set(squad)
    lineup = squad - {2, 6, 7, 12}
    bench = [2, 6, 7, 12]
    return [{"id": p, "week": week, "squad": int(p in squad),
             "lineup": int(p in squad if chip == "BB" else p in lineup),
             "bench": -1 if chip == "BB" else (bench.index(p) if p in bench else -1),
             "captain": int(p == 15), "vicecaptain": int(p == 14),
             "transfer_in": int(p in ins), "transfer_out": int(p in outs),
             "chip": chip if chip != "TC" or p == 15 else ""}
            for p in sorted(squad | set(outs))]


def solution(rows=None, ft=2, chip="", bank=2.0, nt=0, pt=0):
    return {"picks": pd.DataFrame(week_rows(chip=chip) if rows is None else rows),
            "statistics": {1: {"itb": bank, "ft": ft, "nt": nt, "pt": pt, "chip": chip or None}},
            "score": 123.0}


def test_stable_ids_bank_and_modern_rollover():
    result = normalize_solution(solution(), inputs())[0]
    assert result["gameweek"] == 229
    assert result["captain"] == [1015]
    assert result["budget_start"] == result["budget_end"] == 20
    assert result["q_end"] == 3
    assert result["bench_order"] == [1002, 1006, 1007, 1012]


@pytest.mark.parametrize("chip", ["BB", "TC", "WC"])
def test_chip_layout_and_bank(chip):
    output = normalize_solution(solution(chip=chip), inputs())[0]
    assert len(output["lineup"]) == 11
    assert len(output["bench_order"]) == 4
    assert output["captain"][0] in output["lineup"]
    assert output["vice_captain"][0] in output["lineup"]
    assert output["chip_played"] == ("TC_15" if chip == "TC" else chip)
    assert output["q_end"] == (2 if chip == "WC" else 3)


def test_patched_bench_boost_retains_solver_xi_and_bench():
    rows = week_rows()
    for row in rows:
        row["chip"] = "BB"
    data = solution(rows, chip="BB")
    output = normalize_solution(data, inputs())[0]
    assert output["chip_played"] == "BB"
    assert output["lineup"] == [p + 1000 for p in range(1, 16) if p not in {2, 6, 7, 12}]
    assert output["bench_order"] == [1002, 1006, 1007, 1012]


def test_legacy_bench_boost_rejects_two_goalkeeper_captains():
    data = solution(chip="BB")
    data["picks"]["captain"] = (data["picks"].id == 1).astype(int)
    data["picks"]["vicecaptain"] = (data["picks"].id == 2).astype(int)
    with pytest.raises(ValueError, match="No legal Bench Boost XI"):
        normalize_solution(data, inputs())


def test_sale_appreciation_is_used_only_once():
    prepared = inputs()
    rows = week_rows(1, set(range(1, 16)) - {3} | {16}, ins={16}, outs={3})
    rows += week_rows(2, ins={3}, outs={16})
    rows += week_rows(3, set(range(1, 16)) - {3} | {16}, ins={16}, outs={3})
    data = solution(rows, bank=1.8, nt=1)
    data["statistics"][2] = {"itb": 1.8, "ft": 2, "nt": 1, "pt": 0, "chip": None}
    data["statistics"][3] = {"itb": 1.8, "ft": 2, "nt": 1, "pt": 0, "chip": None}
    output = normalize_solution(data, prepared)
    assert [r["budget_end"] for r in output] == [18, 18, 18]
    assert all(r["q_end"] == 2 for r in output)


def test_free_hit_preserves_squad_bank_and_free_transfers():
    prepared = inputs()
    temporary = set(range(1, 16)) - {3} | {16}
    rows = week_rows(1, temporary, chip="FH") + week_rows(2)
    data = solution(rows, chip="FH")
    data["statistics"][2] = {"itb": 2.0, "ft": 2, "nt": 0, "pt": 0, "chip": None}
    output = normalize_solution(data, prepared)
    assert 1016 in output[0]["squad"] and 1016 not in output[1]["squad"]
    assert output[0]["budget_end"] == output[1]["budget_start"] == 20
    assert output[0]["transfers_in"] == []
    assert output[0]["q_end"] == 2


def test_fresh_team_build_has_no_hits():
    prepared = inputs()
    prepared.update(initial_squad=[], itb=100.0, ft=0)
    data = solution(week_rows(ins=set(range(1, 16))), ft=0, bank=25.0)
    output = normalize_solution(data, prepared)[0]
    assert output["alpha"] == 0
    assert len(output["transfers_in"]) == 15
    assert output["budget_start"] == 1000
    assert output["budget_end"] == 250
    assert output["q_end"] == 1


@pytest.mark.parametrize("corruption", ["duplicate", "fractional", "missing_identity", "club",
                                       "bench", "captain", "bank", "ft", "nt", "pt", "chip", "state"])
def test_corrupt_solutions_fail_closed(corruption):
    prepared = inputs()
    data = solution()
    if corruption == "duplicate":
        data["picks"] = pd.concat([data["picks"], data["picks"].iloc[[0]]])
    elif corruption == "fractional":
        data["picks"]["lineup"] = data["picks"]["lineup"].astype(float)
        data["picks"].loc[0, "lineup"] = 0.5
    elif corruption == "missing_identity":
        del prepared["element_to_stable"][1]
    elif corruption == "club":
        prepared["merged_data"]["name"] = "one club"
    elif corruption == "bench":
        data["picks"].loc[1, "bench"] = 1
    elif corruption == "captain":
        data["picks"].loc[0, "captain"] = 1
    elif corruption in {"bank", "ft", "nt", "pt"}:
        field = "itb" if corruption == "bank" else corruption
        data["statistics"][1][field] = 3
    elif corruption == "chip":
        data["statistics"][1]["chip"] = "FH"
    elif corruption == "state":
        data["picks"].loc[2, "transfer_out"] = 1
    with pytest.raises(ValueError):
        normalize_solution(data, prepared)


def test_free_hit_cannot_make_permanent_transfers():
    data = solution(week_rows(chip="FH", outs={3}), chip="FH", nt=1)
    with pytest.raises(ValueError, match="Free Hit"):
        normalize_solution(data, inputs())


def test_free_hit_cannot_overspend():
    prepared = inputs()
    prepared["buy_price"][16] = 20.0
    data = solution(week_rows(squad=set(range(1, 16)) - {3} | {16}, chip="FH"), chip="FH")
    with pytest.raises(ValueError, match="wealth"):
        normalize_solution(data, prepared)


def test_inputs_are_not_mutated():
    prepared, data = inputs(), solution(chip="BB")
    before = deepcopy(data)
    normalize_solution(data, prepared)
    pd.testing.assert_frame_equal(data["picks"], before["picks"])


@pytest.mark.parametrize("ft,ins,expected", [(5, (), 5), (2, (16,), 2), (0, (16,), 1)])
def test_free_transfer_spending_and_cap(ft, ins, expected):
    prepared = inputs()
    prepared["ft"] = ft
    if ins:
        data = solution(week_rows(squad=set(range(1, 16)) - {3} | {16}, ins=ins, outs={3}),
                        ft=ft, bank=1.8, nt=1, pt=max(0, 1 - ft))
    else:
        data = solution(ft=ft)
    assert normalize_solution(data, prepared)[0]["q_end"] == expected


def test_fh_retained_originals_do_not_require_rebuying_at_market_price():
    prepared = inputs()
    prepared["itb"] = 0.0
    output = normalize_solution(solution(chip="FH", bank=0.0), prepared)[0]
    assert output["budget_end"] == 0.0


def test_fh_after_resale_uses_market_price_for_rebought_original():
    prepared = inputs()
    prepared["buy_price"][3] = 10.0
    prepared["sell_price"][3] = 5.0
    prepared["buy_price"][16] = 10.1
    prepared["itb"] = 5.15
    rows = week_rows(1, set(range(1, 16)) - {3} | {16}, ins={16}, outs={3})
    rows += week_rows(2, ins={3}, outs={16})
    rows += week_rows(3, set(range(1, 16)) - {3} | {16}, chip="FH")
    data = solution(rows, bank=0.05, nt=1)
    data["statistics"][2] = {"itb": 0.15, "ft": 2, "nt": 1, "pt": 0, "chip": None}
    data["statistics"][3] = {"itb": 0.15, "ft": 2, "nt": 0, "pt": 0, "chip": "FH"}
    output = normalize_solution(data, prepared)
    assert output[2]["budget_end"] == 1.5


@pytest.mark.parametrize("missing", ["name", "buy_price", "1_Pts"])
def test_missing_price_or_metadata_fails_clearly(missing):
    prepared = inputs()
    if missing == "buy_price":
        del prepared["buy_price"][1]
    else:
        prepared["merged_data"] = prepared["merged_data"].drop(columns=missing)
    with pytest.raises(ValueError):
        normalize_solution(solution(chip="BB"), prepared)


def test_fresh_build_exports_empty_start_then_continuing_permanent_state():
    prepared = inputs()
    prepared.update(initial_squad=[], itb=100.0, ft=0)
    rows = week_rows(1, ins=set(range(1, 16))) + week_rows(2)
    data = solution(rows, ft=0, bank=25.0)
    data["statistics"][2] = {"itb": 25.0, "ft": 1, "nt": 0, "pt": 0, "chip": None}
    first, second = normalize_solution(data, prepared)
    stable = list(range(1001, 1016))
    assert first["state_mode"] == "fresh_build"
    assert first["permanent_squad_start"] == []
    assert first["permanent_squad_end"] == stable
    assert first["permanent_bank_start"] == first["budget_start"] == 1000
    assert first["permanent_bank_end"] == first["budget_end"] == 250
    assert first["temporary_selection_bank_end"] is None
    assert (first["q_start"], first["q_end"]) == (0, 1)
    assert second["state_mode"] == "continuing"
    assert second["permanent_squad_start"] == second["permanent_squad_end"] == stable
    assert second["permanent_bank_start"] == second["permanent_bank_end"] == 250
    assert second["temporary_selection_bank_end"] is None
    assert (second["q_start"], second["q_end"]) == (1, 2)


def test_continuing_transfer_exports_exact_permanent_squad_transition():
    prepared = inputs()
    changed = set(range(1, 16)) - {3} | {16}
    rows = week_rows(1, changed, ins={16}, outs={3}) + week_rows(2, changed)
    data = solution(rows, bank=1.8, nt=1)
    data["statistics"][2] = {"itb": 1.8, "ft": 2, "nt": 0, "pt": 0, "chip": None}
    first, second = normalize_solution(data, prepared)
    expected = sorted(code + 1000 for code in changed)
    assert first["state_mode"] == second["state_mode"] == "continuing"
    assert first["permanent_squad_start"] == list(range(1001, 1016))
    assert first["permanent_squad_end"] == first["squad"] == expected
    assert first["transfers_in"] == [1016] and first["transfers_out"] == [1003]
    assert first["permanent_bank_start"] == first["budget_start"] == 20
    assert first["permanent_bank_end"] == first["budget_end"] == 18
    assert second["permanent_squad_start"] == second["permanent_squad_end"] == expected
    assert second["permanent_bank_start"] == second["budget_start"] == 18
    assert (first["q_start"], first["q_end"], second["q_start"], second["q_end"]) == (2, 2, 2, 3)
    assert first["temporary_selection_bank_end"] is second["temporary_selection_bank_end"] is None


def test_free_hit_exports_temporary_residual_without_changing_permanent_next_week():
    prepared = inputs()
    prepared["buy_price"][16] = 6.0
    changed = set(range(1, 16)) - {3} | {16}
    rows = week_rows(1, changed, ins={16}, outs={3})
    rows += week_rows(2, chip="FH")
    rows += week_rows(3, changed)
    data = solution(rows, bank=.8, nt=1)
    data["statistics"][2] = {"itb": .8, "ft": 2, "nt": 0, "pt": 0, "chip": "FH"}
    data["statistics"][3] = {"itb": .8, "ft": 2, "nt": 0, "pt": 0, "chip": None}
    first, free_hit, next_week = normalize_solution(data, prepared)
    permanent = sorted(code + 1000 for code in changed)
    assert first["permanent_squad_end"] == permanent
    assert free_hit["squad"] == list(range(1001, 1016))
    assert free_hit["permanent_squad_start"] == free_hit["permanent_squad_end"] == permanent
    assert free_hit["permanent_bank_start"] == free_hit["permanent_bank_end"] == 8
    assert free_hit["budget_start"] == free_hit["budget_end"] == 8
    # Wealth74m minus72.2m temporary squad; permanent bank remains0.8m.
    assert free_hit["temporary_selection_bank_end"] == 18
    assert free_hit["transfers_in"] == free_hit["transfers_out"] == []
    assert free_hit["q_start"] == free_hit["q_end"] == 2
    assert next_week["permanent_squad_start"] == next_week["permanent_squad_end"] == permanent
    assert next_week["squad"] == permanent
    assert next_week["permanent_bank_start"] == next_week["budget_start"] == 8
    assert next_week["temporary_selection_bank_end"] is None
    assert (next_week["q_start"], next_week["q_end"]) == (2, 3)


def test_permanent_squad_exports_sort_stable_codes_instead_of_element_ids():
    prepared = inputs()
    prepared["element_to_stable"] = {element: 2000 - element for element in range(1, 17)}
    output = normalize_solution(solution(), prepared)[0]
    expected = list(range(1985, 2000))
    assert output["permanent_squad_start"] == output["permanent_squad_end"] == expected


def test_cli_serializes_permanent_state_lists_as_json(tmp_path, monkeypatch):
    import json
    import sys

    from fpl.milp import solio

    source, target = tmp_path / "predictions.csv", tmp_path / "decisions.csv"
    pd.DataFrame({"player_id": [1]}).to_csv(source, index=False)
    normalized = normalize_solution(solution(), inputs())
    monkeypatch.setattr(solio, "run_schedule", lambda *args, **kwargs: pd.DataFrame(normalized))
    monkeypatch.setattr(sys, "argv", ["fpl.milp.solio", "--predictions-csv", str(source),
        "--season", "2026-27", "--start-gw", "229", "--max-gw", "229", "--output", str(target)])
    solio.main()
    row = pd.read_csv(target).iloc[0]
    assert json.loads(row.permanent_squad_start) == normalized[0]["permanent_squad_start"]
    assert json.loads(row.permanent_squad_end) == normalized[0]["permanent_squad_end"]
    assert row.permanent_bank_start == row.budget_start == 20
    assert row.permanent_bank_end == row.budget_end == 20
