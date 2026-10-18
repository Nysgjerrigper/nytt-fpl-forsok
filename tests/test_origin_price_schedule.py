"""Mocked rolling origin-price state tests; no solver or model fits execute."""

import gzip
import hashlib
import json

import pandas as pd
import pytest

from fpl.milp import solio
from fpl.milp.origin_prices import load_origin_captures


_OWNED = list(range(1001, 1016))
_REPLACEMENT = [*range(1002, 1017)]


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _captures(tmp_path, price_changes):
    roles = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3 + [2]
    records = {}
    for index, changes in enumerate(price_changes):
        origin = 153 + index
        local = index + 1
        folder = tmp_path / f"origin-{origin}"
        folder.mkdir()
        deadline = pd.Timestamp("2024-08-16T17:30:00Z") + pd.Timedelta(days=index * 7)
        bootstrap = {"teams": [{"id": team, "name": f"club{team}"} for team in range(1, 21)],
            "elements": [{"id": player, "code": player + 1000, "team": (player - 1) % 5 + 1,
                          "element_type": role, "now_cost": changes.get(player + 1000, 50),
                          "web_name": f"captured_{player}"} for player, role in enumerate(roles, 1)],
            "events": [{"id": event, "deadline_time": (pd.Timestamp("2024-08-16T17:30:00Z") +
                         pd.Timedelta(days=(event - 1) * 7)).isoformat()} for event in range(1, len(price_changes) + 1)]}
        payload = json.dumps(bootstrap).encode()
        (folder / "bootstrap-static.json.gz").write_bytes(gzip.compress(payload))
        manifest = {"event": local, "event_deadline": deadline.isoformat(),
                    "captured_at": (deadline - pd.Timedelta(hours=1)).isoformat(), "backfill": {"season": "2024-25"},
                    "files": [{"name": "bootstrap-static.json.gz", "endpoint": "/bootstrap-static/", "sha256": _sha(payload)}]}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        records[str(origin)] = {"folder": folder.name, "input_sha256": {
            name: _sha((folder / name).read_bytes()) for name in ("manifest.json", "bootstrap-static.json.gz")}}
    descriptor = tmp_path / "captures.json"
    descriptor.write_text(json.dumps({"schema_version": 1, "season": "2024-25", "captures": records}))
    return load_origin_captures(descriptor, season="2024-25", origins=list(range(153, 153 + len(price_changes))))


def _forecasts(rounds):
    return pd.DataFrame([
        {"origin_gw": origin, "GW": target, "player_id": player, "predicted_total_points": 3.,
         "fixture": target, "name": "wrong_future_name", "team": "wrong_future_club", "position": "FWD",
         "value": 999, "actual_total_points": "outcome must never reach solver", "minutes": "unknown future"}
        for origin in range(153, 153 + rounds) for target in range(origin, 153 + rounds)
        for player in [*range(1001, 1017), 9999]
    ])


def _decision(gw, squad, *, ins=(), outs=(), bank=20, ft=2, chip=None):
    return pd.DataFrame([{"gameweek": gw, "squad": list(squad), "transfers_in": list(ins), "transfers_out": list(outs),
        "chip_played": chip, "budget_end": bank, "q_end": ft,
        "lineup": list(squad[:11]), "bench_order": list(squad[11:]),
        "captain": [squad[0]], "vice_captain": [squad[1]]}])


def test_fresh_acquisition_carries_into_second_origin_half_profit_without_outcome_access(tmp_path, monkeypatch):
    captures = _captures(tmp_path, [{}, {1001: 54}])
    forecasts = _forecasts(2)
    original = forecasts.copy(deep=True)
    seen = []
    def solve(frame, **kwargs):
        seen.append((frame.copy(), kwargs.copy()))
        assert not any(column.startswith("actual_") for column in frame)
        assert "minutes" not in frame and "wrong_future_name" not in set(frame.name)
        assert not frame.player_id.eq(9999).any()
        assert set(frame.team) != {"wrong_future_club"}
        assert set(frame.name) <= {f"captured_{i}" for i in range(1, 17)}
        if kwargs["start_gw"] == 153:
            assert kwargs["initial_squad"] is None
            assert kwargs["sell_prices"] == {}
            return _decision(153, _OWNED, ins=_OWNED, bank=250, ft=1)
        assert kwargs["initial_squad"] == _OWNED
        assert kwargs["bank"] == 250 and kwargs["free_transfers"] == 1
        assert kwargs["sell_prices"][1001] == 52
        assert frame.loc[frame.player_id == 1001, "value"].eq(54).all()
        return _decision(154, _OWNED, bank=250, ft=2)
    monkeypatch.setattr(solio, "solve_window", solve)
    monkeypatch.setattr(solio, "score_decisions", lambda *a, **k: pytest.fail("capture-mode schedule read outcomes"))
    result = solio.run_schedule(forecasts, season="2024-25", start_gw=153, max_gw=154, horizon=2,
                               free_transfers=0, origin_price_captures=captures)
    assert len(seen) == 2
    assert result.gameweek.tolist() == [153, 154]
    assert "actual_total_points" not in result and "actual_net_points" not in result
    assert result.solver_outcome_columns.eq("ABSENT").all()
    assert result.price_policy.eq("origin_capture_half_profit_static_horizon_v1").all()
    assert result.price_input_receipt.iloc[0]["excluded_player_ids"] == [9999]
    pd.testing.assert_frame_equal(forecasts, original)


def test_sale_then_rebuy_resets_basis_before_later_half_profit_sale(tmp_path, monkeypatch):
    captures = _captures(tmp_path, [{1001: 54}, {1001: 60}, {1001: 64}])
    calls = []
    def solve(frame, **kwargs):
        calls.append(kwargs)
        week = kwargs["start_gw"]
        if week == 153:
            assert kwargs["sell_prices"][1001] == 52
            return _decision(week, _REPLACEMENT, ins=[1016], outs=[1001], bank=22)
        if week == 154:
            assert kwargs["initial_squad"] == _REPLACEMENT
            assert kwargs["sell_prices"][1016] == 50
            assert 1001 not in kwargs["sell_prices"]
            return _decision(week, _OWNED, ins=[1001], outs=[1016], bank=12)
        assert kwargs["initial_squad"] == _OWNED
        assert kwargs["sell_prices"][1001] == 62  # Rebuy at60; old50 basis would give57.
        return _decision(week, _OWNED, bank=12, ft=3)
    monkeypatch.setattr(solio, "solve_window", solve)
    basis = {player: 50 for player in _OWNED}
    solio.run_schedule(_forecasts(3), season="2024-25", start_gw=153, max_gw=155, horizon=3,
        initial_squad=_OWNED, bank=20, free_transfers=2, purchase_prices=basis, origin_price_captures=captures)
    assert len(calls) == 3
    assert basis == {player: 50 for player in _OWNED}


def test_free_hit_keeps_original_acquisition_basis_and_permanent_bank(tmp_path, monkeypatch):
    captures = _captures(tmp_path, [{1001: 54}, {1001: 60, 1016: 80}, {1001: 64}])
    calls = []
    def solve(frame, **kwargs):
        calls.append(kwargs)
        if kwargs["start_gw"] == 153:
            assert kwargs["sell_prices"][1001] == 52
            return _decision(153, _REPLACEMENT, chip="FH", bank=999, ft=2)
        assert kwargs["initial_squad"] == _OWNED
        assert kwargs["bank"] == 20  # Temporary reported bank cannot replace permanent bank.
        assert kwargs["previous_free_hit_gw"] == 1
        assert 1016 not in kwargs["sell_prices"]
        expected = 55 if kwargs["start_gw"] == 154 else 57
        assert kwargs["sell_prices"][1001] == expected
        return _decision(kwargs["start_gw"], _OWNED, bank=20, ft=3)
    monkeypatch.setattr(solio, "solve_window", solve)
    result = solio.run_schedule(_forecasts(3), season="2024-25", start_gw=153, max_gw=155, horizon=3,
        initial_squad=_OWNED, bank=20, free_transfers=2,
        purchase_prices={player: 50 for player in _OWNED}, origin_price_captures=captures,
        chip_inventory={"fh": {1: 1}})
    assert len(calls) == 3 and result.iloc[0].chip_played == "FH"


@pytest.mark.parametrize("conflict", ["missing_capture", "extra_capture", "bootstrap", "sell_prices",
                                      "fresh_basis", "missing_owned_basis", "no_origin"])
def test_capture_mode_rejects_incomplete_coverage_or_conflicting_state_before_solving(tmp_path, monkeypatch, conflict):
    captures = _captures(tmp_path, [{}, {}])
    predictions = _forecasts(2)
    kwargs = {"origin_price_captures": captures}
    if conflict == "missing_capture": captures.pop(154)
    elif conflict == "extra_capture": captures[155] = captures[154]
    elif conflict == "bootstrap": kwargs["bootstrap"] = {}
    elif conflict == "sell_prices": kwargs["sell_prices"] = {}
    elif conflict == "fresh_basis": kwargs["purchase_prices"] = {1001: 50}
    elif conflict == "missing_owned_basis": kwargs["initial_squad"] = _OWNED
    else: predictions = predictions.drop(columns="origin_gw")
    monkeypatch.setattr(solio, "solve_window", lambda *a, **k: pytest.fail("invalid capture state reached solve"))
    with pytest.raises(ValueError):
        solio.run_schedule(predictions, season="2024-25", start_gw=153, max_gw=154, **kwargs)


def test_purchase_basis_cannot_be_passed_without_origin_capture_mode(monkeypatch):
    monkeypatch.setattr(solio, "solve_window", lambda *a, **k: pytest.fail("invalid basis reached solve"))
    with pytest.raises(ValueError, match="purchase_prices"):
        solio.run_schedule(_forecasts(1), season="2024-25", start_gw=153, max_gw=153, purchase_prices={1001: 50})
