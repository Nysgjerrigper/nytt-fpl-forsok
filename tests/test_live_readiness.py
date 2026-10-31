"""Deadline-aware target selection and complete published-squad reconciliation."""

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from fpl import run_week


NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def _event(identifier, deadline, *, finished=False):
    return {"id": identifier, "deadline_time": deadline, "finished": finished}


def _bootstrap():
    return {
        "events": [
            _event(7, "2026-10-10T10:00:00Z", finished=True),
            _event(8, "2026-10-17T11:00:00Z"),
            _event(9, "2026-10-24T10:00:00Z"),
        ],
        "elements": [{"id": i, "code": 1000 + i, "first_name": "Player", "second_name": str(i)}
                     for i in range(1, 16)],
    }


def _published():
    return {"picks": [{"element": i} for i in range(1, 16)], "entry_history": {"bank": 27}}


def _mock_picks(monkeypatch, payload):
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: deepcopy(payload))
    monkeypatch.setattr(run_week.requests, "get", get)
    return calls


def test_current_unfinished_round_with_passed_deadline_is_skipped():
    assert run_week.determine_target_gw(_bootstrap(), now=NOW) == 9


def test_round_at_exact_deadline_is_no_longer_actionable():
    bootstrap = _bootstrap()
    bootstrap["events"][1]["deadline_time"] = NOW.isoformat()
    assert run_week.determine_target_gw(bootstrap, now=NOW) == 9


def test_lowest_unfinished_future_event_id_selected_independent_of_input_order():
    bootstrap = {"events": [
        _event(10, "2026-10-31T10:00:00Z"),
        _event(8, "2026-10-17T13:00:00Z"),
        _event(9, "2026-10-24T10:00:00Z"),
    ]}
    assert run_week.determine_target_gw(bootstrap, now=NOW) == 8


def test_offset_aware_deadlines_and_now_compare_as_instants():
    bootstrap = {"events": [
        _event(8, "2026-10-17T14:00:00+02:00"),
        _event(9, "2026-10-17T13:01:00+01:00"),
    ]}
    offset_now = datetime.fromisoformat("2026-10-17T08:00:00-04:00")
    assert run_week.determine_target_gw(bootstrap, now=offset_now) == 9


@pytest.mark.parametrize("deadline", ["not-a-date", "2026-10-24T10:00:00", "", None])
def test_malformed_or_naive_unfinished_deadline_rejected(deadline):
    bootstrap = {"events": [_event(8, deadline), _event(9, "2026-10-24T10:00:00Z")]}
    with pytest.raises(ValueError):
        run_week.determine_target_gw(bootstrap, now=NOW)


def test_missing_unfinished_deadline_rejected():
    with pytest.raises(ValueError):
        run_week.determine_target_gw({"events": [{"id": 8, "finished": False}]}, now=NOW)


def test_naive_now_rejected():
    with pytest.raises(ValueError):
        run_week.determine_target_gw(_bootstrap(), now=NOW.replace(tzinfo=None))


@pytest.mark.parametrize("events", [[], [_event(38, "2026-05-24T14:00:00Z", finished=True)],
                                    [_event(38, "2026-05-24T14:00:00Z")]])
def test_season_without_actionable_deadline_rejected(events):
    with pytest.raises(ValueError):
        run_week.determine_target_gw({"events": events}, now=NOW)


def test_duplicate_event_ids_rejected():
    bootstrap = {"events": [_event(8, "2026-10-24T10:00:00Z"), _event(8, "2026-10-31T10:00:00Z")]}
    with pytest.raises(ValueError):
        run_week.determine_target_gw(bootstrap, now=NOW)


def test_latest_published_picks_use_passed_deadline_instead_of_finished_flag(monkeypatch):
    bootstrap = _bootstrap()
    # Input ordering must not determine which past event is newest.
    bootstrap["events"].reverse()
    calls = _mock_picks(monkeypatch, _published())
    ids, bank = run_week.get_user_squad(42, bootstrap, set(range(1001, 1016)), now=NOW)
    assert ids == list(range(1001, 1016))
    assert bank == 27.0 and isinstance(bank, float)
    assert len(calls) == 1
    assert calls[0][0].endswith("/entry/42/event/8/picks/")


def test_exact_deadline_counts_as_published_round_for_picks(monkeypatch):
    bootstrap = _bootstrap()
    bootstrap["events"][1]["deadline_time"] = NOW.isoformat()
    calls = _mock_picks(monkeypatch, _published())
    run_week.get_user_squad(42, bootstrap, set(range(1001, 1016)), now=NOW)
    assert calls[0][0].endswith("/event/8/picks/")


def test_unknown_owned_player_fails_instead_of_silently_dropping(monkeypatch):
    _mock_picks(monkeypatch, _published())
    with pytest.raises(ValueError):
        run_week.get_user_squad(42, _bootstrap(), set(range(1001, 1015)), now=NOW)


@pytest.mark.parametrize("change", ["missing_element", "missing_code", "null_code"])
def test_owned_player_identity_must_reconcile_completely(monkeypatch, change):
    bootstrap = _bootstrap()
    if change == "missing_element":
        bootstrap["elements"].pop()
    elif change == "missing_code":
        bootstrap["elements"][-1].pop("code")
    else:
        bootstrap["elements"][-1]["code"] = None
    _mock_picks(monkeypatch, _published())
    with pytest.raises(ValueError):
        run_week.get_user_squad(42, bootstrap, set(range(1001, 1016)), now=NOW)


@pytest.mark.parametrize("change", ["fourteen", "sixteen", "duplicate_pick", "duplicate_code"])
def test_published_squad_requires_exactly_fifteen_unique_mapped_players(monkeypatch, change):
    bootstrap, published = _bootstrap(), _published()
    if change == "fourteen":
        published["picks"].pop()
    elif change == "sixteen":
        published["picks"].append({"element": 16})
    elif change == "duplicate_pick":
        published["picks"][-1] = published["picks"][0].copy()
    else:
        bootstrap["elements"][-1]["code"] = bootstrap["elements"][0]["code"]
    _mock_picks(monkeypatch, published)
    with pytest.raises(ValueError):
        run_week.get_user_squad(42, bootstrap, set(range(1001, 1017)), now=NOW)


@pytest.mark.parametrize("bank", [True, False, -1, 2.5, float("nan"), float("inf"), "27", None])
def test_bank_requires_nonnegative_integer_tenths(monkeypatch, bank):
    published = _published()
    published["entry_history"]["bank"] = bank
    _mock_picks(monkeypatch, published)
    with pytest.raises(ValueError):
        run_week.get_user_squad(42, _bootstrap(), set(range(1001, 1016)), now=NOW)


def test_zero_bank_is_valid(monkeypatch):
    published = _published()
    published["entry_history"]["bank"] = 0
    _mock_picks(monkeypatch, published)
    _, bank = run_week.get_user_squad(42, _bootstrap(), set(range(1001, 1016)), now=NOW)
    assert bank == 0.0


def test_preseason_has_no_public_picks_and_must_fail_before_request(monkeypatch):
    bootstrap = _bootstrap()
    bootstrap["events"] = [_event(1, "2026-10-24T10:00:00Z")]
    calls = _mock_picks(monkeypatch, _published())
    with pytest.raises(ValueError):
        run_week.get_user_squad(42, bootstrap, set(range(1001, 1016)), now=NOW)
    assert calls == []


@pytest.mark.parametrize("scenario", ["expired_season", "team_without_free_transfers"])
def test_main_rejects_invalid_live_setup_before_refresh_or_fitting(monkeypatch, scenario):
    import sys

    bootstrap = _bootstrap()
    argv = ["fpl.run_week"]
    if scenario == "expired_season":
        bootstrap["events"] = [_event(38, "2026-10-10T10:00:00Z", finished=True)]
    else:
        argv += ["--team-id", "42"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(run_week, "_utc_now", lambda now: NOW)
    monkeypatch.setattr(run_week, "fetch_bootstrap", lambda: bootstrap)
    monkeypatch.setattr(run_week.fetch, "build_master_dataset",
                        lambda: pytest.fail("invalid setup refreshed mutable dataset"))
    monkeypatch.setattr(run_week.features, "build_feature_frame",
                        lambda *args, **kwargs: pytest.fail("invalid setup built features"))
    monkeypatch.setattr(run_week, "fit_holdout_weights",
                        lambda *args, **kwargs: pytest.fail("invalid setup fitted weights"))
    monkeypatch.setattr(run_week, "fit_position_ensembles",
                        lambda *args, **kwargs: pytest.fail("invalid setup fitted models"))
    if scenario == "expired_season":
        with pytest.raises(ValueError, match="No open"):
            run_week.main()
    else:
        with pytest.raises(SystemExit) as error:
            run_week.main()
        assert error.value.code == 2


def test_main_converts_only_solio_boundary_to_global_gameweeks(monkeypatch, tmp_path):
    import sys

    import pandas as pd

    from fpl.milp import solio

    bootstrap = {"events": [
        _event(1, "2026-08-15T10:00:00Z", finished=True),
        *[_event(gw, "2099-10-17T10:00:00Z") for gw in (8, 9, 10)],
    ]}
    local_predictions = pd.DataFrame({
        "player_id": [1001] * 3, "GW": [8, 9, 10], "name": ["Player"] * 3,
        "position": ["MID"] * 3, "team": ["Arsenal"] * 3,
        "value": [70] * 3, "predicted_total_points": [5.0, 6.0, 7.0],
    })
    dataset = tmp_path / "live.csv"
    pd.DataFrame({"player_id": [1001], "GW_global": [235]}).to_csv(dataset, index=False)
    forecasts = tmp_path / "forecasts" / "weekly.csv"
    output = tmp_path / "squads"
    monkeypatch.setattr(sys, "argv", ["fpl.run_week", "--optimizer", "solio", "--horizon", "3"])
    monkeypatch.setattr(run_week.config, "MASTER_DATASET_PATH", dataset)
    monkeypatch.setattr(run_week.config, "LIVE_FORECAST_DIR", tmp_path / "captures")
    monkeypatch.setattr(run_week.config, "PREDICTIONS_PATH", forecasts)
    monkeypatch.setattr(run_week.config, "SQUAD_OUTPUT_DIR", output)
    monkeypatch.setattr(run_week, "fetch_bootstrap", lambda: bootstrap)
    monkeypatch.setattr(run_week, "determine_target_gw", lambda _: 8)
    monkeypatch.setattr(run_week.fetch, "active_season_from_bootstrap", lambda _: "2026-27")
    monkeypatch.setattr(run_week.fetch, "build_master_dataset", lambda: None)
    monkeypatch.setattr(run_week.features, "build_feature_frame", lambda frame: frame.copy())
    monkeypatch.setattr(run_week.features, "feature_columns", lambda _: [])
    monkeypatch.setattr(run_week.features, "team_form_asof", lambda _: {})
    monkeypatch.setattr(run_week, "fit_holdout_weights", lambda *a, **k: {})
    monkeypatch.setattr(run_week, "fit_position_ensembles", lambda *a, **k: {})
    monkeypatch.setattr(run_week, "build_live_snapshot", lambda frame: frame.copy())
    monkeypatch.setattr(run_week, "apply_live_identity", lambda frame, _: frame)
    monkeypatch.setattr(run_week, "build_future_predictions", lambda *a, **k: local_predictions.copy())
    monkeypatch.setattr(run_week, "filter_to_registered", lambda frame, _: frame)
    monkeypatch.setattr(run_week, "availability_multipliers", lambda _: {})
    monkeypatch.setattr(run_week, "live_prices", lambda _: {})
    monkeypatch.setattr(run_week, "apply_availability", lambda frame, _: frame)
    monkeypatch.setattr(run_week, "apply_live_prices", lambda frame, _: frame)
    captured = []
    def schedule(predictions, **kwargs):
        captured.append((predictions.copy(), kwargs))
        return pd.DataFrame({"gameweek": [236], "captain": [1001], "transfers_in": [[]],
                             "transfers_out": [[]], "chip_played": [None], "budget_end": [1000.0]})
    monkeypatch.setattr(solio, "run_schedule", schedule)
    run_week.main()
    assert len(captured) == 1
    predictions, arguments = captured[0]
    assert predictions.GW.tolist() == [236, 237, 238]
    assert arguments["season"] == "2026-27"
    assert arguments["start_gw"] == 236
    assert arguments["max_gw"] == 238
    assert arguments["horizon"] == 3
    assert arguments["initial_squad"] is None
    assert arguments["bank"] == 1000.0
    assert arguments["free_transfers"] == 0
    assert pd.read_csv(forecasts).GW.tolist() == [8, 9, 10]
    assert pd.read_csv(output / "solio_weekly.csv").gameweek.tolist() == [236]
    assert local_predictions.GW.tolist() == [8, 9, 10]
