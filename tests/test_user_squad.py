"""Public previous-round owner state must never silently discard owned assets."""
from unittest.mock import Mock
from datetime import datetime, timezone

import numpy as np
import pytest

from fpl import run_week
from fpl.milp.optimize import validate_owned_forecast_pool


def bootstrap():
    return {"events": [{"id": 1, "finished": True, "deadline_time": "2026-10-01T10:00:00Z"},
                       {"id": 2, "finished": False, "deadline_time": "2026-10-24T10:00:00Z"},
                       {"id": 3, "finished": False, "deadline_time": "2026-11-07T10:00:00Z"}],
            "elements": [{"id": element, "code": 1000 + element} for element in range(1, 16)]}


def payload():
    return {"picks": [{"element": element} for element in range(1, 16)],
            "entry_history": {"bank": 5}}


def response(monkeypatch, data=None):
    monkeypatch.setattr(run_week, "_utc_now", lambda now: datetime(2026, 10, 31, tzinfo=timezone.utc))
    reply = Mock()
    reply.json.return_value = payload() if data is None else data
    request = Mock(return_value=reply)
    monkeypatch.setattr(run_week.requests, "get", request)
    return request, reply


def test_valid_fifteen_ids_and_bank_use_latest_published_deadline(monkeypatch):
    request, reply = response(monkeypatch)
    codes = list(range(1001, 1016))
    ids, bank = run_week.get_user_squad(77, bootstrap(), set(codes))
    assert ids == codes and bank == 5.0
    request.assert_called_once_with(f"{run_week.FPL_API}/entry/77/event/2/picks/", timeout=30)
    reply.raise_for_status.assert_called_once_with()
    validate_owned_forecast_pool(ids, codes)


def test_unknown_owned_code_rejected_before_it_can_bypass_owned_pool_guard(monkeypatch):
    response(monkeypatch)
    with pytest.raises(ValueError) as exc:
        run_week.get_user_squad(77, bootstrap(), set(range(1001, 1015)))
    assert "1015" in str(exc.value) and "complete owned squad" in str(exc.value)


@pytest.mark.parametrize("count", [0, 14, 16])
def test_partial_or_excess_pick_count_rejected(monkeypatch, count):
    data = payload()
    data["picks"] = [{"element": 1}] * count
    response(monkeypatch, data)
    with pytest.raises(ValueError, match="exactly 15"):
        run_week.get_user_squad(77, bootstrap(), set(range(1001, 1016)))


def test_duplicate_owned_stable_ids_rejected(monkeypatch):
    data = payload()
    data["picks"][-1] = {"element": 1}
    response(monkeypatch, data)
    with pytest.raises(ValueError, match="15 unique player codes"):
        run_week.get_user_squad(77, bootstrap(), set(range(1001, 1016)))


@pytest.mark.parametrize("missing", [None, True, 0, 99])
def test_missing_or_invalid_pick_element_rejected(monkeypatch, missing):
    data = payload()
    data["picks"][-1] = {"element": missing}
    response(monkeypatch, data)
    with pytest.raises(ValueError, match="Owned pick element ID.*missing or invalid") as exc:
        run_week.get_user_squad(77, bootstrap(), set(range(1001, 1016)))
    assert "--cold-start-priors" not in str(exc.value)


@pytest.mark.parametrize("key,bad", [("id", None), ("id", True), ("code", None),
                                    ("code", False), ("code", 1.5), ("code", -1)])
def test_invalid_bootstrap_element_or_code_rejected_without_request(monkeypatch, key, bad):
    register = bootstrap()
    register["elements"][-1][key] = bad
    request, _ = response(monkeypatch)
    with pytest.raises(ValueError, match="Invalid bootstrap element/code"):
        run_week.get_user_squad(77, register, set(range(1001, 1016)))
    request.assert_not_called()


@pytest.mark.parametrize("key", ["id", "code"])
def test_bootstrap_identity_collisions_rejected(monkeypatch, key):
    register = bootstrap()
    register["elements"][-1][key] = register["elements"][0][key]
    request, _ = response(monkeypatch)
    with pytest.raises(ValueError, match="Ambiguous bootstrap element/code"):
        run_week.get_user_squad(77, register, set(range(1001, 1016)))
    request.assert_not_called()


@pytest.mark.parametrize("bank", [None, True, -1, np.nan, np.inf, "5", 5j])
def test_bank_must_be_finite_nonnegative_numeric(monkeypatch, bank):
    data = payload()
    data["entry_history"]["bank"] = bank
    response(monkeypatch, data)
    with pytest.raises(ValueError, match="bank must be nonnegative integer tenths"):
        run_week.get_user_squad(77, bootstrap(), set(range(1001, 1016)))


def test_before_first_deadline_fails_with_explicit_state_guidance(monkeypatch):
    register = bootstrap()
    for event in register["events"]:
        event["deadline_time"] = "2026-11-07T10:00:00Z"
    request, _ = response(monkeypatch)
    with pytest.raises(ValueError) as exc:
        run_week.get_user_squad(77, register, set(range(1001, 1016)))
    assert "No published deadline squad" in str(exc.value)
    assert "owner state" in str(exc.value)
    request.assert_not_called()


def test_missing_bootstrap_element_mapping_and_invalid_payloads(monkeypatch):
    request, _ = response(monkeypatch)
    for register in (None, {}, {"events": [{"id": 1, "finished": True}]}):
        with pytest.raises(ValueError):
            run_week.get_user_squad(77, register, set(range(1001, 1016)))
    request.assert_not_called()
    register = bootstrap()
    register["elements"] = register["elements"][:-1]
    with pytest.raises(ValueError, match="Owned pick element ID 15.*missing or invalid"):
        run_week.get_user_squad(77, register, set(range(1001, 1016)))
