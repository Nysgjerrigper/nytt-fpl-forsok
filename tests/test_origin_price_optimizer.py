"""Opt-in lagged-price optimizer contracts using tiny synthetic exact MILPs."""
import numpy as np
import pandas as pd
import pytest

from fpl.milp import optimize

ROLES = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3


def _files(tmp_path, *, end=2, origin=True, missing_owned=False):
    rows, prices = [], []
    for current in range(2, end + 1):
        for gw in range(current, end + 1):
            for pid in range(1, 17):
                row = {"player_id": pid, "name": f"p{pid}", "position": ROLES[pid - 1] if pid <= 15 else "MID",
                       "team": f"club{(pid - 1) % 5}", "GW": gw, "value": 50. + (gw - 2) * 10,
                       "predicted_total_points": 100. if pid == 16 else 2., "actual_total_points": 2., "minutes": 90.}
                if origin:
                    row["origin_gw"] = current
                rows.append(row)
        for pid in range(1, 16):
            if missing_owned and pid == 1:
                continue
            prices.append({"player_id": pid, "origin_gw": current, "value": 50. + (current - 2) * 5,
                           "price_observed_gw": current - 1})
    predictions = tmp_path / "predictions.csv"
    catalog = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(predictions, index=False)
    pd.DataFrame(prices).to_csv(catalog, index=False)
    return predictions, catalog


def _arguments(tmp_path, predictions, *, end=2, extras=()):
    return optimize.parse_args(["--predictions-csv", str(predictions), "--start-gw", "2", "--max-gw", str(end),
                                "--horizon", "2", "--solver", "cbc", "--gap-rel", "0",
                                "--scoring-mode", "corrected", "--output", str(tmp_path / "output.csv"), *extras])


def test_asof_price_is_constant_across_horizon_and_updates_at_next_origin():
    catalog = pd.DataFrame({"player_id": [1, 2, 1, 2], "origin_gw": [2, 2, 3, 3],
                            "value": [50., 60., 55., 65.], "price_observed_gw": [1, 1, 2, 2]})
    replay = pd.DataFrame({2: [70., 80., 99.], 3: [75., 85., 999.], 4: [90., 100., 1999.]}, index=[1, 2, 3])
    first, eligible = optimize.origin_price_state(catalog, 2, [1, 2, 3], [2, 3, 4], replay, "origin-asof")
    assert eligible == {1, 2}
    assert first.loc[1].tolist() == [50., 50., 50.]
    assert first.loc[2].tolist() == [60., 60., 60.]
    assert first.loc[3].tolist() == [0., 0., 0.]
    next_origin, eligible = optimize.origin_price_state(catalog, 3, [1, 2, 3], [3, 4], replay, "origin-asof")
    assert next_origin.loc[1].tolist() == [55., 55.]
    assert next_origin.loc[2].tolist() == [65., 65.]
    target, same_eligible = optimize.origin_price_state(catalog, 2, [1, 2, 3], [2, 3, 4], replay, "target-replay")
    assert same_eligible == eligible
    assert target.loc[1].tolist() == [70., 75., 90.]
    assert target.loc[3].tolist() == [0., 0., 0.]
    pd.testing.assert_frame_equal(replay, pd.DataFrame({2: [70., 80., 99.], 3: [75., 85., 999.], 4: [90., 100., 1999.]}, index=[1, 2, 3]))


@pytest.mark.parametrize("policy", ["target-replay", "origin-asof", "origin-current-proxy"])
def test_catalog_unknown_price_excludes_high_scoring_player_from_synthetic_milp(tmp_path, policy):
    predictions, catalog = _files(tmp_path)
    args = _arguments(tmp_path, predictions, extras=("--origin-price-catalog", str(catalog), "--origin-price-policy", policy))
    result = optimize.run(args)
    assert len(result) == 1 and result.iloc[0].squad == list(range(1, 16))
    assert 16 not in result.iloc[0].transfers_in and 16 not in result.iloc[0].lineup
    assert result.origin_price_policy.tolist() == [policy]
    assert result.price_eligibility.tolist() == ["strictly_prior_observed_price"]
    assert result.iloc[0].objective_value is not None


def test_rolling_asof_milp_uses_each_origin_prices_with_frozen_horizon(tmp_path, monkeypatch):
    predictions, catalog = _files(tmp_path, end=3)
    calls = []
    original = optimize.origin_price_state
    def record(catalog, origin, players, horizon, replay, policy):
        matrix, eligible = original(catalog, origin, players, horizon, replay, policy)
        calls.append((origin, list(horizon), matrix.copy()))
        return matrix, eligible
    monkeypatch.setattr(optimize, "origin_price_state", record)
    args = _arguments(tmp_path, predictions, end=3, extras=("--origin-price-catalog", str(catalog), "--origin-price-policy", "origin-asof"))
    result = optimize.run(args)
    assert len(result) == 2 and [call[0] for call in calls] == [2, 3]
    assert calls[0][1] == [2, 3] and calls[0][2].loc[1].tolist() == [50., 50.]
    assert calls[1][1] == [3] and calls[1][2].loc[1].tolist() == [55.]
    assert all(squad == list(range(1, 16)) for squad in result.squad)


def test_default_without_catalog_preserves_historical_target_replay(tmp_path, monkeypatch):
    predictions, _ = _files(tmp_path)
    monkeypatch.setattr(optimize, "origin_price_state", lambda *a, **k: pytest.fail("default reached opt-in price helper"))
    default_args = _arguments(tmp_path, predictions)
    assert default_args.origin_price_catalog is None and default_args.origin_price_policy == "target-replay"
    default = optimize.run(default_args)
    explicit = optimize.run(_arguments(tmp_path, predictions, extras=("--origin-price-policy", "target-replay")))
    pd.testing.assert_frame_equal(default, explicit)
    assert 16 in default.iloc[0].squad and default.iloc[0].captain == [16]
    assert "origin_price_policy" not in default and "price_eligibility" not in default


@pytest.mark.parametrize("policy", ["origin-asof", "origin-current-proxy"])
def test_frozen_price_policy_without_catalog_rejected_before_solve(tmp_path, monkeypatch, policy):
    predictions, _ = _files(tmp_path)
    monkeypatch.setattr(optimize.pulp.LpProblem, "solve", lambda *a, **k: pytest.fail("invalid price policy reached solve"))
    with pytest.raises(ValueError, match=f"{policy} requires"):
        optimize.run(_arguments(tmp_path, predictions, extras=("--origin-price-policy", policy)))


@pytest.mark.parametrize("policy", ["target-replay", "origin-asof", "origin-current-proxy"])
def test_catalog_requires_origin_forecasts(tmp_path, monkeypatch, policy):
    predictions, catalog = _files(tmp_path, origin=False)
    monkeypatch.setattr(optimize.pulp.LpProblem, "solve", lambda *a, **k: pytest.fail("nonorigin prices reached solve"))
    with pytest.raises(ValueError, match="requires origin forecasts"):
        optimize.run(_arguments(tmp_path, predictions, extras=("--origin-price-catalog", str(catalog), "--origin-price-policy", policy)))


@pytest.mark.parametrize("policy", ["target-replay", "origin-asof", "origin-current-proxy"])
def test_owned_player_missing_origin_price_fails_closed_before_solve(tmp_path, monkeypatch, policy):
    predictions, catalog = _files(tmp_path, missing_owned=True)
    monkeypatch.setattr(optimize.pulp.LpProblem, "solve", lambda *a, **k: pytest.fail("missing owned price reached solve"))
    args = _arguments(tmp_path, predictions, extras=("--origin-price-catalog", str(catalog), "--origin-price-policy", policy,
                                                    "--initial-squad", ",".join(map(str, range(1, 16))), "--initial-budget", "250"))
    with pytest.raises(ValueError, match="owned player lacks an origin-known price"):
        optimize.run(args)


def test_target_replay_known_prices_still_must_be_finite_and_positive():
    catalog = pd.DataFrame({"player_id": [1], "origin_gw": [2], "value": [50.], "price_observed_gw": [1]})
    replay = pd.DataFrame({2: [np.nan], 3: [60.]}, index=[1])
    with pytest.raises(ValueError, match="finite and positive"):
        optimize.origin_price_state(catalog, 2, [1], [2, 3], replay, "target-replay")
    matrix, eligible = optimize.origin_price_state(catalog, 2, [1], [2, 3], replay, "origin-asof")
    assert matrix.loc[1].tolist() == [50., 50.] and eligible == {1}


def _distinct_all_known_inputs(tmp_path):
    """Sixteen eligible players with a unique affordable squad and captain ordering."""
    predictions, catalog = _files(tmp_path, end=3)
    frame = pd.read_csv(predictions)
    frame["predicted_total_points"] = np.where(frame.player_id == 16, 100., frame.player_id * 2.)
    frame["actual_total_points"] = frame.player_id.astype(float)
    frame.to_csv(predictions, index=False)
    prices = pd.read_csv(catalog)
    extra = pd.DataFrame({"player_id": [16, 16], "origin_gw": [2, 3], "value": [50., 55.],
                          "price_observed_gw": [1, 2]})
    pd.concat([prices, extra], ignore_index=True).to_csv(catalog, index=False)
    return predictions, catalog


def test_end_to_end_asof_budget_and_decisions_ignore_future_target_price_changes(tmp_path):
    predictions, catalog = _distinct_all_known_inputs(tmp_path)
    args = _arguments(tmp_path, predictions, end=3, extras=("--origin-price-catalog", str(catalog),
                                                          "--origin-price-policy", "origin-asof"))
    before = optimize.run(args)
    before_export = pd.read_csv(args.output)
    frame = pd.read_csv(predictions)
    # Deliberately change every target-price field. Both sets remain positive and
    # nonbinding for a 15-player squad; only admitted prior catalog prices may enter.
    frame["value"] = 25. + frame.player_id * .25 + frame.GW * .5
    assert frame.value.gt(0).all() and frame.groupby(["origin_gw", "GW"]).value.sum().max() < 1000.
    frame.to_csv(predictions, index=False)
    after = optimize.run(args)
    substantive = ["gameweek", "squad", "lineup", "captain", "vice_captain", "transfers_in", "transfers_out",
                   "budget_start", "budget_end", "alpha", "q_start", "objective_value"]
    pd.testing.assert_frame_equal(before[substantive], after[substantive], check_exact=True)
    pd.testing.assert_frame_equal(before_export, pd.read_csv(args.output), check_exact=True)
    assert before.captain.tolist() == [[16], [16]]
    assert all(len(squad) == 15 for squad in before.squad)
    assert before.iloc[0].budget_end == pytest.approx(250.)
    assert all(value is not None for value in before.objective_value)


def test_all_known_target_replay_catalog_preserves_legacy_decisions_and_budget(tmp_path):
    predictions, catalog = _distinct_all_known_inputs(tmp_path)
    legacy = optimize.run(_arguments(tmp_path, predictions, end=3))
    replay = optimize.run(_arguments(tmp_path, predictions, end=3,
                                    extras=("--origin-price-catalog", str(catalog),
                                            "--origin-price-policy", "target-replay")))
    pd.testing.assert_frame_equal(replay.drop(columns=["origin_price_policy", "price_eligibility"]),
                                  legacy, check_exact=True)
    assert legacy.captain.tolist() == [[16], [16]]
    assert legacy.iloc[0].budget_end == pytest.approx(250.)


def test_current_proxy_freezes_current_replay_price_and_updates_next_origin():
    # Reconstructed current values are shared by the paired policies. This is not
    # an assertion of actual deadline publication or blank-price availability.
    catalog = pd.DataFrame({"player_id": [1, 2, 1, 2], "origin_gw": [2, 2, 3, 3],
                            "value": [50., 60., 55., 65.], "price_observed_gw": [1, 1, 2, 2]})
    replay = pd.DataFrame({2: [70., 80., 99.], 3: [75., 85., 999.], 4: [90., 100., 1999.]}, index=[1, 2, 3])
    frozen, eligible = optimize.origin_price_state(catalog, 2, [1, 2, 3], [2, 3, 4], replay, "origin-current-proxy")
    target, target_eligible = optimize.origin_price_state(catalog, 2, [1, 2, 3], [2, 3, 4], replay, "target-replay")
    pd.testing.assert_series_equal(frozen[2], target[2])
    assert frozen.loc[1].tolist() == [70., 70., 70.]
    assert frozen.loc[2].tolist() == [80., 80., 80.]
    assert frozen.loc[3].tolist() == [0., 0., 0.]
    assert target.loc[1].tolist() == [70., 75., 90.]
    assert eligible == target_eligible == {1, 2}
    following, following_eligible = optimize.origin_price_state(catalog, 3, [1, 2, 3], [3, 4], replay, "origin-current-proxy")
    assert following.loc[1].tolist() == [75., 75.]
    assert following.loc[2].tolist() == [85., 85.]
    assert following_eligible == eligible
    prior, _ = optimize.origin_price_state(catalog, 2, [1, 2, 3], [2, 3, 4], replay, "origin-asof")
    assert prior.loc[1].tolist() == [50., 50., 50.]
    assert prior.loc[2].tolist() == [60., 60., 60.]


def _three_round_choice_inputs(tmp_path):
    roles = [*ROLES, "MID", "DEF", "FWD", "GK"]
    clubs = [*[(pid - 1) % 5 for pid in range(1, 16)], 0, 2, 2, 0]
    rows, prices = [], []
    spikes = {16: {2: 70., 3: 80., 4: 3.}, 17: {2: 3., 3: 90., 4: 2.},
              18: {2: 2., 3: 3., 4: 100.}, 19: {2: 60., 3: 3., 4: 5.}}
    for origin in (2, 3, 4):
        for gw in range(origin, 5):
            for pid, (role, club) in enumerate(zip(roles, clubs), start=1):
                rows.append({"player_id": pid, "name": f"p{pid}", "position": role, "team": f"club{club}",
                             "GW": gw, "origin_gw": origin, "value": 50. + (gw - 2) * 10.,
                             "predicted_total_points": spikes[pid][gw] if pid in spikes else round(1.123 ** pid, 6),
                             "actual_total_points": float(pid), "minutes": 90.})
        prices.extend({"player_id": pid, "origin_gw": origin, "value": 40. + (origin - 2) * 5.,
                       "price_observed_gw": origin - 1} for pid in range(1, 20))
    predictions, catalog = tmp_path / "choice_predictions.csv", tmp_path / "choice_prices.csv"
    pd.DataFrame(rows).to_csv(predictions, index=False)
    pd.DataFrame(prices).to_csv(catalog, index=False)
    return predictions, catalog


def test_three_round_current_proxy_keeps_current_budget_costs_with_multiple_choices(tmp_path, monkeypatch):
    predictions, catalog = _three_round_choice_inputs(tmp_path)
    calls = {"target-replay": [], "origin-current-proxy": []}
    original = optimize.origin_price_state
    def record(catalog, origin, players, horizon, replay, policy):
        matrix, eligible = original(catalog, origin, players, horizon, replay, policy)
        calls[policy].append((origin, matrix.copy(), eligible))
        return matrix, eligible
    monkeypatch.setattr(optimize, "origin_price_state", record)
    results = {}
    for policy in calls:
        args = _arguments(tmp_path, predictions, end=4,
                          extras=("--origin-price-catalog", str(catalog), "--origin-price-policy", policy))
        args.horizon = 3
        results[policy] = optimize.run(args)
    target, proxy = results["target-replay"], results["origin-current-proxy"]
    assert len(target) == len(proxy) == 3
    assert target.iloc[0].budget_start == proxy.iloc[0].budget_start == 1000.
    assert target.iloc[0].budget_end == proxy.iloc[0].budget_end == 250.
    assert len(target.iloc[0].squad) == len(proxy.iloc[0].squad) == 15
    assert target.iloc[0].objective_value is not None and proxy.iloc[0].objective_value is not None
    for replay_call, frozen_call in zip(calls["target-replay"], calls["origin-current-proxy"]):
        origin, replay_matrix, eligible = replay_call
        frozen_origin, frozen_matrix, frozen_eligible = frozen_call
        assert origin == frozen_origin and eligible == frozen_eligible == set(range(1, 20))
        pd.testing.assert_series_equal(replay_matrix[origin], frozen_matrix[origin])
        assert (frozen_matrix.to_numpy() == frozen_matrix[origin].to_numpy()[:, None]).all()
        if origin < 4:
            assert not replay_matrix[origin + 1].equals(frozen_matrix[origin + 1])
        for frame in (target, proxy):
            row = frame.loc[frame.gameweek == origin].iloc[0]
            current_price = 50. + (origin - 2) * 10.
            previous = row.budget_start
            expected_budget = (1000. - 15 * current_price if origin == 2 else
                               previous + len(row.transfers_out) * current_price - len(row.transfers_in) * current_price)
            assert row.budget_end == pytest.approx(expected_budget)
    # Uniform, nonbinding current prices make this deliberately small comparison
    # share decisions; more general research may differ on the frozen future path.
    substantive = ["gameweek", "squad", "lineup", "captain", "vice_captain", "transfers_in", "transfers_out",
                   "budget_start", "budget_end", "alpha", "q_start"]
    pd.testing.assert_frame_equal(target[substantive], proxy[substantive], check_exact=True)
