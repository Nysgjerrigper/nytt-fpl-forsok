"""Explicit blanks need a complete fixture ledger and a consistent captured register."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from fpl.milp.blank_outcomes import explicit_blank_outcomes


def _evidence():
    teams = pd.DataFrame({"id": range(1, 21), "name": [f"club{i}" for i in range(1, 21)]})
    # Circle schedule supplies every directed pair exactly once in 38 rounds.
    wheel = list(range(1, 21))
    rows = []
    for event in range(1, 20):
        for index in range(10):
            home, away = wheel[index], wheel[-1-index]
            rows.append({"event": event, "team_h": home, "team_a": away})
            rows.append({"event": event + 19, "team_h": away, "team_a": home})
        wheel = [wheel[0], wheel[-1], *wheel[1:-1]]
    fixtures = pd.DataFrame(rows)
    fixtures["id"] = range(1, 381)
    fixtures["finished"] = True
    fixtures["team_h_score"] = 0
    fixtures["team_a_score"] = 0
    # Move club1's opening fixture to GW2: club1/club20 have explicit GW1 blanks.
    fixtures.loc[(fixtures.event == 1) & ((fixtures.team_h == 1) | (fixtures.team_a == 1)), "event"] = 2
    fixtures["kickoff_time"] = [
        (pd.Timestamp("2024-08-17T14:00:00Z") + pd.Timedelta(days=7 * (event - 1))).isoformat()
        for event in fixtures.event
    ]
    register = {"teams": teams.to_dict("records"),
                "elements": [{"code": 101, "team": 1, "element_type": 3},
                             {"code": 102, "team": 2, "element_type": 2}],
                "events": [{"id": 1, "deadline_time": "2024-08-16T17:30:00Z"},
                           {"id": 2, "deadline_time": "2024-08-23T17:30:00Z"}]}
    snapshot = {"decisions": [{"gameweek": 153, "squad": [101, 102]},
                              {"gameweek": 154, "squad": [999]}]}
    predictions = pd.DataFrame({"player_id": [102], "GW": [153], "team": ["club2"],
                                "position": ["DEF"], "actual_total_points": [2.], "minutes": [90.],
                                "fixture": [2], "predicted_total_points": [4.]})
    return snapshot, predictions, fixtures, teams, register


def _run(evidence, **kwargs):
    snapshot, predictions, fixtures, teams, register = evidence
    return explicit_blank_outcomes(snapshot, predictions, fixtures=fixtures, teams=teams, register=register,
                                   season=kwargs.get("season", "2024-25"),
                                   first_global_gw=kwargs.get("first_global_gw", 153), event=kwargs.get("event", 1))


def test_only_target_missing_blank_produces_separate_zero_audit_without_mutation():
    evidence = _evidence()
    before = deepcopy(evidence)
    unchanged, audit = _run(evidence)
    pd.testing.assert_frame_equal(unchanged, evidence[1])
    assert unchanged is not evidence[1]
    assert audit.to_dict("records") == [{"player_id": 101, "GW": 153, "team": "club1", "position": "MID",
        "actual_total_points": 0., "minutes": 0., "outcome_basis": "EXPLICIT_RETROSPECTIVE_BLANK"}]
    assert "fixture" not in audit and "predicted_total_points" not in audit
    assert evidence[0] == before[0] and evidence[4] == before[4]
    for index in (1, 2, 3):
        pd.testing.assert_frame_equal(evidence[index], before[index])


@pytest.mark.parametrize("change", ["incomplete", "duplicate_id", "duplicate_pair", "self_pair", "unfinished",
                                    "numeric_finished", "bad_kickoff", "negative_score", "fractional_score",
                                    "unknown_club", "zero_event", "bad_fixture_id"])
def test_incomplete_or_contradictory_season_ledger_rejected(change):
    evidence = list(_evidence())
    frame = evidence[2]
    if change == "incomplete":
        evidence[2] = frame.iloc[:-1]
    elif change == "duplicate_id":
        frame.loc[1, "id"] = frame.loc[0, "id"]
    elif change == "duplicate_pair":
        frame.loc[1, ["team_h", "team_a"]] = frame.loc[0, ["team_h", "team_a"]].to_numpy()
    elif change == "self_pair":
        frame.loc[0, "team_a"] = frame.loc[0, "team_h"]
    elif change == "unfinished":
        frame.loc[0, "finished"] = False
    elif change == "numeric_finished":
        frame["finished"] = frame.finished.astype(object)
        frame.loc[0, "finished"] = 1
    elif change == "bad_kickoff":
        frame.loc[0, "kickoff_time"] = "not-a-time"
    elif change in ("negative_score", "fractional_score"):
        frame["team_h_score"] = frame.team_h_score.astype(float)
        frame.loc[0, "team_h_score"] = -1 if change == "negative_score" else .5
    elif change == "unknown_club":
        frame.loc[0, "team_h"] = 99
    elif change == "zero_event":
        frame.loc[0, "event"] = 0
    else:
        frame["id"] = frame.id.astype(object)
        frame.loc[0, "id"] = True
    with pytest.raises(ValueError):
        _run(evidence)


@pytest.mark.parametrize("change", ["missing_team", "duplicate_team", "unknown_player_team", "duplicate_code",
                                    "bad_position", "wrong_year", "missing_event", "missing_player", "contradict_team", "contradict_position"])
def test_register_mapping_season_and_selected_identity_must_match(change):
    evidence = _evidence()
    register = evidence[4]
    if change == "missing_team":
        register["teams"].pop()
    elif change == "duplicate_team":
        register["teams"][-1]["name"] = register["teams"][0]["name"]
    elif change == "unknown_player_team":
        register["elements"][0]["team"] = 99
    elif change == "duplicate_code":
        register["elements"][1]["code"] = 101
    elif change == "bad_position":
        register["elements"][0]["element_type"] = 5
    elif change == "wrong_year":
        register["events"][0]["deadline_time"] = "2025-08-16T17:30:00Z"
    elif change == "missing_event":
        register["events"].pop(0)
    elif change == "missing_player":
        register["elements"].pop(0)
    elif change == "contradict_team":
        evidence[1].loc[0, "team"] = "club3"
    else:
        evidence[1].loc[0, "position"] = "FWD"
    with pytest.raises(ValueError):
        _run(evidence)


@pytest.mark.parametrize("kwargs", [{"event": 0}, {"event": 39}, {"event": True}, {"event": 2},
                                    {"first_global_gw": 1}, {"season": "2024-26"}, {"season": "2025-26"}])
def test_wrong_event_or_global_season_mapping_rejected(kwargs):
    with pytest.raises(ValueError):
        _run(_evidence(), **kwargs)


def test_missing_player_at_playing_club_cannot_be_inferred_to_be_blank():
    evidence = _evidence()
    evidence[4]["elements"][0]["team"] = 3
    with pytest.raises(ValueError, match="club with a fixture"):
        _run(evidence)


def test_consistent_repeated_origin_fixture_rows_are_preserved():
    evidence = list(_evidence())
    original = evidence[1]
    evidence[1] = pd.concat([original.assign(origin_gw=152), original.assign(origin_gw=153)], ignore_index=True)
    unchanged, audit = _run(evidence)
    pd.testing.assert_frame_equal(unchanged, evidence[1])
    assert len(audit) == 1


def test_present_blank_zero_gets_no_additional_audit_row():
    evidence = list(_evidence())
    explicit = pd.DataFrame({"player_id": [101], "GW": [153], "team": ["club1"], "position": ["MID"],
                             "actual_total_points": [0.], "minutes": [0.], "fixture": [np.nan]})
    evidence[1] = pd.concat([evidence[1], explicit], ignore_index=True)
    unchanged, audit = _run(evidence)
    pd.testing.assert_frame_equal(unchanged, evidence[1])
    assert audit.empty


def test_explicit_blank_audit_enables_hand_calculated_autosub_replay():
    from fpl.milp.replay import replay_snapshot

    _, _, fixtures, teams, register = _evidence()
    roles = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    role_ids = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    elements, outcomes = [], []
    for code, position in enumerate(roles, 1):
        club = 1 if code == 3 else code % 5 + 2
        elements.append({"code": code, "team": club, "element_type": role_ids[position]})
        if code == 3:
            continue
        current = fixtures.loc[(fixtures.event == 1) & ((fixtures.team_h == club) | (fixtures.team_a == club))]
        points = {6: 6., 13: 2., 14: 3.}.get(code, 1.)
        outcomes.append({"player_id": code, "GW": 153, "team": f"club{club}", "position": position,
                         "actual_total_points": points, "minutes": 90., "fixture": int(current.id.iloc[0])})
    register["elements"] = elements
    snapshot = {"schema_version": 1, "settlement_completed": False, "decisions": [{
        "gameweek": 153, "squad": list(range(1, 16)),
        "lineup": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
        "bench_order": [2, 6, 7, 12], "captain": [13], "vice_captain": [14],
        "chip_played": None, "alpha": 1,
    }]}
    predictions = pd.DataFrame(outcomes)
    with pytest.raises(ValueError, match="missing selected-player"):
        replay_snapshot(snapshot, predictions)
    original, audit = explicit_blank_outcomes(snapshot, predictions, fixtures=fixtures, teams=teams,
        register=register, season="2024-25", first_global_gw=153, event=1)
    assert audit.player_id.tolist() == [3]
    settled = replay_snapshot(snapshot, original, blank_outcomes=audit)
    # Ten playing starters score13; reserve DEF6 adds6; captain13 adds2; one hit costs4.
    assert settled.iloc[0].actual_total_points == 21
    assert settled.iloc[0].actual_net_points == 17
    assert settled.iloc[0].missing_outcome_players == 0
    assert "fixture" not in audit and "predicted_total_points" not in audit
    pd.testing.assert_frame_equal(original, predictions)


@pytest.mark.parametrize("change", ["wrong_year", "before_first_deadline", "before_target_deadline"])
def test_fixture_calendar_must_match_season_and_registered_deadlines(change):
    evidence = _evidence()
    event = 1
    if change == "wrong_year":
        evidence[2]["kickoff_time"] = "2023-08-17T14:00:00Z"
    elif change == "before_first_deadline":
        evidence[2].loc[0, "kickoff_time"] = "2024-08-15T14:00:00Z"
    else:
        event = 2
        evidence[4]["events"][1]["deadline_time"] = "2024-08-24T20:00:00Z"
    with pytest.raises(ValueError, match="kickoff calendar/deadlines"):
        _run(evidence, event=event)
