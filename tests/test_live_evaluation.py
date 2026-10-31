"""Live captures must retain provenance and never turn missing outcomes into zeros."""
import pandas as pd
import pytest

from fpl.live_evaluation import capture, evaluate


def bootstrap(checked=True):
    return {'events': [{'id': 1, 'deadline_time': '2026-08-01T12:00:00Z'},
                       {'id': 8, 'deadline_time': '2026-10-23T12:00:00Z', 'data_checked': checked}]}


def forecasts():
    return pd.DataFrame({'player_id': [11, 11, 22], 'GW': [8, 8, 8],
                         'position': ['MID'] * 3, 'opponent_team': ['A', 'B', 'A'],
                         'was_home': [True, False, True], 'predicted_total_points': [2., 3., 4.]})


def outcomes():
    return pd.DataFrame({'player_code': [11, 11, 11], 'GW': [8, 8, 8],
                         'season': ['2026-27', '2026-27', '2025-26'], 'total_points': [3., 4., 99.]})


def test_dgw_partial_coverage_and_season_isolation(tmp_path):
    folder = capture(forecasts(), bootstrap(), strategy='single:catboost', recovered=True, root=tmp_path)
    report = evaluate(folder, outcomes(), bootstrap())
    row = report['rounds'][0]
    assert row['status'] == 'partial'
    assert row['missing_player_ids'] == [22]
    assert row['diagnostics']['overall']['mae'] == 2.
    assert row['diagnostics']['overall']['bias'] == -2.
    assert report['metadata']['deadline_evidence'] == 'unknown'


def test_pending_does_not_score_even_if_rows_exist(tmp_path):
    folder = capture(forecasts(), bootstrap(), strategy='single:catboost', root=tmp_path)
    assert evaluate(folder, outcomes(), bootstrap(False))['rounds'][0]['status'] == 'pending'
    (folder / 'predictions.csv').write_text('modified')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluate(folder, outcomes(), bootstrap())


def test_complete_coverage_and_immutable_runs(tmp_path):
    frame = forecasts().iloc[:2]
    a = capture(frame, bootstrap(), strategy='single:catboost', root=tmp_path)
    b = capture(frame, bootstrap(), strategy='single:catboost', root=tmp_path)
    assert a != b
    assert evaluate(a, outcomes(), bootstrap())['rounds'][0]['status'] == 'complete'
    other = bootstrap()
    other['events'][0]['deadline_time'] = '2025-08-01T12:00:00Z'
    with pytest.raises(ValueError, match='season'):
        evaluate(a, outcomes(), other)


def test_ambiguous_fixture_rows_rejected(tmp_path):
    frame = pd.concat([forecasts(), forecasts().iloc[:1]], ignore_index=True)
    folder = capture(frame, bootstrap(), strategy='single:catboost', root=tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        evaluate(folder, outcomes(), bootstrap())


def test_weekly_monitor_preserves_reports(tmp_path, monkeypatch):
    from fpl import config
    from fpl.live_evaluation import record_evaluation

    monkeypatch.setattr(config, 'LIVE_FORECAST_DIR', tmp_path)
    assert record_evaluation(outcomes(), bootstrap()) is None
    capture(forecasts(), bootstrap(), strategy='single:catboost', root=tmp_path)
    a = record_evaluation(outcomes(), bootstrap())
    b = record_evaluation(outcomes(), bootstrap())
    assert a != b and a.exists() and b.exists()
