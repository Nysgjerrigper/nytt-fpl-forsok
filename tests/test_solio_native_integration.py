"""Native diagnostics preserve real FPL models and never bypass solver rejection."""
import json

import highspy
import pandas as pd
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from fpl.artifacts import sha256_file
from fpl.milp import solio
from fpl.milp.native_diagnostics import deserialize_linear_model, serialize_linear_model


def forecasts():
    roles = ['GK']*2 + ['DEF']*5 + ['MID']*5 + ['FWD']*3
    return pd.DataFrame([{'player_id': p+1, 'GW': 1, 'name': f'p{p+1}',
        'position': roles[p], 'team': f'club{p % 5}', 'value': 50,
        'predicted_total_points': 2} for p in range(15)])


def read(path):
    return json.loads(path.read_text())


def verify_seal(folder):
    result = read(folder / 'diagnostics-complete.json')
    hashes = {p.name: sha256_file(p) for p in folder.iterdir() if p.name != 'diagnostics-complete.json'}
    assert hashes == result['file_sha256']
    assert result['started_sha256'] == hashes['started.json']
    return result


def test_diagnostics_roundtrip_real_fpl_problem_without_changing_output(tmp_path):
    frame = forecasts()
    ordinary = solio.solve_window(frame, season='2020-21', start_gw=1, horizon=1)
    folder = tmp_path / 'native'
    instrumented = solio.solve_window(frame, season='2020-21', start_gw=1, horizon=1,
                                     native_diagnostics_dir=folder)
    columns = [c for c in ordinary if c not in ('solver_seconds', 'window_seconds')]
    pd.testing.assert_frame_equal(ordinary[columns], instrumented[columns])
    seal = verify_seal(folder)
    assert seal['solver_result_accepted'] is True and seal['status'] == 'DIAGNOSTICS_RECORDED'
    payload = read(next(folder.glob('*-model.json')))
    fresh = highspy.Highs()
    fresh.setOptionValue('output_flag', False)
    assert fresh.passModel(deserialize_linear_model(payload)) == highspy.HighsStatus.kOk
    assert serialize_linear_model(deserialize_linear_model(payload)) == payload
    loaded = serialize_linear_model(fresh.getModel())
    assert loaded['lp'] == payload['lp']
    def coefficients(record):
        matrix = record['matrix']
        constructor = csc_matrix if matrix['format'] == int(highspy.MatrixFormat.kColwise) else csr_matrix
        return constructor((matrix['value'], matrix['index'], matrix['start']),
                           shape=(matrix['num_row'], matrix['num_col'])).tocsc()
    assert (coefficients(loaded) != coefficients(payload)).nnz == 0
    assert fresh.run() == highspy.HighsStatus.kOk
    info = read(next(folder.glob('*-native-info.json')))
    assert info['primal_value_valid'] is True
    assert fresh.getObjectiveValue() == pytest.approx(info['values']['objective_function_value'], abs=1e-10)
    assert info['values']['mip_gap'] == 0
    events = [read(p) for p in folder.glob('*-event.json')]
    assert {e['stage'] for e in events} == {'native-start', 'native-finish'}
    start = next(e for e in events if e['stage'] == 'native-start')
    assert start['effective_options']['mip_rel_gap']['value'] == 0
    assert start['caller_timings']['model_build_seconds'] >= 0
    assert info['caller_timings']['native_call_wall_seconds'] >= 0
    assert info['interpretation'] == 'DIAGNOSTIC_ONLY'


def test_timeout_preserves_model_native_status_and_rejected_admission(tmp_path):
    folder = tmp_path / 'rejected'
    with pytest.raises(RuntimeError, match='solve rejected'):
        solio.solve_window(forecasts(), season='2020-21', start_gw=1, horizon=1,
                          time_limit=1e-9, native_diagnostics_dir=folder)
    assert verify_seal(folder)['solver_result_accepted'] is False
    assert next(folder.glob('*-model.json')).exists()
    info = read(next(folder.glob('*-native-info.json')))
    assert info['status_name'] != 'Optimal'
    if not info['primal_value_valid']:
        assert info['values']['objective_function_value'] is None
    assert not (folder / 'decisions.json').exists()
    assert not (folder / 'complete.json').exists()


def test_normalization_rejection_does_not_become_accepted_native_result(tmp_path, monkeypatch):
    def invalid(*args): raise ValueError('independent normalization rejected')
    monkeypatch.setattr(solio, 'normalize_solution', invalid)
    folder = tmp_path / 'invalid-normalization'
    with pytest.raises(ValueError, match='normalization rejected'):
        solio.solve_window(forecasts(), season='2020-21', start_gw=1, horizon=1,
                          native_diagnostics_dir=folder)
    assert verify_seal(folder)['solver_result_accepted'] is False
    assert read(next(folder.glob('*-native-info.json')))['status_name'] == 'Optimal'


def test_existing_native_directory_cannot_be_reused(tmp_path):
    folder = tmp_path / 'existing'
    folder.mkdir()
    events = []
    with pytest.raises(FileExistsError):
        solio.solve_window(forecasts(), season='2020-21', start_gw=1, horizon=1,
                          native_diagnostics_dir=folder,
                          _event_callback=lambda stage, detail: events.append(stage))
    assert events == ["prepared"]
    assert list(folder.iterdir()) == []
