"""Receipt consistency tests without solver/model imports or execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from fpl.operational_evidence import inspect_evidence


STEPS = ['origin-start', 'origin-admitted', 'prepared', 'vendor-start', 'vendor-complete',
         'normalization-start', 'normalization-complete', 'state-update-start', 'decision']


def put(path, value):
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    return {'source_sha256': {'fpl/old-source.py': 'a'*64}, 'runtime': {'python': 'historical-1', 'optional': None}}


def rehash(root, terminal):
    receipt = read(root / terminal)
    key = 'file_sha256' if terminal == 'diagnostics-complete.json' else 'receipt_sha256'
    receipt[key] = {path.name: sha(path) for path in root.iterdir() if path.is_file() and path.name != terminal}
    put(root / terminal, receipt)


def inspect(root, kind='solio_schedule', terminal='complete.json'):
    return inspect_evidence(root, kind=kind, terminal_sha256=sha(root / terminal) if terminal else None)


def schedule(root, *, events=18, terminal='complete.json', vendor_failure=False, scoring=False):
    root.mkdir()
    put(root / 'started.json', {'schema_version': 1, 'status': 'STARTED', **provenance(),
         'call_count_scope': 'VENDOR_WRAPPER_INVOCATIONS', 'settings': {'start_gw': 137, 'max_gw': 138}})
    identity = sha(root / 'started.json')
    attempted = returned = saved = 0
    origin, last = None, 'schedule-validation'
    records = []
    for index in range(events):
        origin = 137 + index // 9
        event = STEPS[index % 9]
        if vendor_failure and event == 'vendor-complete': event = 'vendor-failed'
        attempted += event == 'vendor-start'
        returned += event == 'vendor-complete'
        saved += event == 'decision'
        row = {'schema_version': 1, 'event': event, 'sequence': index+1, 'origin_gw': origin,
               'started_sha256': identity, 'attempted_vendor_calls': attempted, 'completed_vendor_calls': returned}
        if event == 'origin-start': row['state'] = {}
        elif event == 'origin-admitted': row['admission_seconds'] = 1.
        elif event == 'prepared': row.update(preparation_seconds=2., prepared_inputs={}, options={'secs': 120})
        elif event == 'vendor-start': row['time_limit_seconds'] = 120
        elif event in ('vendor-complete', 'vendor-failed'):
            row.update(vendor_wall_seconds=4., native_seconds=3. if event == 'vendor-complete' else 'NOT MEASURED',
                       non_native_overhead_seconds=1. if event == 'vendor-complete' else 'NOT MEASURED')
        elif event == 'normalization-complete': row['normalization_seconds'] = .5
        elif event == 'decision':
            row.update(decision={'gameweek': origin}, settlement_completed=False, next_state={}, state_update_seconds=.25)
        put(root / f'{index+1:04d}-{event}.json', row)
        records.append(row)
        last = event
    if scoring:
        row = {'schema_version': 1, 'event': 'schedule-scoring-start', 'sequence': events+1, 'origin_gw': origin,
               'started_sha256': identity, 'attempted_vendor_calls': attempted, 'completed_vendor_calls': returned,
               'completed_origins': saved}
        put(root / f'{events+1:04d}-schedule-scoring-start.json', row)
        last = row['event']
    if terminal:
        receipt = {'schema_version': 1, 'status': 'FAILED' if terminal == 'failed.json' else 'COMPLETED',
                   'started_sha256': identity, 'attempted_vendor_calls': attempted, 'completed_vendor_calls': returned,
                   'completed_origins': saved, 'requested_origins': [137, 138],
                   'call_count_scope': 'VENDOR_WRAPPER_INVOCATIONS'}
        if terminal == 'failed.json': receipt.update(failed_origin=origin, failed_stage=last,
                            exception_type='RuntimeError', exception='recorded wrapper rejection',
                            failed_native_seconds='NOT MEASURED', failed_build_seconds='NOT MEASURED')
        put(root / terminal, receipt)
        rehash(root, terminal)
    return root


def native(root, *, accepted=False, sealed=True, outcome='native-finish'):
    root.mkdir()
    put(root / 'started.json', {'schema_version': 1, 'status': 'DIAGNOSTICS_STARTED', **provenance(),
                              'context': {'start_gw': 139}, 'highs_version': 'old-version'})
    identity = sha(root / 'started.json')
    prefix = '0001-native-start-0'
    put(root / f'{prefix}-model.json', {'schema_version': 1,
        'lp': {'num_col': 0, 'num_row': 0, 'col_cost': [], 'col_lower': [], 'col_upper': [], 'row_lower': [], 'row_upper': []},
        'matrix': {'num_col': 0, 'num_row': 0, 'start': [0], 'p_end': [], 'index': [], 'value': []}})
    (root / f'{prefix}-model.mps').write_text('NAME TINY\nENDATA\n')
    put(root / f'{prefix}-event.json', {'schema_version': 1, 'stage': 'native-start', 'sequence': 1, 'iteration': 0,
        'started_sha256': identity, 'artifact_sha256': {f'{prefix}-model.json': sha(root / f'{prefix}-model.json'),
            f'{prefix}-model.mps': sha(root / f'{prefix}-model.mps')},
        'effective_options': {'time_limit': {'status': 0, 'value': 120}, 'mip_rel_gap': {'status': 0, 'value': 0.}},
        'model_authority': 'LOSSLESS_JSON', 'mps_write_status': 0, 'mps_interpretation': 'REVIEW_ONLY_GENERATED_NAMES_POSSIBLE', 'caller_timings': {'model_build_seconds': 2.}})
    if outcome:
        prefix = f'0002-{outcome}-0'
        timing = {'native_call_wall_seconds': 10.}
        put(root / f'{prefix}-native-info.json', {'status_name': 'Optimal', 'status_value': 7,
            'info_valid': True, 'primal_value_valid': True, 'dual_value_valid': False,
            'values': {'mip_gap': 0., 'objective_function_value': 4.},
            'availability': {'mip_gap': 'RECORDED', 'objective_function_value': 'RECORDED'},
            'native_runtime_seconds': 9., 'caller_timings': timing})
        put(root / f'{prefix}-event.json', {'schema_version': 1, 'stage': outcome, 'sequence': 2, 'iteration': 0,
            'started_sha256': identity, 'artifact_sha256': {f'{prefix}-native-info.json': sha(root / f'{prefix}-native-info.json')},
            'caller_timings': timing})
    if sealed:
        put(root / 'diagnostics-complete.json', {'schema_version': 1, 'status': 'DIAGNOSTICS_RECORDED',
            'started_sha256': identity, 'solver_result_accepted': accepted,
            'error': None if accepted else {'type': 'RuntimeError', 'message': 'normalization rejected'}})
        rehash(root, 'diagnostics-complete.json')
    return root


def repin_artifact(root, name):
    for path in root.glob('*-event.json'):
        row = read(path)
        if name in row['artifact_sha256']:
            row['artifact_sha256'][name] = sha(root / name)
            put(path, row)
    rehash(root, 'diagnostics-complete.json')


def test_completed_schedule_and_directory_independent_copy(tmp_path):
    root = schedule(tmp_path / 'original', scoring=True)
    card = inspect(root)
    assert card['status'] == 'COMPLETED'
    assert card['completed_origins'] == [137, 138]
    assert card['counts'] == {'attempted_vendor_calls': 2, 'returned_vendor_calls': 2,
         'failed_vendor_calls': 0, 'unknown_in_flight_calls': 0, 'saved_origins': 2}
    assert card['timings']['preparation_seconds'] == {'count': 2, 'total': 4., 'observations': 2, 'not_measured': 0}
    assert 'time_limit_seconds' not in card['timings']
    assert card['promotion_eligible'] is False
    assert card['assessment'] == 'RECORDED_OPERATIONAL_EVIDENCE_ONLY'
    assert card['provenance']['runtime']['python'] == 'historical-1'
    shutil.copytree(root, tmp_path / 'copied')
    assert inspect(tmp_path / 'copied') == card


@pytest.mark.parametrize('events', range(19))
def test_every_schedule_prefix_is_incomplete_without_terminal(tmp_path, events):
    card = inspect(schedule(tmp_path / 'prefix', events=events, terminal=None), terminal=None)
    assert card['status'] == 'INCOMPLETE'
    assert card['counts']['saved_origins'] == events // 9
    assert card['counts']['unknown_in_flight_calls'] == int(events % 9 == 4)
    assert card['failed_origin'] is None


def test_vendor_failure_and_failed_prefix_counts(tmp_path):
    root = schedule(tmp_path / 'failure', events=5, vendor_failure=True, terminal='failed.json')
    card = inspect(root, terminal='failed.json')
    assert card['status'] == 'FAILED' and card['failed_origin'] == 137
    assert card['failed_stage'] == 'vendor-failed'
    assert card['counts']['attempted_vendor_calls'] == card['counts']['failed_vendor_calls'] == 1
    assert card['counts']['returned_vendor_calls'] == card['counts']['unknown_in_flight_calls'] == 0
    assert card['timings']['native_seconds']['total'] is None
    assert card['timings']['native_seconds']['not_measured'] == 1


@pytest.mark.parametrize('events', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18])
def test_sealed_failure_can_record_any_valid_prefix(tmp_path, events):
    root = schedule(tmp_path / 'failure', events=events, terminal='failed.json')
    assert inspect(root, terminal='failed.json')['status'] == 'FAILED'


@pytest.mark.parametrize('change', ['counter', 'sequence', 'origin', 'transition', 'decision_gw', 'started_hash', 'timing', 'settled'])
def test_rehashed_schedule_structure_corruption_rejected(tmp_path, change):
    root = schedule(tmp_path / 'bad')
    name = '0009-decision.json' if change in ('decision_gw', 'settled') else '0003-prepared.json'
    row = read(root / name)
    if change == 'counter': row['attempted_vendor_calls'] = 1
    elif change == 'sequence': row['sequence'] = 99
    elif change == 'origin': row['origin_gw'] = 138
    elif change == 'transition': row['event'] = 'vendor-start'
    elif change == 'decision_gw': row['decision']['gameweek'] = 138
    elif change == 'started_hash': row['started_sha256'] = '0'*64
    elif change == 'timing': row['preparation_seconds'] = -1
    else: row['settlement_completed'] = True
    put(root / name, row)
    rehash(root, 'complete.json')
    with pytest.raises(ValueError): inspect(root)


@pytest.mark.parametrize('field,value', [('completed_origins', 1), ('requested_origins', [137]),
                                       ('attempted_vendor_calls', True), ('started_sha256', '0'*64)])
def test_terminal_claims_rejected_even_with_correct_pin(tmp_path, field, value):
    root = schedule(tmp_path / 'bad')
    receipt = read(root / 'complete.json'); receipt[field] = value; put(root / 'complete.json', receipt)
    with pytest.raises(ValueError): inspect(root)


def test_early_false_complete_and_event_after_vendor_failure_rejected(tmp_path):
    root = schedule(tmp_path / 'early', events=1)
    with pytest.raises(ValueError): inspect(root)
    root = schedule(tmp_path / 'failed', events=9, vendor_failure=True, terminal='failed.json')
    with pytest.raises(ValueError): inspect(root, terminal='failed.json')


@pytest.mark.parametrize('change', ['missing_pin', 'wrong_pin', 'missing', 'extra', 'unsafe_path', 'conflict', 'duplicate_json', 'nonfinite'])
def test_pin_file_coverage_and_strict_json(tmp_path, change):
    root = schedule(tmp_path / 'bad')
    pin = sha(root / 'complete.json')
    if change == 'missing_pin': pin = None
    elif change == 'wrong_pin': pin = '0'*64
    elif change == 'missing': (root / '0001-origin-start.json').unlink()
    elif change == 'extra': put(root / 'extra.json', {'schema_version': 1})
    elif change == 'unsafe_path':
        receipt = read(root / 'complete.json'); receipt['receipt_sha256']['../escape'] = '0'*64
        put(root / 'complete.json', receipt); pin = sha(root / 'complete.json')
    elif change == 'conflict': shutil.copy(root / 'complete.json', root / 'failed.json')
    elif change == 'duplicate_json': (root / 'started.json').write_text('{"schema_version":1,"schema_version":1}')
    else: (root / 'started.json').write_text('{"schema_version":1,"bad":1e400}')
    with pytest.raises(ValueError): inspect_evidence(root, kind='solio_schedule', terminal_sha256=pin)


@pytest.mark.parametrize('location', ['root', 'file', 'directory', 'fifo'])
def test_symlinks_and_nonregular_files_rejected(tmp_path, location):
    root = schedule(tmp_path / 'bad')
    target = root
    if location == 'root':
        target = tmp_path / 'link'; target.symlink_to(root, target_is_directory=True)
    elif location == 'file': (root / 'link').symlink_to(root / 'started.json')
    elif location == 'directory': (root / 'linked-directory').symlink_to(tmp_path, target_is_directory=True)
    else: os.mkfifo(root / 'pipe')
    with pytest.raises(ValueError): inspect_evidence(target, kind='solio_schedule', terminal_sha256=None)


def test_native_optimal_rejected_is_diagnostic_completion_only(tmp_path):
    root = native(tmp_path / 'native')
    card = inspect(root, 'native_diagnostics', 'diagnostics-complete.json')
    assert card['status'] == 'DIAGNOSTICS_RECORDED' and card['solver_result_accepted'] is False
    assert card['native_info']['status_name'] == 'Optimal'
    assert card['independent_legality'] == 'NOT ASSESSED'
    assert card['counts'] == {'native_started': 1, 'native_finished': 1, 'native_error': 0, 'native_run_unknown': 0}
    assert card['timings']['model_build_seconds']['total'] == 2.
    assert card['timings']['native_call_wall_seconds']['count'] == 1
    assert card['timings']['native_runtime_seconds']['total'] == 9.
    assert card['effective_options']['time_limit']['value'] == 120


@pytest.mark.parametrize('outcome', [None, 'native-finish', 'native-error'])
def test_native_unsealed_observation_never_infers_acceptance(tmp_path, outcome):
    root = native(tmp_path / 'native', sealed=False, outcome=outcome)
    card = inspect(root, 'native_diagnostics', None)
    assert card['status'] == 'INCOMPLETE' and card['solver_result_accepted'] is None
    assert card['counts']['native_run_unknown'] == int(outcome is None)


@pytest.mark.parametrize('field,value', [('status_name', 'Time limit reached'), ('status_value', 13), ('info_valid', False),
    ('primal_value_valid', False), ('mip_gap', .001), ('mip_gap', None)])
def test_accepted_native_claim_requires_qualified_fields(tmp_path, field, value):
    root = native(tmp_path / 'bad', accepted=True)
    path = root / '0002-native-finish-0-native-info.json'
    info = read(path)
    if field == 'mip_gap':
        info['values'][field] = value
        if value is None: info['availability'][field] = 'NOT AVAILABLE'
    else: info[field] = value
    put(path, info); repin_artifact(root, path.name)
    with pytest.raises(ValueError): inspect(root, 'native_diagnostics', 'diagnostics-complete.json')


def test_accepted_native_finish_with_valid_fields(tmp_path):
    root = native(tmp_path / 'native', accepted=True)
    assert inspect(root, 'native_diagnostics', 'diagnostics-complete.json')['solver_result_accepted'] is True


@pytest.mark.parametrize('change', ['artifact_hash', 'extra_reference', 'iteration', 'sequence', 'started_hash', 'model_schema', 'timing'])
def test_rehashed_native_structural_corruption(tmp_path, change):
    root = native(tmp_path / 'native')
    path = root / '0001-native-start-0-event.json'
    event = read(path)
    if change == 'artifact_hash': event['artifact_sha256']['0001-native-start-0-model.json'] = '0'*64
    elif change == 'extra_reference': event['artifact_sha256']['started.json'] = sha(root / 'started.json')
    elif change == 'iteration': event['iteration'] = 1
    elif change == 'sequence': event['sequence'] = 2
    elif change == 'started_hash': event['started_sha256'] = '0'*64
    elif change == 'timing': event['caller_timings']['model_build_seconds'] = -1
    else:
        modelpath = root / '0001-native-start-0-model.json'
        put(modelpath, {'schema_version': 1, 'lp': {}, 'matrix': {}})
        event['artifact_sha256'][modelpath.name] = sha(modelpath)
    put(path, event); rehash(root, 'diagnostics-complete.json')
    with pytest.raises(ValueError): inspect(root, 'native_diagnostics', 'diagnostics-complete.json')


def test_native_error_excludes_acceptance_and_conflicting_outcomes(tmp_path):
    root = native(tmp_path / 'error', outcome='native-error', accepted=True)
    with pytest.raises(ValueError): inspect(root, 'native_diagnostics', 'diagnostics-complete.json')
    root = native(tmp_path / 'extra')
    put(root / '0003-native-error-0-event.json', {'schema_version': 1, 'stage': 'native-error', 'sequence': 3,
          'iteration': 0, 'started_sha256': sha(root / 'started.json'), 'artifact_sha256': {}})
    rehash(root, 'diagnostics-complete.json')
    with pytest.raises(ValueError): inspect(root, 'native_diagnostics', 'diagnostics-complete.json')


def test_pin_for_incomplete_and_missing_provenance_rejected(tmp_path):
    root = schedule(tmp_path / 'prefix', events=1, terminal=None)
    with pytest.raises(ValueError): inspect_evidence(root, kind='solio_schedule', terminal_sha256='a'*64)
    started = read(root / 'started.json'); started['runtime'] = {}; put(root / 'started.json', started)
    with pytest.raises(ValueError): inspect(root, terminal=None)


def test_fabricated_decision_immediately_after_vendor_failure_rejected(tmp_path):
    root = schedule(tmp_path / 'failure', events=5, vendor_failure=True, terminal='failed.json')
    row = read(root / '0005-vendor-failed.json')
    row.update(event='decision', sequence=6, decision={'gameweek': 137}, settlement_completed=False, next_state={})
    put(root / '0006-decision.json', row)
    receipt = read(root / 'failed.json'); receipt.update(completed_origins=1, failed_stage='decision')
    put(root / 'failed.json', receipt); rehash(root, 'failed.json')
    with pytest.raises(ValueError): inspect(root, terminal='failed.json')


@pytest.mark.parametrize('pending', ['model_only', 'model_mps', 'finish_info', 'error_info'])
def test_native_mid_observe_interruption_keeps_artifacts_uninterpreted(tmp_path, pending):
    root = native(tmp_path / 'pending', sealed=False)
    finish_event = root / '0002-native-finish-0-event.json'
    finish_info = root / '0002-native-finish-0-native-info.json'
    finish_event.unlink()
    if pending.startswith('model'):
        finish_info.unlink()
        (root / '0001-native-start-0-event.json').unlink()
        if pending == 'model_only': (root / '0001-native-start-0-model.mps').unlink()
    elif pending == 'error_info': finish_info.rename(root / '0002-native-error-0-native-info.json')
    card = inspect(root, 'native_diagnostics', None)
    assert card['status'] == 'INCOMPLETE' and card['solver_result_accepted'] is None
    assert card['native_info'] is None
    assert card['uncommitted_artifacts']
    assert card['counts']['native_finished'] == card['counts']['native_error'] == 0
    assert card['counts']['native_started'] == int(not pending.startswith('model'))


@pytest.mark.parametrize('change', ['opposing_orphans', 'unknown_orphan', 'sealed_orphan'])
def test_pending_native_artifacts_are_strictly_scoped(tmp_path, change):
    root = native(tmp_path / 'pending', sealed=False)
    (root / '0002-native-finish-0-event.json').unlink()
    if change == 'opposing_orphans':
        shutil.copy(root / '0002-native-finish-0-native-info.json', root / '0002-native-error-0-native-info.json')
    elif change == 'unknown_orphan': (root / 'arbitrary.txt').write_text('extra')
    else:
        put(root / 'diagnostics-complete.json', {'schema_version': 1, 'status': 'DIAGNOSTICS_RECORDED',
           'started_sha256': sha(root / 'started.json'), 'solver_result_accepted': False, 'error': None})
        rehash(root, 'diagnostics-complete.json')
    with pytest.raises(ValueError):
        inspect(root, 'native_diagnostics', 'diagnostics-complete.json' if change == 'sealed_orphan' else None)


@pytest.mark.parametrize('change', ['invalid_info', 'invalid_primal', 'negative_gap', 'negative_count'])
def test_native_measured_field_consistency_even_when_rejected(tmp_path, change):
    root = native(tmp_path / 'bad')
    path = root / '0002-native-finish-0-native-info.json'
    info = read(path)
    if change == 'invalid_info': info['info_valid'] = False
    elif change == 'invalid_primal': info['primal_value_valid'] = False
    elif change == 'negative_gap': info['values']['mip_gap'] = -1
    else:
        info['values']['mip_node_count'] = -1
        info['availability']['mip_node_count'] = 'RECORDED'
    put(path, info); repin_artifact(root, path.name)
    with pytest.raises(ValueError): inspect(root, 'native_diagnostics', 'diagnostics-complete.json')


@pytest.mark.parametrize('change', ['empty_mps', 'bad_mps_status', 'bad_option_status', 'bad_option_value', 'gap', 'time'])
def test_native_export_and_effective_option_consistency(tmp_path, change):
    root = native(tmp_path / 'bad', accepted=True)
    path = root / '0001-native-start-0-event.json'
    event = read(path)
    if change == 'empty_mps':
        modelpath = root / '0001-native-start-0-model.mps'; modelpath.write_bytes(b'')
        event['artifact_sha256'][modelpath.name] = sha(modelpath)
    elif change == 'bad_mps_status': event['mps_write_status'] = -1
    elif change == 'bad_option_status': event['effective_options']['time_limit']['status'] = True
    elif change == 'bad_option_value': event['effective_options']['time_limit']['value'] = {'bad': 1}
    elif change == 'gap': event['effective_options']['mip_rel_gap']['value'] = .01
    else: event['effective_options']['time_limit']['value'] = 0
    put(path, event); rehash(root, 'diagnostics-complete.json')
    with pytest.raises(ValueError): inspect(root, 'native_diagnostics', 'diagnostics-complete.json')


def test_nonoptimal_native_outcome_is_valid_rejected_observation(tmp_path):
    root = native(tmp_path / 'native')
    path = root / '0002-native-finish-0-native-info.json'
    info = read(path); info.update(status_name='Time limit reached', status_value=13)
    put(path, info); repin_artifact(root, path.name)
    card = inspect(root, 'native_diagnostics', 'diagnostics-complete.json')
    assert card['status'] == 'DIAGNOSTICS_RECORDED' and card['solver_result_accepted'] is False
    assert card['native_info']['status_name'] == 'Time limit reached'


@pytest.mark.parametrize('pending', ['model_only', 'model_empty_mps', 'finish_info'])
def test_sealed_export_or_event_write_failure_keeps_uncommitted_artifacts(tmp_path, pending):
    root = native(tmp_path / 'export-failure')
    (root / '0002-native-finish-0-event.json').unlink()
    if pending.startswith('model'):
        (root / '0002-native-finish-0-native-info.json').unlink()
        (root / '0001-native-start-0-event.json').unlink()
        mps = root / '0001-native-start-0-model.mps'
        if pending == 'model_only': mps.unlink()
        else: mps.write_bytes(b'')
    receipt = read(root / 'diagnostics-complete.json')
    receipt['error'] = {'type': 'RuntimeError', 'message': 'native diagnostic MPS export failed'}
    put(root / 'diagnostics-complete.json', receipt); rehash(root, 'diagnostics-complete.json')
    card = inspect(root, 'native_diagnostics', 'diagnostics-complete.json')
    assert card['status'] == 'DIAGNOSTICS_RECORDED' and card['solver_result_accepted'] is False
    assert card['uncommitted_artifacts'] and card['native_info'] is None
    assert card['counts']['native_finished'] == card['counts']['native_error'] == 0
    assert card['counts']['native_started'] == int(pending == 'finish_info')
    assert card['recorded_error']['message'] == 'native diagnostic MPS export failed'


def test_recorded_error_and_small_execution_settings(tmp_path):
    root = schedule(tmp_path / 'failed', events=1, terminal='failed.json')
    card = inspect(root, terminal='failed.json')
    assert card['recorded_error'] == {'type': 'RuntimeError', 'message': 'recorded wrapper rejection'}
    assert card['execution_settings'] == {'start_gw': 137, 'max_gw': 138}
    root = native(tmp_path / 'native')
    card = inspect(root, 'native_diagnostics', 'diagnostics-complete.json')
    assert card['execution_settings'] == {'start_gw': 139}
    assert card['recorded_error']['message'] == 'normalization rejected'
    (root / 'diagnostics-complete.json').unlink()
    card = inspect(root, 'native_diagnostics', None)
    assert card['recorded_error'] is None
