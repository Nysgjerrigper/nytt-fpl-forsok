"""Operational imports are immutable, offline-readable and never ranked outcomes."""
import json
from pathlib import Path

import pandas as pd
import pytest

from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.solio_receipts import ScheduleReceipts
from fpl.operational_reports import import_operations, read_operations, render_operations
from fpl.playground import league, main, render_league


def source(tmp_path, sealed=True):
    directory = tmp_path / 'source'
    receipt = ScheduleReceipts(directory, pd.DataFrame({'GW': [137]}), {'start_gw': 137, 'max_gw': 138})
    receipt.emit('origin-start', {'origin_gw': 137, 'state': {}, 'horizon': 2})
    receipt.emit('origin-admitted', {'admitted_frame': {}, 'admission_seconds': .2})
    receipt.emit('prepared', {'prepared_inputs': {}, 'options': {}, 'preparation_seconds': .1})
    receipt.emit('vendor-start', {'time_limit_seconds': 120, 'relative_gap': 0})
    if sealed:
        receipt.emit('vendor-failed', {'vendor_wall_seconds': 120.1, 'native_seconds': 'NOT MEASURED',
                                       'build_seconds': 'NOT MEASURED'})
        receipt.terminal(RuntimeError('time limit'))
    spec = {'schema_version': 1, 'id': 'failed-candidate', 'name': 'Candidate captured-pool run',
            'hypothesis': 'Evaluate operational completion; no point comparison.',
            'source': {'kind': 'solio_schedule', 'path': str(directory),
                       'terminal_sha256': sha256_file(directory / 'failed.json') if sealed else None}}
    path = tmp_path / 'spec.json'
    write_json_new(path, spec)
    return path, directory


def test_import_report_reuse_and_unranked_league(tmp_path, capsys):
    spec, original = source(tmp_path)
    output = import_operations(spec, runs_dir=tmp_path / 'runs')
    card = read_operations(output)
    assert card['operations']['status'] == 'FAILED'
    assert card['operations']['counts']['attempted_vendor_calls'] == 1
    assert card['performance'] == {'rank': None, 'points': None, 'delta': None, 'ci': None}
    assert card['promotion_eligible'] is False
    assert (output / 'REPORT.md').read_text() == render_operations(card)
    assert import_operations(spec, runs_dir=tmp_path / 'runs') == output
    groups = league(tmp_path / 'runs')
    row = groups[0]['entries'][0]
    assert row['rank'] is row['net_or_gross_points'] is row['delta'] is row['ci'] is None
    assert row['operations'] == card['operations']
    assert 'FAILED' in render_league(groups)
    assert 'attempted_vendor_calls=1' in render_league(groups)
    # Reports depend only on the captured evidence, not the original filesystem.
    original.rename(tmp_path / 'moved-source')
    main(['operations-report', str(output)])
    assert 'Unranked' in capsys.readouterr().out
    assert read_operations(output) == card


def test_interrupted_snapshot_remains_unknown_and_distinct_attempt(tmp_path):
    spec, _ = source(tmp_path, sealed=False)
    root = tmp_path / 'runs'
    output = import_operations(spec, runs_dir=root)
    card = read_operations(output)
    assert card['operations']['status'] == 'INCOMPLETE'
    assert card['operations']['counts']['unknown_in_flight_calls'] == 1
    assert card['operations']['counts']['failed_vendor_calls'] == 0
    assert league(root)[0]['entries'][0]['rank'] is None
    second = import_operations(spec, runs_dir=root, attempt='observation-2')
    assert second != output


@pytest.mark.parametrize('target', ['REPORT.md', 'operational-card.json', 'evidence/started.json', 'spec.json'])
def test_changed_saved_evidence_or_report_rejected(tmp_path, target):
    spec, _ = source(tmp_path)
    output = import_operations(spec, runs_dir=tmp_path / 'runs')
    path = output / target
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError): read_operations(output)
    with pytest.raises(ValueError): league(tmp_path / 'runs')


def test_rehashed_fabricated_card_still_rejected(tmp_path):
    spec, _ = source(tmp_path)
    output = import_operations(spec, runs_dir=tmp_path / 'runs')
    path = output / 'operational-card.json'
    card = json.loads(path.read_text())
    card['performance']['points'] = 999
    path.write_text(json.dumps(card))
    receipt = json.loads((output / 'operational-complete.json').read_text())
    receipt['outputs']['operational-card.json'] = sha256_file(path)
    (output / 'operational-complete.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='card differs'): read_operations(output)


def test_pin_required_and_registration_bytes_bound(tmp_path):
    spec_path, _ = source(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec['source']['terminal_sha256'] = None
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError): import_operations(spec_path, runs_dir=tmp_path / 'runs')
    spec['source']['terminal_sha256'] = sha256_file(tmp_path / 'source/failed.json')
    registration = tmp_path / 'plan.json'
    registration.write_text('{"purpose":"synthetic"}')
    spec['registration'] = {'path': str(registration), 'sha256': sha256_file(registration)}
    spec_path.write_text(json.dumps(spec))
    output = import_operations(spec_path, runs_dir=tmp_path / 'runs')
    registration.unlink()
    assert read_operations(output)['registration']['sha256'] == spec['registration']['sha256']


def test_forbidden_outputs_and_reuse_identity(tmp_path):
    spec, _ = source(tmp_path)
    from fpl import config
    with pytest.raises(ValueError): import_operations(spec, runs_dir=config.ROOT / 'fpl/unsafe')
    output = import_operations(spec, runs_dir=tmp_path / 'runs')
    (output / 'complete.json').write_text('{}')
    with pytest.raises(ValueError): read_operations(output)
    with pytest.raises(ValueError): import_operations(spec, runs_dir=tmp_path / 'runs')


def test_cli_import_and_json_league(tmp_path, capsys):
    spec, _ = source(tmp_path)
    main(['import-operations', str(spec), '--runs-dir', str(tmp_path / 'runs')])
    assert 'failed-candidate' in capsys.readouterr().out
    main(['league', '--runs-dir', str(tmp_path / 'runs'), '--json'])
    groups = json.loads(capsys.readouterr().out)
    assert groups[0]['entries'][0]['status'] == 'FAILED'
    assert groups[0]['entries'][0]['net_or_gross_points'] is None


def test_failed_import_preserved_without_claiming_solver_failure(tmp_path, monkeypatch):
    spec, _ = source(tmp_path)
    from fpl import operational_reports
    def broken_report(card):
        raise ValueError('reporting interrupted')
    monkeypatch.setattr(operational_reports, 'render_operations', broken_report)
    with pytest.raises(ValueError, match='reporting interrupted'):
        import_operations(spec, runs_dir=tmp_path / 'runs')
    output = tmp_path / 'runs/failed-candidate/first'
    assert (output / 'operational-failed.json').exists()
    assert not (output / 'operational-complete.json').exists()
    row = league(tmp_path / 'runs')[0]['entries'][0]
    assert row['status'] == 'IMPORT_FAILED' and row['rank'] is None
    assert 'operations' not in row  # No telemetry is displayed from an unfinished import.
    with pytest.raises(FileExistsError): import_operations(spec, runs_dir=tmp_path / 'runs')


def test_source_and_destination_symlinks_are_refused(tmp_path):
    spec_path, original = source(tmp_path)
    linked = tmp_path / 'linked-source'
    linked.symlink_to(original, target_is_directory=True)
    spec = json.loads(spec_path.read_text())
    spec['source']['path'] = str(linked)
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError): import_operations(spec_path, runs_dir=tmp_path / 'runs')
    spec['source']['path'] = str(original)
    spec_path.write_text(json.dumps(spec))
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'failed-candidate').symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError): import_operations(spec_path, runs_dir=runs)
