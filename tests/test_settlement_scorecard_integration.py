"""Report plumbing; the adapter's real receipt admission is tested separately."""
from copy import deepcopy
import json

import pytest

from fpl import playground
from fpl.artifacts import sha256_file
from fpl.scorecard import build_scorecard, render_scorecard


def _spec(tmp_path, monkeypatch):
    path = playground.init_demo(tmp_path / 'demo')
    spec = json.loads(path.read_text())
    spec['settlement_replays'] = {}
    sides = {}
    for side in ('candidate', 'control'):
        root = path.parent / side
        root.mkdir()
        receipt = root / 'complete.json'
        receipt.write_text(json.dumps({'side': side}))
        spec['settlement_replays'][side] = {'path': str(receipt), 'sha256': sha256_file(receipt)}
        sides[side] = {'receipt_sha256':sha256_file(receipt),
            'snapshot_sha256':'d'*64, 'prediction_sha256':'e'*64,
            'comparison_input_sha256':spec[side]['input']['sha256'],
            'gross_points':179 if side=='candidate' else 170,
            'net_points':171 if side=='candidate' else 170,
            'hit_points':8 if side=='candidate' else 0, 'decision_rounds':3,
            'transfers_in':17, 'transfers_out':2, 'chip_usage':{},
            'recorded':{'solver_status':{'status':'RECORDED','values':['Optimal']},
                'mip_gap':{'status':'RECORDED','values':[0]},
                'solver_seconds':{'status':'RECORDED','values':[1.25]}}}
    summary = {'schema_version':1, 'status':'VERIFIED_SETTLEMENT_INPUTS',
        'metric':spec['metric'], 'gameweeks':[153,154,155], 'sides':sides, 'promotion_eligible':False,
        'checks':{key:'NOT ASSESSED' for key in ('price_information_validity','forecast_blank_planning',
            'fixture_causality','full_outcome_blind_execution','native_certificate','independent_legality')}}
    def adapter(candidate, control, **kwargs):
        assert candidate == (path.parent/'candidate').resolve()
        assert control == (path.parent/'control').resolve()
        assert kwargs == {'candidate_sha256':spec['candidate']['input']['sha256'],
            'control_sha256':spec['control']['input']['sha256'], 'gameweeks':[153,154,155], 'metric':spec['metric']}
        return deepcopy(summary)
    monkeypatch.setattr(playground, 'summarize_replay_pair', adapter)
    path.write_text(json.dumps(spec))
    return path, summary


def test_report_binds_captured_receipts_and_measured_values_without_upgrading_gates(tmp_path, monkeypatch):
    path, summary = _spec(tmp_path, monkeypatch)
    directory = playground.run_experiment(path, runs_dir=tmp_path/'runs')
    card = playground.read_run(directory)
    assert card['settlement_evidence'] == summary
    assert card['verdict']['promotion_eligible'] is False
    assert card['operations']['failure_rate'] == 'NOT MEASURED'
    report = (directory/'REPORT.md').read_text()
    assert report == render_scorecard(card)
    assert '| Gross points | 179 | 170 |' in report
    assert '| Transfer hit points | 8 | 0 |' in report
    assert '| price information validity | NOT ASSESSED |' in report
    assert 'zero-filled planning forecasts' in report
    spec = json.loads(path.read_text())
    for side in ('candidate','control'):
        copied = directory/f'{side}-settlement-receipt.json'
        assert sha256_file(copied) == spec['settlement_replays'][side]['sha256']
        (path.parent/side/'complete.json').unlink()
    assert playground.read_run(directory) == card
    (directory/'candidate-settlement-receipt.json').write_text('corrupted')
    with pytest.raises(ValueError,match='changed'):
        playground.read_run(directory)


@pytest.mark.parametrize('problem',['one_side','unpinned','wrong_name','mismatched_summary'])
def test_settlement_inputs_fail_before_output_creation(tmp_path, monkeypatch, problem):
    path, _ = _spec(tmp_path, monkeypatch)
    spec = json.loads(path.read_text())
    if problem=='one_side': spec['settlement_replays'].pop('control')
    elif problem=='unpinned': spec['settlement_replays']['candidate']['sha256'] = '0'*64
    elif problem=='wrong_name':
        original = path.parent/'candidate/complete.json'
        replacement = original.with_name('result.json')
        original.rename(replacement)
        spec['settlement_replays']['candidate']['path'] = str(replacement)
    else:
        spec['settlement_replays']['candidate']['sha256'] = 'a'*64
        monkeypatch.setattr(playground,'_pinned',lambda record,base:(
            (base/record['path']).resolve(), (base/record['path']).read_bytes()))
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError): playground.run_experiment(path,runs_dir=tmp_path/'runs')
    assert not (tmp_path/'runs').exists()


def test_direct_scorecard_cannot_accept_unlinked_evidence(tmp_path, monkeypatch):
    path, summary = _spec(tmp_path,monkeypatch)
    directory = playground.run_experiment(path,runs_dir=tmp_path/'runs')
    card = playground.read_run(directory)
    spec = json.loads(path.read_text())
    with pytest.raises(ValueError,match='validated evidence'):
        build_scorecard(spec,card['decision_outcomes'],reporting_seconds=0)
    summary['sides']['candidate']['comparison_input_sha256'] = '0'*64
    with pytest.raises(ValueError,match='bytes'):
        build_scorecard(spec,card['decision_outcomes'],reporting_seconds=0,settlement_evidence=summary)


def test_recorded_solver_claims_cannot_upgrade_evidence_gates(tmp_path, monkeypatch):
    path, summary = _spec(tmp_path, monkeypatch)
    directory = playground.run_experiment(path, runs_dir=tmp_path/'runs')
    card = playground.read_run(directory)
    spec = json.loads(path.read_text())
    summary['checks']['native_certificate'] = 'PASS'
    with pytest.raises(ValueError, match='identity'):
        build_scorecard(spec, card['decision_outcomes'], reporting_seconds=0, settlement_evidence=summary)
