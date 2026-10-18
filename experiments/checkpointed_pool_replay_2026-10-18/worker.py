"""Execute one separately registered preserved-forecast replay; never retry."""
from pathlib import Path
import json
import time
import traceback

import pandas as pd

from fpl import config
from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.origin_prices import load_origin_captures
from fpl.milp.solio import run_schedule
from fpl.milp.optimize import write_decision_snapshot
from fpl.milp.replay import run_replay, read_replay
from fpl.milp.compare_backtests import compare
from fpl.research_runner import _source_hashes, _runtime_versions
from fpl.playground import run_experiment

root = Path(__file__).resolve().parent
plan = json.loads((root / 'plan.json').read_text())
assert sha256_file(root / 'worker.py') == plan['worker_sha256']
assert _source_hashes() == plan['source_sha256']
assert _runtime_versions() == plan['runtime']
old = Path(plan['prior_bundle'])
assert sha256_file(old / 'attempt-seal.json') == plan['prior_attempt_seal_sha256']
prior = json.loads((old / 'attempt-seal.json').read_text())
for name, digest in prior['output_sha256'].items():
    assert sha256_file(old / name) == digest, name
out = root / 'first'
out.mkdir(exist_ok=False)
started = time.monotonic()
write_json_new(out / 'started.json', {'plan_sha256': sha256_file(root / 'plan.json'), 'status': 'STARTED'})
try:
    captures = load_origin_captures(old / 'captures/captures.json', season='2023-24', origins=list(range(137,153)))
    for arm in ['control', 'candidate']:
        raw = pd.read_csv(old / f'{arm}-predictions.csv')
        solver_input = raw.drop(columns=[c for c in raw if c.startswith('actual_') or c == 'minutes'])
        solver_input.to_csv(out / f'{arm}-decision-input.csv', index=False)
        decisions = run_schedule(solver_input, season='2023-24', start_gw=137, max_gw=152,
            horizon=3, bank=1000, free_transfers=1, time_limit=120,
            origin_price_captures=captures, checkpoint_dir=out / f'{arm}-checkpoints')
        assert len(decisions) == 16
        for row in decisions.to_dict('records'):
            assert set(row['squad']) <= set(captures[row['gameweek']]['metadata'].index)
        write_decision_snapshot(decisions, out / f'{arm}-decisions.json')
        decisions.to_csv(out / f'{arm}-decisions.csv', index=False)
        print(arm, 'completed16 decisions', flush=True)
    # No totals are read until both arms have saved pre-settlement snapshots.
    for arm in ['control', 'candidate']:
        target = run_replay(out / f'{arm}-decisions.json', old / f'{arm}-predictions.csv',
            out / f'{arm}-settlement', blank_evidence_path=old / 'settlement-evidence/collection.json')
        read_replay(target)
    result = compare(str(out / 'candidate-settlement/replayed.csv'),
                     str(out / 'control-settlement/replayed.csv'), n_boot=10000, block_len=3, seed=0,
                     points_col='actual_net_points')
    write_json_new(out / 'comparison.json', result)
    pin = lambda p: {'path': str(p), 'sha256': sha256_file(p)}
    spec = {'schema_version':1, 'id':'ownership-capture-120s', 'purpose':'historical_reanalysis',
        'hypothesis': plan['hypothesis'], 'metric':'actual_net_points',
        'bootstrap':{'block_len':3,'n_boot':10000,'seed':0},
        'forecasts':{'information_protocol':'origin','grain':'fixture',
            'population':'Same preserved forecast pool; registered capture filter applied only in planning. Full unavailable-player coverage NOT ASSESSED.',
            **{arm:pin(old / f'{arm}-predictions.csv') for arm in ['candidate','control']}},
        'settlement_replays':{arm:pin(out / f'{arm}-settlement/complete.json') for arm in ['candidate','control']}}
    for arm in ['candidate','control']:
        spec[arm]={'name':'CatBoost MAE without5 ownership summaries' if arm=='candidate' else 'CatBoost MAE full115-feature control',
                   'input':pin(out / f'{arm}-settlement/replayed.csv')}
    write_json_new(out / 'comparison-spec.json', spec)
    run_experiment(out / 'comparison-spec.json', runs_dir=out / 'reports', attempt='first')
    assert _source_hashes() == plan['source_sha256']
    assert _runtime_versions() == plan['runtime']
    write_json_new(out / 'worker-complete.json', {'status':'COMPLETED_UNVERIFIED_COMPARISON',
        'elapsed_seconds':time.monotonic()-started,'model_fits':0,'plan_sha256':sha256_file(root/'plan.json'),
        'promotion_eligible':False,'output_sha256':{str(p.relative_to(out)):sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(result), flush=True)
except BaseException as exc:
    write_json_new(out / 'worker-failed.json', {'status':'FAILED','error':str(exc),'type':type(exc).__name__,
        'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started,
        'promotion_eligible':False,'plan_sha256':sha256_file(root/'plan.json')})
    raise
