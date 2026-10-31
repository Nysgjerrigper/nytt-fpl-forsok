"""Independent artifact arithmetic/admission verification after a completed run."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl import experiment

root = Path.cwd()
run = root / 'experiments/temporal_blend_2026-10-03/seed0'
read = lambda path: json.loads(path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
registration, summary = read(run / 'registration.json'), read(run / 'summary.json')
assert summary['status'] == 'complete'
assert sha(Path(registration['dataset'])) == registration['dataset_sha256']
for relative, digest in registration['code_sha256'].items():
    assert sha(run / registration['source_archive'] / relative) == digest
    assert sha(root / relative) == digest
for path, digest in registration['cached_input_sha256'].items():
    assert sha(Path(path)) == digest
weights = read(run / 'weights.json')['weights']
assert weights == registration['weights']
weight_lock, pilot_lock = read(run / 'weights_lock.json'), read(run / 'pilot_lock.json')
assert sha(run / 'weights.json') == weight_lock['weights_sha256'] == pilot_lock['weights_sha256']
assert sha(run / 'pilot_summary.json') == pilot_lock['pilot_summary_sha256']
assert pilot_lock['parameters'] == registration['parameters']
assert pilot_lock['feature_variants'] == registration['feature_variants']
keys = ['player_id', 'position', 'origin_gw', 'GW', 'fixture']
source = Path(registration['source_run'])
original = read(source / 'summary.json')
cnn = pd.read_csv(source / 'selection' / original['selection']['cnn_sequence']['predictions']).sort_values(keys).reset_index(drop=True)
tree = pd.read_csv(source / 'selection' / original['selection']['rmse_sequence']['predictions']).sort_values(keys).reset_index(drop=True)
metadata = sorted(set(tree) - {'predicted_total_points'})
pd.testing.assert_frame_equal(cnn[metadata], tree[metadata], check_dtype=False)
first, last = registration['windows']['weights']
mask = tree.GW.between(first, last) & (tree.GW == tree.origin_gw)
rounds = tree.loc[mask, ['player_id', 'position', 'GW', 'actual_total_points', 'predicted_total_points']].copy()
rounds['cnn'] = cnn.loc[mask, 'predicted_total_points']
rounds = rounds.groupby(['player_id', 'position', 'GW']).sum()
for position, frame in rounds.groupby('position'):
    d = frame.cnn.to_numpy() - frame.predicted_total_points.to_numpy()
    residual = frame.actual_total_points.to_numpy() - frame.predicted_total_points.to_numpy()
    expected = np.clip(np.sum(d * residual) / np.sum(d * d), 0, 1) if np.sum(d * d) else 0
    assert np.isclose(weights[position], expected, rtol=1e-13, atol=1e-15)
pilot_blend = pd.read_csv(run / 'pilot' / summary['pilot']['blend']['predictions']).sort_values(keys).reset_index(drop=True)
first, last = registration['windows']['pilot']
mask = tree.origin_gw.between(first, last) & tree.GW.between(first, last)
late_tree, late_cnn = tree.loc[mask].reset_index(drop=True), cnn.loc[mask].reset_index(drop=True)
pd.testing.assert_frame_equal(pilot_blend[metadata], late_tree[metadata], check_dtype=False)
w = late_tree.position.map(weights).to_numpy()
expected = (1-w)*late_tree.predicted_total_points + w*late_cnn.predicted_total_points
np.testing.assert_allclose(pilot_blend.predicted_total_points, expected, rtol=1e-13, atol=1e-13)
assert summary['advances'] and pilot_lock['advances']
assert summary['pilot']['blend']['total_points'] > summary['pilot']['rmse_sequence']['total_points']
assert read(run / 'pilot_summary.json')['evaluation'] == {}
reports, solves = {}, 0
for phase in ('pilot', 'evaluation'):
    for variant, modes in summary[phase].items():
        for mode, result in ({'origin': modes}.items() if phase == 'pilot' else modes.items()):
            directory = run / phase
            for filename, digest in (('predictions', 'prediction_sha256'), ('squads', 'squad_sha256')):
                assert sha(directory/result[filename]) == result[digest]
            squads = pd.read_csv(directory/result['squads'])
            assert set(squads.scoring_protocol) == {'autosubs_v2'}
            assert set(squads.points_accounting) == {'net_after_transfer_hits'}
            assert np.isfinite(squads.alpha).all() and (squads.alpha >= 0).all() and (squads.alpha % 1 == 0).all()
            np.testing.assert_allclose(squads.actual_total_points, squads.gross_actual_total_points-4*squads.alpha)
            assert np.isclose(squads.actual_total_points.sum(), result['total_points'])
            assert np.isclose(squads.gross_actual_total_points.sum(), result['gross_points'])
            assert np.isclose(4*squads.alpha.sum(), result['transfer_hit_points'])
            predictions = pd.read_csv(directory/result['predictions'])
            start, end = registration['windows']['pilot' if phase=='pilot' else 'reference']
            assert set(predictions.GW) == set(range(start,end+1)) and predictions.GW.max() < 232
            prefix = directory/f'{variant}_{mode}'
            optimal = prefix.with_suffix('.log').read_text().count('Status: Optimal (')
            assert optimal == end-start+1
            solves += optimal
            reports[f'{phase}/{variant}/{mode}'] = {'net': result['total_points'], 'gross': result['gross_points'], 'hits': result['transfer_hit_points']}
for mode in ('standard', 'origin'):
    candidate = pd.read_csv(run/'evaluation'/summary['evaluation']['blend'][mode]['predictions'])
    sort_keys = ['player_id', 'position', 'GW', 'fixture'] + (['origin_gw'] if mode=='origin' else [])
    columns = sorted(set(candidate)-{'predicted_total_points'})
    for control in ('rmse_sequence', 'mae_full'):
        baseline = pd.read_csv(run/'evaluation'/summary['evaluation'][control][mode]['predictions'])
        pd.testing.assert_frame_equal(candidate.sort_values(sort_keys)[columns].reset_index(drop=True), baseline.sort_values(sort_keys)[columns].reset_index(drop=True), check_dtype=False)
        comparison = summary['comparisons'][f'reference_blend_vs_{control}'][mode]
        assert comparison['n_gws'] == 31
        assert comparison['total_diff'] == summary['evaluation']['blend'][mode]['total_points']-summary['evaluation'][control][mode]['total_points']
verification = {'status': 'VERIFIED', 'summary_sha256': sha(run/'summary.json'),
                'registration_sha256': sha(run/'registration.json'), 'weights_sha256': sha(run/'weights.json'),
                'weights_lock_sha256': sha(run/'weights_lock.json'), 'pilot_lock_sha256': sha(run/'pilot_lock.json'),
                'optimal_solves': solves, 'archived_sources': len(registration['code_sha256']),
                'weights': weights, 'scores': reports, 'comparisons': summary['comparisons'],
                'elapsed_seconds': summary['elapsed_seconds'], 'verifier_sha256': sha(Path(__file__)),
                'promotion_eligible': False}
assert not (run/'verification.json').exists()
(run/'verification.json').write_text(json.dumps(verification,indent=2,sort_keys=True))
(run/'verify_outputs.py').write_bytes(Path(__file__).read_bytes())
experiment.log_result('blend_research_verified', {'run': str(run), 'verification_sha256': sha(run/'verification.json')},
                      {'status': 'VERIFIED', 'promotion_eligible': False, 'comparisons': summary['comparisons']})
print(json.dumps(verification,indent=2))
