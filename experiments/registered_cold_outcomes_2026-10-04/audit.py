"""Coverage-only outcome linkage for previously exported registered pooled priors.

No fitting, forecast scores, model selection, or optimization. Observed fixtures
are retrospective dataset labels, never evidence of deadline-known fixtures.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

WINDOW = (137, 183)
LABEL_COLUMNS = ['actual_total_points', 'actual_minutes', 'actual_fixture_count', 'position_mismatch', 'team_mismatch', 'outcome_observed', 'phase']
LIMITATIONS = [
    'Coverage only: no forecast accuracy, model selection, or MILP decision gain assessment.',
    'Retrospective fixture occurrence does not prove deadline-known fixtures or player eligibility.',
    'Repeated player codes across origins are not independent observations.',
    'Selected captures are strictly pre-deadline and at most 24 hours old; they do not prove exact deadline or throughout-horizon registration completeness.',
    'Absent dataset labels stay missing; partial label coverage is not zero points or zero minutes.',
    'Observed fixture counts never adjust saved priors or generate new forecasts.',
    'Previously examined historical outcomes; no fresh holdout claim.',
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def integers(series: pd.Series, name: str) -> pd.Series:
    canonical = series.astype('string').str.fullmatch(r'[+]?\d+(?:\.0+)?', na=False)
    numeric = pd.to_numeric(series, errors='coerce')
    if not canonical.all() or not np.isfinite(numeric).all() or (numeric <= 0).any() or (numeric % 1 != 0).any():
        raise ValueError(f'{name} requires positive whole keys excluding booleans')
    return numeric.astype('int64')


def link_outcomes(priors: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Preserve prior rows and attach same-round observed fixture sums only."""
    for frame in (priors, raw):
        if not frame.columns.is_unique:
            raise ValueError('Duplicate input columns')
    required = {'player_id', 'origin_gw', 'position', 'team', 'status'}
    if not required.issubset(priors) or not {'player_id', 'GW_global', 'position', 'team', 'total_points', 'minutes'}.issubset(raw):
        raise ValueError('Missing required prior/outcome columns')
    if set(LABEL_COLUMNS) & set(priors):
        raise ValueError('Outcome column collision')
    saved = priors.copy().reset_index(drop=True)
    for name in ('player_id', 'origin_gw'):
        saved[name] = integers(saved[name], name)
    if not saved.origin_gw.between(*WINDOW).all() or saved.duplicated(['player_id', 'origin_gw']).any():
        raise ValueError('Prior window/identity ambiguous')
    if not saved.position.isin(['GK', 'DEF', 'MID', 'FWD']).all() or not saved.status.isin(list('adisun')).all() or saved.team.isna().any():
        raise ValueError('Invalid prior metadata')
    # Cap before target validation/grouping: future outcome values do not enter this diagnostic.
    rounds = integers(raw.GW_global, 'GW_global')
    rows = raw.loc[rounds.between(*WINDOW)].copy()
    rows['GW_global'] = rounds.loc[rounds.between(*WINDOW)]
    rows['player_id'] = integers(rows.player_id, 'player_id')
    if 'fixture' in rows:
        if rows.fixture.isna().any() or rows.duplicated(['player_id', 'GW_global', 'fixture']).any():
            raise ValueError('Duplicate/unknown exact fixture identity')
    rows = rows.merge(saved[['player_id', 'origin_gw', 'position', 'team']].rename(columns={'position': 'registered_position', 'team': 'registered_team'}),
                      left_on=['player_id', 'GW_global'], right_on=['player_id', 'origin_gw'], how='inner', validate='many_to_one')
    for name in ('total_points', 'minutes'):
        values = pd.to_numeric(rows[name], errors='coerce')
        if rows[name].astype('string').isin(['True', 'False']).any() or not np.isfinite(values).all() or (name == 'minutes' and (values < 0).any()):
            raise ValueError(f'Invalid observed {name}')
        rows[name] = values
    if rows.position.isna().any() or rows.team.isna().any():
        raise ValueError('Missing observed position/team')
    rows['position_mismatch'] = rows.position.ne(rows.registered_position)
    rows['team_mismatch'] = rows.team.ne(rows.registered_team)
    labels = rows.groupby(['player_id', 'origin_gw'], as_index=False).agg(
        actual_total_points=('total_points', 'sum'), actual_minutes=('minutes', 'sum'),
        actual_fixture_count=('player_id', 'size'), position_mismatch=('position_mismatch', 'any'), team_mismatch=('team_mismatch', 'any'))
    result = saved.merge(labels, on=['player_id', 'origin_gw'], how='left', sort=False, validate='one_to_one')
    result['outcome_observed'] = result.actual_fixture_count.notna()
    result['phase'] = np.where(result.origin_gw <= 152, 'selection', 'reference')
    if len(result) != len(priors):
        raise ValueError('Prior rows were changed')
    return result


def coverage_counts(frame: pd.DataFrame) -> dict:
    """Counts and sums of observed labels; absent labels contribute no imputed values."""
    observed = frame.loc[frame.outcome_observed]
    return {'player_origin_rows': len(frame), 'observed_rows': len(observed), 'unobserved_rows': len(frame) - len(observed),
            'positive_minutes_rows': int(observed.actual_minutes.gt(0).sum()), 'positive_points_rows': int(observed.actual_total_points.gt(0).sum()),
            'observed_total_points': float(observed.actual_total_points.sum()) if len(observed) else None,
            'observed_fixture_rows': int(observed.actual_fixture_count.sum()),
            'position_mismatch_rows': int(observed.position_mismatch.eq(True).sum()), 'team_mismatch_rows': int(observed.team_mismatch.eq(True).sum())}


def read_frozen(path: Path, expected: str) -> pd.DataFrame:
    """Hash and parse the same explicitly admitted frozen bytes without a live refresh."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Frozen bytes changed before parsing')
    return pd.read_csv(io.BytesIO(payload))


def run(*, priors_path: Path, audit_path: Path, registration_path: Path, dataset_path: Path,
        dataset_sha256: str, output_dir: Path, repository: Path) -> dict:
    """Preregister fixed coverage-only linkage before parsing any frozen outcomes."""
    if output_dir.exists():
        raise ValueError('Preserve existing output directory; no overwrite/resume')
    if len(dataset_sha256) != 64 or any(c not in '0123456789abcdef' for c in dataset_sha256):
        raise ValueError('Explicit frozen dataset SHA256 required')
    locked = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Input SHA mismatch: {path}')
        if str(path) in locked and locked[str(path)] != actual:
            raise ValueError(f'Input drift: {path}')
        locked[str(path)] = actual
        return actual
    sources = {str(p.relative_to(repository)): digest(p) for p in sorted((repository / 'fpl').rglob('*.py'))}
    if not sources:
        raise ValueError('Repository source archive is empty')
    for path in (priors_path, audit_path, registration_path): bind(path)
    audit = json.loads(audit_path.read_bytes()); previous = json.loads(registration_path.read_bytes())
    if audit.get('status') != 'complete' or audit.get('promotion_eligible') is not False:
        raise ValueError('Prior audit is incomplete/promoted')
    bind(priors_path, audit['export_sha256']); bind(registration_path, audit['registration_sha256'])
    if previous['dataset_sha256'] != dataset_sha256 or Path(previous['dataset']).resolve() != dataset_path.resolve():
        raise ValueError('Frozen dataset differs from saved prior registration')
    bind(dataset_path, dataset_sha256); bind(Path(__file__))
    priors = pd.read_csv(priors_path)
    if len(priors) != audit['covered_player_origin_rows'] or len(priors) != 191:
        raise ValueError('Expected preserved 191 registered player-origin rows')
    output_dir.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    summary = {'status': 'incomplete', 'promotion_eligible': False}
    def check():
        current = {str(p.relative_to(repository)): digest(p) for p in sorted((repository / 'fpl').rglob('*.py'))}
        if current != sources: raise ValueError('Pipeline source drift')
        for path, expected in locked.items():
            if digest(Path(path)) != expected: raise ValueError(f'Preserved byte drift: {path}')
    def write_json(name, value):
        path = output_dir / name
        with path.open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
        bind(path)
    try:
        check()
        for relative, expected in sources.items():
            data = (repository / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected: raise ValueError('Source changed during archive')
            target = output_dir / 'evaluated_sources' / relative; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write(data)
            bind(target, expected)
        write_json('registration.json', {'status': 'registered', 'window': list(WINDOW), 'dataset_sha256': dataset_sha256,
                    'input_sha256': dict(locked), 'code_sha256': sources, 'script_sha256': digest(Path(__file__)),
                    'source_archive': 'evaluated_sources', 'saved_rows': 191, 'coverage_only': True,
                    'forecast_scores': False, 'model_selection': False, 'milp': False, 'promotion_eligible': False,
                    'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'limitations': LIMITATIONS})
        check()
        raw = read_frozen(dataset_path, dataset_sha256)
        check()
        linked = link_outcomes(priors, raw)
        check()
        csv = output_dir / 'registered_outcomes.csv'
        with csv.open('x') as f: linked.to_csv(f, index=False)
        bind(csv)
        check()
        grouped = {name: {str(key): coverage_counts(group) for key, group in linked.groupby(name, sort=True)}
                   for name in ('phase', 'position', 'status')}
        summary = {'status': 'complete', 'promotion_eligible': False, 'coverage_only': True,
                   'counts': coverage_counts(linked), 'by': grouped, 'limitations': LIMITATIONS,
                   'elapsed_seconds': time.perf_counter() - began, 'immutable_input_output_sha256': dict(locked)}
        check(); write_json('summary.json', summary); check()
        return summary
    except BaseException as exc:
        summary.update(status='incomplete', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-began)
        write_json('incomplete.json', summary)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('priors', 'audit', 'registration', 'dataset', 'output-dir', 'repository'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--dataset-sha256', required=True)
    a = p.parse_args()
    result = run(priors_path=a.priors, audit_path=a.audit, registration_path=a.registration,
                 dataset_path=a.dataset, dataset_sha256=a.dataset_sha256, output_dir=a.output_dir, repository=a.repository)
    print(json.dumps(result['counts'], indent=2))

if __name__ == '__main__': main()
