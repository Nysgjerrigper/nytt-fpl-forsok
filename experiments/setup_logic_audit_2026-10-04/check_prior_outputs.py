"""Read-only audit of preserved forecasts and the limits of old solver evidence."""
import json
from pathlib import Path

import pandas as pd

from fpl import config
from fpl.model import research_package as rp

DIRECTORIES = [
    'research_priorities_2026-10-03', 'availability_research_2026-10-03',
    'price_provenance_2026-10-03', 'price_horizon_2026-10-03',
    'cnn_research_2026-10-03', 'temporal_blend_2026-10-03',
    'league_workload_2026-10-04',
]


def audit() -> dict:
    """Validate old forecast identities/pools; do not refit or certify old solves."""
    root = config.ROOT
    hashes, counts, failures = {}, {}, []
    for name in DIRECTORIES:
        directory = root / 'experiments' / name
        logs = sorted(directory.rglob('*.log'))
        reported = 0
        for path in logs:
            hashes[str(path.relative_to(root))] = rp.file_hash(path)
            reported += path.read_text().count('Status: Optimal (')
        forecasts = sorted(directory.rglob('*.predictions.csv'))
        for path in forecasts:
            hashes[str(path.relative_to(root))] = rp.file_hash(path)
            frame = pd.read_csv(path)
            try:
                rp.validate_forecasts(frame, int(frame.GW.min()), int(frame.GW.max()), 'origin_gw' in frame)
            except ValueError as exc:
                failures.append({'file': str(path.relative_to(root)), 'error': str(exc)})
        counts[name] = dict(logs=len(logs), reported_optimal_statuses=reported,
                            forecast_files=len(forecasts))
    paired = []
    base = root / 'experiments' / DIRECTORIES[0]
    for directory in sorted(base.glob('selection_seed*/selection')):
        control = directory / 'mae_full_origin.predictions.csv'
        for path in sorted(directory.glob('*.predictions.csv')):
            if path == control:
                continue
            try:
                rp.validate_paired_forecasts(path, control)
                paired.append(str(path.relative_to(root)))
            except ValueError as exc:
                failures.append({'file': str(path.relative_to(root)), 'error': str(exc)})
    for name, digest in hashes.items():
        if rp.file_hash(root / name) != digest:
            raise ValueError(f'Historical file changed during read-only audit: {name}')
    return dict(status='checked' if not failures else 'findings', promotion_eligible=False,
                studies=counts, failures=failures, paired_selection_forecasts=paired,
                historical_file_sha256=hashes, audit_source_sha256=rp.source_hashes(),
                total_reported_optimal_statuses=sum(c['reported_optimal_statuses'] for c in counts.values()),
                total_forecast_files=sum(c['forecast_files'] for c in counts.values()),
                limitation='Old PuLP status text does not prove exact optimality; old wrapper omitted zero gap and solution-status checks. Free-transfer lookahead also changed. New paired optimizer replays are required; no historical outputs were changed or solves rerun.')


if __name__ == '__main__':
    destination = Path(__file__).with_name('prior_outputs.json')
    if destination.exists():
        raise ValueError('Preserve existing audit evidence; choose a new audit invocation')
    result = audit()
    result['audit_script_sha256'] = rp.file_hash(Path(__file__))
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'total_reported_optimal_statuses', 'total_forecast_files', 'failures')}))
