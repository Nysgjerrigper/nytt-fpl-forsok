"""Preserve live forecasts and evaluate only explicitly checked round outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from fpl import config
from fpl.forecast_scorecard import summarize_forecast_pair


def capture(predictions: pd.DataFrame, bootstrap: dict, *, strategy: str,
            recovered: bool = False, root: Path | None = None) -> Path:
    """Capture immutable bytes; recovered files never acquire deadline provenance."""
    from fpl.data.fetch import active_season_from_bootstrap

    season = active_season_from_bootstrap(bootstrap)
    now = datetime.now(timezone.utc)
    rounds = sorted(int(value) for value in predictions.GW.unique())
    events = {event['id']: event for event in bootstrap['events']}
    deadlines = {str(gw): events[gw]['deadline_time'] for gw in rounds}
    payload = predictions.to_csv(index=False).encode()
    folder = (root or config.LIVE_FORECAST_DIR) / f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'predictions.csv').write_bytes(payload)
    metadata = {
        'schema_version': 1, 'season': season, 'captured_at': now.isoformat(),
        'strategy': strategy, 'origin_gw': min(rounds), 'gameweeks': rounds,
        'deadlines': deadlines, 'recovered': recovered,
        'deadline_evidence': 'unknown' if recovered else 'captured_before_deadline',
        'sha256': hashlib.sha256(payload).hexdigest(),
    }
    if not recovered and any(now >= datetime.fromisoformat(value.replace('Z', '+00:00'))
                             for value in deadlines.values()):
        metadata['deadline_evidence'] = 'captured_after_deadline'
    (folder / 'capture.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return folder


def evaluate(folder: Path, actuals: pd.DataFrame, bootstrap: dict) -> dict:
    """Join preserved forecasts to checked outcomes; absent player rows stay unknown.

    Actuals use stable player_code and season-local GW from the mutable dataset.
    DGWs are summed before diagnostics. Forecast origins are never pooled together.
    """
    from fpl.data.fetch import active_season_from_bootstrap

    metadata = json.loads((folder / 'capture.json').read_text())
    payload = (folder / 'predictions.csv').read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata['sha256']:
        raise ValueError('forecast capture hash mismatch')
    if active_season_from_bootstrap(bootstrap) != metadata['season']:
        raise ValueError('bootstrap season does not match capture')
    predictions = pd.read_csv(io.BytesIO(payload))
    required = ['player_id', 'GW', 'position', 'predicted_total_points']
    if not set(required) <= set(predictions):
        raise ValueError('missing forecast fields')
    if predictions[required].isna().any().any() or not np.isfinite(predictions.predicted_total_points).all():
        raise ValueError('invalid forecast values')
    identity = ['player_id', 'GW'] + [c for c in ('opponent_team', 'was_home') if c in predictions]
    if predictions.duplicated(identity).any():
        raise ValueError('duplicate or ambiguous forecast fixture identity')
    positions = predictions.groupby(['player_id', 'GW']).position.nunique()
    if positions.gt(1).any():
        raise ValueError('conflicting forecast positions')
    predictions = predictions.groupby(['player_id', 'GW'], as_index=False).agg(
        position=('position', 'first'), predicted_total_points=('predicted_total_points', 'sum'))
    checked = {event['id'] for event in bootstrap['events'] if event.get('data_checked')}
    history = actuals.loc[actuals.season == metadata['season']].copy()
    # A malformed existing outcome must fail, rather than disappearing in aggregation.
    if history.total_points.isna().any() or not np.isfinite(history.total_points).all():
        raise ValueError('invalid actual points')
    outcomes = history.groupby(['player_code', 'GW'], as_index=False).total_points.sum().rename(
        columns={'player_code': 'player_id', 'total_points': 'actual_total_points'})
    reports = []
    for gw, panel in predictions.groupby('GW'):
        item = {'gameweek': int(gw), 'lead': int(gw) - metadata['origin_gw'],
                'forecast_player_rounds': len(panel)}
        if gw not in checked:
            item['status'] = 'pending'
        else:
            joined = panel.merge(outcomes, on=['player_id', 'GW'], how='left', validate='one_to_one')
            missing = joined.actual_total_points.isna()
            item.update(missing_outcomes=int(missing.sum()), evaluated_player_rounds=int((~missing).sum()),
                        missing_player_ids=joined.loc[missing, 'player_id'].astype(int).tolist(),
                        status='complete' if not missing.any() else 'partial')
            if (~missing).any():
                covered = joined.loc[~missing].copy()
                summary = summarize_forecast_pair(covered, covered, information_protocol='standard',
                    grain='player_round', gameweeks=[int(gw)], population='saved live forecast rows with explicit outcomes')
                item['diagnostics'] = summary['summaries'][0]['candidate']
        reports.append(item)
    return {'capture': str(folder), 'metadata': metadata, 'rounds': reports,
            'limitations': 'Forecast diagnostics only; not realized squad points or model-promotion evidence.'}


def main() -> None:
    """Evaluate all saved runs using current checked-event and local outcome bytes."""
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=config.LIVE_FORECAST_DIR)
    parser.add_argument('--actuals', type=Path, default=config.MASTER_DATASET_PATH)
    args = parser.parse_args()
    response = requests.get(f'{config.FPL_API_BASE}/bootstrap-static/', timeout=config.FPL_API_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    actuals = pd.read_csv(args.actuals)
    reports = [evaluate(path.parent, actuals, payload) for path in sorted(args.root.glob('*/capture.json'))]
    print(json.dumps({'reports': reports}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
