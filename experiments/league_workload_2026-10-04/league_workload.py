"""Historical EPL calendar workload from already frozen fixture kickoff fields.

No cup/Europe/Elo signal or exact deadline schedule-publication claim. Recorded
kickoffs do not prove when fixture schedules were published. Origin research must
freeze current-origin values over its horizon rather than use later round clocks.
Team names are already corrected by ingestion and span the historical seasons.
"""
from __future__ import annotations
from numbers import Integral
import numpy as np
import pandas as pd

COLUMNS = ['league_rest_days', 'league_matches_last7_days', 'league_matches_last14_days']


def _positive_integers(series: pd.Series, label: str) -> pd.Series:
    canonical = series.astype('string').str.fullmatch(r'[+]?\d+(?:\.0+)?', na=False)
    values = pd.to_numeric(series, errors='coerce')
    if not canonical.all() or not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
        raise ValueError(f'{label} requires positive whole IDs/gameweeks excluding booleans')
    return values.astype('int64')


def build_league_workload(raw: pd.DataFrame, max_gw: int) -> pd.DataFrame:
    """Return three prior-EPL workload features aligned with admitted fixture rows.

    For each team/round, use its earliest recorded kickoff. Eligible unique team
    fixtures have a strictly smaller round AND precede that clock. Rest is elapsed
    fractional days from the most recent eligible kickoff. Counts include elapsed
    times (0,7] and (0,14] days. With no eligible history all three values are NaN;
    otherwise genuine zero recent counts remain zero. Team-level date/GW masks are
    vectorized; no player/row feature loop is used. All same-round DGW fixtures
    share values and never count another current-round fixture as past workload.
    """
    if isinstance(max_gw, bool) or not isinstance(max_gw, Integral) or max_gw <= 0:
        raise ValueError('max_gw must be a positive integer')
    required = {'player_id', 'team', 'GW_global', 'fixture', 'kickoff_time'}
    if not isinstance(raw, pd.DataFrame) or not raw.columns.is_unique or not required.issubset(raw.columns):
        raise ValueError('Unique required player/team/GW/fixture/kickoff columns needed')
    rounds = _positive_integers(raw.GW_global, 'GW_global')
    admitted = rounds.le(max_gw)
    rows = raw.loc[admitted, list(required)].copy()
    if rows.empty:
        return pd.DataFrame(index=raw.index[admitted], columns=COLUMNS, dtype=float)
    rows['GW_global'] = rounds.loc[admitted].to_numpy()
    for column in ('player_id', 'fixture'):
        rows[column] = _positive_integers(rows[column], column)
    try:
        teams = rows.team.str.strip()
    except AttributeError as exc:
        raise ValueError('team must contain nonempty text') from exc
    if teams.isna().any() or teams.eq('').any():
        raise ValueError('team must contain nonempty text')
    explicit = rows.kickoff_time.astype('string').str.fullmatch(
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})', na=False)
    clocks = pd.to_datetime(rows.kickoff_time, format='mixed', utc=True, errors='coerce')
    if not explicit.all() or clocks.isna().any():
        raise ValueError('kickoff_time requires valid explicit-timezone ISO timestamps')
    rows['clock'] = clocks.to_numpy()
    unique_keys = ['team', 'GW_global', 'fixture']
    if rows.groupby(unique_keys, dropna=False).clock.nunique().gt(1).any():
        raise ValueError('Conflicting kickoff times for the same team/round/fixture')
    fixtures = rows.drop_duplicates(unique_keys)[unique_keys + ['clock']]
    references = fixtures.groupby(['team', 'GW_global'], as_index=False).clock.min()
    pieces = []
    # Fixed team groups: each mask compares round clocks with unique team fixtures,
    # never with individual player observations. Typical historical masks <1M cells.
    for team, targets in references.groupby('team', sort=False):
        historical = fixtures.loc[fixtures.team.eq(team)]
        ref_clock = pd.to_datetime(targets.clock, utc=True).astype('datetime64[ns, UTC]').astype('int64').to_numpy()
        past_clock = pd.to_datetime(historical.clock, utc=True).astype('datetime64[ns, UTC]').astype('int64').to_numpy()
        delta = (ref_clock[:, None] - past_clock[None, :]) / (86400.0 * 1e9)
        eligible = (historical.GW_global.to_numpy()[None, :] < targets.GW_global.to_numpy()[:, None]) & (delta > 0)
        has_history = eligible.any(axis=1)
        targets = targets[['team', 'GW_global']].copy()
        targets[COLUMNS[0]] = np.where(has_history, np.where(eligible, delta, np.inf).min(axis=1), np.nan)
        targets[COLUMNS[1]] = np.where(has_history, (eligible & (delta <= 7)).sum(axis=1), np.nan)
        targets[COLUMNS[2]] = np.where(has_history, (eligible & (delta <= 14)).sum(axis=1), np.nan)
        pieces.append(targets)
    if not pieces:
        return pd.DataFrame(index=raw.index[admitted], columns=COLUMNS, dtype=float)
    values = pd.concat(pieces, ignore_index=True)
    joined = rows[['team', 'GW_global']].merge(values, on=['team', 'GW_global'], how='left', sort=False, validate='many_to_one')
    return pd.DataFrame(joined[COLUMNS].to_numpy(dtype=float), index=raw.index[admitted], columns=COLUMNS)
