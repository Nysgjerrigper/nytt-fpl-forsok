"""Post-run verification of sealed checkpoints and explicit no-chip finances.

This does not reconstruct deadline-known fixtures, exact-deadline registration,
full site legality, native primal/dual certificates or outcome-blind execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.financial_audit import audit_legacy_finances
from fpl.milp.origin_prices import load_origin_captures
from fpl.milp.replay import read_replay


def check_receipts(folder: Path) -> dict:
    """Verify this run's exhaustive terminal hashes, sequence and wrapper counters."""
    terminals = [p for p in (folder / 'complete.json', folder / 'failed.json') if p.exists()]
    if len(terminals) != 1:
        raise ValueError('one terminal required; unsealed directories are incomplete')
    terminal = json.loads(terminals[0].read_text())
    actual = {str(p.relative_to(folder)): sha256_file(p) for p in sorted(folder.rglob('*'))
              if p.is_file() and p != terminals[0]}
    assert actual == terminal['receipt_sha256']
    assert terminal['started_sha256'] == actual['started.json']
    started = json.loads((folder / 'started.json').read_text())
    requested = list(range(started['settings']['start_gw'], started['settings']['max_gw'] + 1))
    assert requested == terminal['requested_origins']
    attempted = returned = completed = 0
    origins, decisions, timings = [], [], []
    for sequence, path in enumerate(sorted(folder.glob('[0-9]*.json')), 1):
        event = json.loads(path.read_text())
        json.dumps(event, allow_nan=False)
        assert event['sequence'] == sequence
        assert path.name == f"{sequence:04d}-{event['event']}.json"
        assert event['started_sha256'] == terminal['started_sha256']
        if event['event'] == 'origin-start':
            origins.append(event['origin_gw'])
            assert origins == requested[:len(origins)]
        assert origins and event['origin_gw'] == origins[-1]
        attempted += event['event'] == 'vendor-start'
        returned += event['event'] == 'vendor-complete'
        completed += event['event'] == 'decision'
        assert event['attempted_vendor_calls'] == attempted
        assert event['completed_vendor_calls'] == returned
        assert 0 <= completed <= returned <= attempted <= len(origins)
        if event['event'] == 'decision':
            assert event['settlement_completed'] is False
            assert not any(k.startswith('actual_') for k in event['decision'])
            decisions.append(event['decision'])
        timings.append({key: value for key, value in event.items() if key.endswith('_seconds') and key != 'time_limit_seconds'})
    assert terminal['attempted_vendor_calls'] == attempted
    assert terminal['completed_vendor_calls'] == returned
    assert terminal['completed_origins'] == completed
    if terminal['status'] == 'COMPLETED':
        assert terminals[0].name == 'complete.json'
        assert origins == requested and completed == returned == attempted == len(requested)
    else:
        assert terminal['status'] == 'FAILED' and terminals[0].name == 'failed.json'
    return {'terminal': terminal, 'decisions': decisions,
            'timing_totals': {key: (sum(t[key] for t in timings if isinstance(t.get(key), (int, float)))
                                   if any(isinstance(t.get(key), (int, float)) for t in timings) else None)
                for key in sorted({key for t in timings for key in t})},
            'timing_measured_counts': {key: sum(isinstance(t.get(key), (int, float)) for t in timings)
                for key in sorted({key for t in timings for key in t})}}


def audit(bundle: Path, output: Path) -> None:
    """Preserve a new post-check without editing sealed experiment bytes."""
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError('post-check must stay outside the sealed attempt')
    output.mkdir(parents=True, exist_ok=False)
    try:
        seal = json.loads((bundle / 'attempt-seal.json').read_text())
        actual = {str(p.relative_to(bundle)): sha256_file(p) for p in sorted(bundle.rglob('*'))
                  if p.is_file() and p.name != 'attempt-seal.json'}
        assert actual == seal['output_sha256']
        plan = json.loads((bundle / 'plan.json').read_text())
        assert sha256_file(bundle / 'worker.py') == plan['worker_sha256']
        prior = Path(plan['prior_bundle'])
        captures = load_origin_captures(prior / 'captures/captures.json', season='2023-24', origins=list(range(137, 153)))
        ledger = pd.concat([capture['metadata'][['value']].reset_index(names='player_id').assign(GW=gw)
                            for gw, capture in captures.items()], ignore_index=True)
        ledger.to_csv(output / 'capture-prices.csv', index=False)
        result = {'status': 'POSTCHECK_COMPLETED', 'sealed_files': len(actual),
                  'attempt_seal_sha256': sha256_file(bundle / 'attempt-seal.json'), 'arms': {},
                  'native_exported_certificate': 'NOT ASSESSED', 'full_site_legality': 'NOT ASSESSED',
                  'exact_deadline_identity': 'NOT ASSESSED', 'promotion_eligible': False}
        for arm in ('control', 'candidate'):
            checkpoints = bundle / 'first' / f'{arm}-checkpoints'
            if not checkpoints.exists():
                result['arms'][arm] = {'status': 'NOT STARTED'}
                continue
            verified = check_receipts(checkpoints)
            rows = verified.pop('decisions')
            for row in rows:
                assert set(row['squad']) <= set(captures[row['gameweek']]['metadata'].index)
                assert row['solver_status'] == 'Optimal' and row['solver_outcome_columns'] == 'ABSENT'
            snapshot_path = bundle / 'first' / f'{arm}-decisions.json'
            if snapshot_path.exists():
                assert json.loads(snapshot_path.read_text())['decisions'] == rows
            if rows:
                # Solio calls initial purchases transfers_in; the existing auditor
                # represents a fresh draft with no paid-transfer list. Admit that
                # single semantic conversion only after checking all acquisitions.
                assert rows[0]['state_mode'] == 'fresh_build' and rows[0]['q_start'] == 0
                assert set(rows[0]['transfers_in']) == set(rows[0]['squad'])
                assert not rows[0]['transfers_out'] and rows[0]['alpha'] == 0
                canonical = deepcopy(rows)
                canonical[0]['transfers_in'] = []
                snapshot = {'schema_version': 1, 'settlement_completed': False, 'decisions': canonical}
                write_json_new(output / f'{arm}-fresh-normalized-finance-input.json', snapshot)
                financial = audit_legacy_finances(snapshot, ledger,
                    initial_state={'mode': 'fresh', 'squad': [], 'budget': 1000, 'free_transfers': 0},
                    ft_cap=5, ft_gain=1, selling_policy='half_profit')
                financial.to_csv(output / f'{arm}-financial-audit.csv', index=False)
                verified['no_chip_financial_arithmetic'] = 'PASS'
                verified['financial_rounds'] = len(financial)
                verified['fresh_normalization'] = 'Initial acquisitions validated, then transfers_in emptied only in derived audit copy.'
            settlement = bundle / 'first' / f'{arm}-settlement'
            if (settlement / 'complete.json').exists():
                verified['settlement'] = read_replay(settlement)
            result['arms'][arm] = verified
        write_json_new(output / 'complete.json', result)
        print(json.dumps({arm: {'status': evidence.get('terminal', {}).get('status'),
            'financial_rounds': evidence.get('financial_rounds')} for arm, evidence in result['arms'].items()}))
    except Exception as exc:
        write_json_new(output / 'failed.json', {'status': 'FAILED', 'type': type(exc).__name__, 'reason': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.bundle, args.output)
