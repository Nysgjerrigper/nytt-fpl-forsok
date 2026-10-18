"""Bounded preserved-forecast modern-rule replay under pinned origin captures."""
from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

from fpl import experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.milp.compare_backtests import compare
from fpl.milp.optimize import write_decision_snapshot
from fpl.milp.origin_prices import load_origin_captures, prepare_origin_frame
from fpl.milp.replay import run_replay, read_replay
from fpl.milp.solio import run_schedule, solve_window
from fpl.research_runner import _source_hashes, _runtime_versions, _verify_outputs, read_result


def main(plan_path: Path, output: Path) -> None:
    """Verify registered bytes, execute once per arm, then separately settle decisions."""
    plan = json.loads(plan_path.read_bytes())
    def verify():
        if (plan['source_sha256'] != _source_hashes() or plan['runtime'] != _runtime_versions()
                or plan['script_sha256'] != sha256_file(Path(__file__))):
            raise ValueError('registered source/runtime/script changed')
        for name, digest in plan['input_sha256'].items():
            if sha256_file(Path(name)) != digest:
                raise ValueError(f'registered input changed: {name}')
        read_result(Path(plan['original_bundle']))
    verify()
    captures = load_origin_captures(plan['capture_descriptor'], season=plan['season'], origins=plan['origins'])
    frames = {arm: pd.read_csv(BytesIO(Path(path).read_bytes())) for arm, path in plan['forecasts'].items()}
    for gw in plan['origins']:
        pools = []
        for frame in frames.values():
            selected = frame.loc[frame.origin_gw.eq(gw) & frame.GW.between(gw, min(gw + 2, 144))]
            prepared, _ = prepare_origin_frame(selected, captures[gw], origin_gw=gw)
            pools.append(prepared[['player_id', 'GW', 'fixture']].reset_index(drop=True))
        pd.testing.assert_frame_equal(*pools)
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / 'plan.json', plan)
    write_json_new(output / 'started.json', {'status':'STARTED', 'plan_sha256':sha256_file(plan_path),
                                           'promotion_eligible':False})
    results = {}
    for arm, frame in frames.items():
        target = output / arm
        target.mkdir()
        write_json_new(target / 'started.json', {'status':'STARTED', 'plan_sha256':sha256_file(plan_path)})
        calls = []
        def recorded_solve(prepared, **kwargs):
            number = len(calls) + 1
            calls.append(kwargs['start_gw'])
            if number > 4:
                raise ValueError('registered solve budget exceeded')
            prepared.to_csv(target / f'window-{number}-inputs.csv', index=False)
            write_json_new(target / f'window-{number}-state.json',
                {key:value for key,value in kwargs.items() if key != 'bootstrap'})
            solved = solve_window(prepared, **kwargs)
            solved.to_json(target / f'window-{number}-solution.json', orient='records', indent=2)
            return solved
        try:
            with patch('fpl.milp.solio.solve_window', recorded_solve):
                decisions = run_schedule(frame, season=plan['season'], start_gw=141, max_gw=144, horizon=3,
                    bank=1000, free_transfers=1, origin_price_captures=captures, time_limit=120)
            write_decision_snapshot(decisions, target / 'decisions.json')
            decisions.to_json(target / 'decisions-dataframe.json', orient='records', indent=2)
            # Settlement is a distinct API after the pre-outcome snapshot is durable.
            run_replay(target / 'decisions.json', plan['forecasts'][arm], target / 'settlement',
                       blank_evidence_path=plan['blank_evidence'])
            settled = read_replay(target / 'settlement')
            results[arm] = {'status':'DECISIONS_AND_SETTLEMENT_COMPLETED',
                'gross_points':settled['gross_points'], 'net_points':settled['net_points'],
                'explicit_blank_outcomes':settled['explicit_blank_outcomes'],
                'solver_seconds':float(decisions.solver_seconds.sum()),
                'origin_receipts':decisions.price_input_receipt.tolist(),
                'solver_status':decisions.solver_status.tolist(), 'mip_gap':decisions.mip_gap.tolist(),
                'promotion_eligible':False}
            write_json_new(target / 'complete.json', results[arm])
        except Exception as exc:
            results[arm] = {'status':'FAILED', 'type':type(exc).__name__, 'reason':str(exc), 'promotion_eligible':False}
            write_json_new(target / 'failed.json', results[arm])
        experiment.log_result('origin-price-'+arm, {'plan_sha256':sha256_file(plan_path),
            'scope':'modern_rule_engineering_diagnostic'}, results[arm], results_path=output / 'registry.csv')
        print(arm, results[arm]['status'], flush=True)
    comparison = None
    if all(value['status']=='DECISIONS_AND_SETTLEMENT_COMPLETED' for value in results.values()):
        comparison = compare(str(output/'control/settlement/replayed.csv'),
            str(output/'candidate/settlement/replayed.csv'), n_boot=10000, block_len=3, seed=0,
            points_col='actual_net_points')
    verify()
    result = {'status':'BOUNDED_ATTEMPTS_COMPLETED', 'plan_sha256':sha256_file(plan_path),
        'model_fits':0, 'maximum_optimizer_solves':8, 'cases':results, 'comparison':comparison,
        'promotion_eligible':False, 'price_policy':plan['price_policy'], 'rules':plan['rules'],
        'solver_outcome_columns':'ABSENT', 'complete_outcome_blind_execution':'NOT ASSESSED',
        'fixture_information_causality':'NOT ASSESSED', 'full_site_legality':'NOT ASSESSED',
        'native_certificate_export':'NOT ASSESSED',
        'limitation':'Four already examined selection rounds, one seed, modern rules on historical forecasts; distinct recipes change capacity as well as loss. Not promotion, loss-only evidence or comparable with legacy controls.'}
    write_json_new(output/'result.json', result)
    hashes = {str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json_new(output/'complete.json', {'output_sha256':hashes, 'promotion_eligible':False})
    _verify_outputs(output, hashes, {'complete.json'})


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
