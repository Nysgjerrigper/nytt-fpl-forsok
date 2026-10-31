"""Register both reference-only price repetitions before any scoring; run once."""
from __future__ import annotations

import json
from pathlib import Path
import time

from fpl import config, experiment
from fpl.model import cnn_research, price_research, research_package


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    started = time.perf_counter()
    parent = Path(__file__).resolve().parent
    protocol_path = parent / 'seed_robustness_protocol.json'
    summary_path = parent / 'seed_robustness_summary.json'
    seeds = (1, 2)
    if protocol_path.exists() or any((parent / f'reference_seed{seed}').exists() for seed in seeds):
        raise ValueError('preserve registered runs; do not overwrite or retry')
    cutoffs = price_research.protocol_cutoffs()
    inputs, source_directories, parameters = {}, {}, {}
    dataset, dataset_sha = None, None
    for seed in seeds:
        source = config.ROOT / f'experiments/research_priorities_2026-10-03/selection_seed{seed}'
        original, hashes, _ = price_research._source_inputs(source, cutoffs, seed=seed)
        if dataset is not None and (original['dataset'] != str(dataset) or original['dataset_sha256'] != dataset_sha):
            raise ValueError('source seed inputs differ')
        dataset, dataset_sha = Path(original['dataset']), original['dataset_sha256']
        for path, digest in hashes.items():
            if path in inputs and inputs[path] != digest:
                raise ValueError('conflicting source input hashes')
            inputs[path] = digest
        source_directories[str(seed)] = str(source)
        parameters[str(seed)] = original['parameters']
    code = research_package.source_hashes()
    archive = parent / 'seed_robustness_sources'
    cnn_research._archive_sources(archive, code)
    archived = {str((archive / relative).resolve()): digest for relative, digest in code.items()}
    protocol = {
        'status': 'registered', 'promotion_eligible': False, 'issue': 13, 'seeds': list(seeds),
        'hypothesis': 'Conditional horizon-price sensitivity and model ranking may vary with refit seed.',
        'dataset': str(dataset), 'dataset_sha256': dataset_sha, 'code_sha256': code,
        'source_directories': source_directories, 'cached_input_sha256': inputs,
        'source_archive_sha256': archived, 'launcher_sha256': research_package.file_hash(Path(__file__)),
        'parameters': parameters, 'reference_window': [cutoffs['backtest_min_gw'], cutoffs['backtest_max_gw']],
        'reference_only': True, 'include_current_proxy': True, 'skip_lagged_proxy': True,
        'policies': ['target-replay', 'origin-current-proxy'], 'models': ['catboost_mae', 'catboost_rmse'],
        'expected_optimal_solves': 248, 'cached_replay_only': True, 'new_model_fits': 0,
        'comparison': {'n_boot': 10000, 'block_len': 3, 'seed': 'matching refit seed'},
        'comparison_labels': ['rmse_vs_mae_replay', 'mae_current_vs_replay', 'rmse_current_vs_replay', 'rmse_vs_mae_current'],
        'horizon': 3, 'retrain_every': 4, 'gap_rel': 0, 'solver': 'highs', 'chips': 'disabled',
        'scoring_protocol': 'autosubs_v2', 'points_accounting': 'net_after_transfer_hits',
        'limitations': ['Reference outcomes were repeatedly examined; no fresh holdout or promotion.',
                        'Seeds reuse fixed seed-zero tuning studies and share outcomes; not independent replications.',
                        'Known-price eligibility differs from unrestricted historical source controls.',
                        'Current prices retain historical reconstruction/filling; no deadline-publication proof.',
                        'Lagged-price budget policy excluded because season repricing changed first-origin budget feasibility, not because of model performance.',
                        'Legacy roster, selling-price and transfer approximations remain.'],
    }
    write(protocol_path, protocol)
    protocol_sha = research_package.file_hash(protocol_path)
    summary = {'status': 'running', 'promotion_eligible': False, 'protocol_sha256': protocol_sha, 'seeds': {}}
    completed_outputs = {}

    def check() -> None:
        if (research_package.source_hashes() != code
                or research_package.file_hash(protocol_path) != protocol_sha
                or research_package.file_hash(Path(__file__)) != protocol['launcher_sha256']
                or research_package.file_hash(dataset) != dataset_sha):
            raise ValueError('registered price robustness source/data/protocol changed')
        for path, digest in {**inputs, **archived, **completed_outputs}.items():
            if research_package.file_hash(Path(path)) != digest:
                raise ValueError(f'sealed robustness input/output changed: {path}')

    experiment.log_result('price_seed_robustness_registration', {'protocol': str(protocol_path), **protocol},
                          {'status': 'REGISTERED', 'promotion_eligible': False})
    try:
        for seed in seeds:
            check()
            output = parent / f'reference_seed{seed}'
            result = price_research.run(dataset, output, source_dir=source_directories[str(seed)], seed=seed,
                                       reference_only=True, include_current_proxy=True, skip_lagged_proxy=True)
            if result['status'] != 'complete':
                raise ValueError('price child did not complete')
            completed_outputs.update(result['immutable_output_sha256'])
            completed_outputs[str((output / 'summary.json').resolve())] = research_package.file_hash(output / 'summary.json')
            summary['seeds'][str(seed)] = {'status': result['status'], 'output_dir': str(output),
                                          'summary_sha256': research_package.file_hash(output / 'summary.json'),
                                          'results': {name: entry['total_points'] for name, entry in result['phases']['reference'].items()},
                                          'comparisons': result['comparisons']['reference']}
            check()
            write(summary_path, summary)
        summary['status'] = 'complete'
        check()
    except BaseException as exc:
        summary.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['elapsed_seconds'] = time.perf_counter() - started
        summary['immutable_output_sha256'] = completed_outputs
        write(summary_path, summary)
        experiment.log_result('price_seed_robustness_checkpoint', {'protocol': str(protocol_path), 'summary_sha256': research_package.file_hash(summary_path)},
                              {'status': summary['status'].upper(), 'promotion_eligible': False})


if __name__ == '__main__':
    main()
