"""Read-only timing ledger from a verified package; no fitting or scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_json(path: Path, expected: str) -> dict[str, Any]:
    data = path.read_bytes()
    if _hash(data) != expected:
        raise ValueError(f"Package-bound artifact SHA256 mismatch: {path}")
    return json.loads(data)


def _seconds(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid recorded timing: {label}")
    return float(value)


def _arm_timings(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(value, dict):
        if 'fit_seconds' in value and 'milp_seconds' in value:
            yield path, value
        for key, child in value.items():
            if key != 'reused_controls':
                yield from _arm_timings(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _arm_timings(child, (*path, str(index)))


def summarize(package_path: Path) -> dict[str, Any]:
    """Rehash package-bound study summaries/reports and extract recorded timings.

    Forecast timing includes repeated refits and prediction generation. Cached price
    replays exclude model refitting. Timing subtotals are arm components, never a
    substitute for whole-study wall time or a grand elapsed-time total.
    """
    package_bytes = package_path.read_bytes()
    package = json.loads(package_bytes)
    if package.get('status') != 'verified':
        raise ValueError('Package status must be verified')
    experiments_root = Path(__file__).resolve().parents[1]
    studies = {}
    for name, metadata in package['studies'].items():
        directory = (experiments_root / name).resolve()
        if not directory.is_relative_to(experiments_root):
            raise ValueError(f'Study path escapes experiments directory: {name}')
        summary = _verified_json(directory / 'summary.json', metadata['summary_sha256'])
        report = _verified_json(directory / 'verification.json', metadata['verification_sha256'])
        if summary.get('status') != metadata['status'] or report.get('status', '').lower() not in ('pass', 'verified'):
            raise ValueError(f'Package study status differs or verification is invalid: {name}')
        if report.get('optimal_solves', 0) != metadata['optimal_solves']:
            raise ValueError(f'Package optimal solve count differs: {name}')
        cached_price = name.startswith(('price_provenance_', 'price_horizon_'))
        generation_label = 'cached_replay_seconds' if cached_price else 'forecast_generation_seconds'
        arms = []
        for location, result in _arm_timings(summary):
            generation = _seconds(result['fit_seconds'], f'{name}/{"/".join(location)}/fit_seconds')
            milp = _seconds(result['milp_seconds'], f'{name}/{"/".join(location)}/milp_seconds')
            if generation is None or milp is None:
                raise ValueError(f'Completed arm has missing timing: {name}/{location}')
            arms.append({'summary_path': list(location), generation_label: generation, 'milp_seconds': milp,
                         'arm_component_subtotal_seconds': generation + milp})
        study = {'status': metadata['status'], 'optimal_solves': metadata['optimal_solves'],
                 'summary_sha256': metadata['summary_sha256'], 'verification_sha256': metadata['verification_sha256'],
                 'whole_study_wall_seconds': _seconds(summary.get('elapsed_seconds'), f'{name}/elapsed_seconds'),
                 'arm_timing_count': len(arms), 'arm_timings': arms,
                 'arm_timing_subtotal_seconds': sum(arm['arm_component_subtotal_seconds'] for arm in arms),
                 generation_label: sum(arm[generation_label] for arm in arms),
                 'milp_seconds': sum(arm['milp_seconds'] for arm in arms)}
        if name.startswith('interval_calibration_'):
            phases = {}
            for phase, result in summary['phases'].items():
                phases[phase] = {key: _seconds(result.get(key), f'{name}/{phase}/{key}')
                                 for key in ('fit_seconds', 'prediction_seconds', 'preprocessing_seconds', 'phase_seconds')}
            study['interval_phase_timings'] = phases
            # No arm/MILP phase exists for interval diagnostics.
            study.pop('milp_seconds')
            study.pop(generation_label)
            study['arm_timing_subtotal_seconds'] = None
        studies[name] = study
    return {'status': 'verified', 'package': str(package_path.resolve()), 'package_sha256': _hash(package_bytes),
            'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'studies': studies,
            'limitations': ['Forecast generation includes repeated refits and predictions; it is not a single-fit benchmark.',
                            'Cached price replay timing excludes model refitting.',
                            'Rejected runs include selection timing only.',
                            'Missing whole-study elapsed time remains null; arm subtotals do not reconstruct wall time.',
                            'No global grand wall-time sum: shared parent phases and reused controls are not independent work.',
                            'Wall time is not CPU-core seconds or active implementation time.',
                            'These timings support compute planning, not scientific model promotion.']}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('Output already exists; preserve the existing timing ledger')
    result = summarize(args.package)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
