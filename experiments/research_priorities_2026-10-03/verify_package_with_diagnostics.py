"""Extend the preserved sixteen-study evidence with two verified diagnostics only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_maps(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'immutable_output_sha256':
                yield child
            else:
                yield from immutable_maps(child)
    elif isinstance(value, list):
        for child in value:
            yield from immutable_maps(child)


def main():
    """Rehash evidence without refitting, solving or overwriting earlier reports."""
    output = Path(__file__).resolve().parent / 'package_with_diagnostics_verification.json'
    if output.exists():
        raise ValueError('Preserve earlier verification; refuse overwrite')
    bound, artifacts, studies = {}, {}, {}
    def bind(path, expected=None):
        path = Path(path).resolve(); actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Hash mismatch: {path}')
        if str(path) in bound and bound[str(path)] != actual:
            raise ValueError(f'Verification-time drift: {path}')
        bound[str(path)] = actual
        return actual
    base_path = output.parent / 'package_with_availability_replications_verification.json'
    base_sha = bind(base_path); base = json.loads(base_path.read_bytes())
    if base['status'] != 'verified' or len(base['studies']) != 16 or base['verified_optimal_solves'] != 2082:
        raise ValueError('Sixteen-study base proof differs')
    total = 0
    for relative, entry in base['studies'].items():
        directory = ROOT / 'experiments' / relative
        bind(directory / 'summary.json', entry['summary_sha256'])
        bind(directory / 'verification.json', entry['verification_sha256'])
        summary = json.loads((directory / 'summary.json').read_bytes())
        verification = json.loads((directory / 'verification.json').read_bytes())
        if verification.get('summary_sha256', verification.get('summary_file_sha256')) != entry['summary_sha256']:
            raise ValueError('Per-study numerical proof differs')
        for mapping in immutable_maps(summary):
            for name, expected in mapping.items():
                path = Path(name); path = path if path.is_absolute() else directory / path
                artifacts[str(path.resolve())] = bind(path, expected)
        if type(entry['optimal_solves']) is not int or entry['optimal_solves'] < 0:
            raise ValueError('Invalid solve count')
        total += entry['optimal_solves']; studies[relative] = entry
    if total != 2082 or len(artifacts) != 953:
        raise ValueError('Original package output/solve count changed')
    diagnostics = ['grouped_interval_calibration_2026-10-04/seed0', 'registered_cold_outcomes_2026-10-04/seed0']
    for relative in diagnostics:
        directory = ROOT / 'experiments' / relative
        summary_sha = bind(directory / 'summary.json'); report_sha = bind(directory / 'verification.json')
        summary = json.loads((directory / 'summary.json').read_bytes())
        report = json.loads((directory / 'verification.json').read_bytes())
        if summary['status'] != 'complete' or report['status'] != 'verified' or report['promotion_eligible'] is not False or report['summary_sha256'] != summary_sha:
            raise ValueError('Additional diagnostic is incomplete or unverified')
        bind(directory / 'registration.json', report['registration_sha256'])
        maps = list(immutable_maps(summary))
        if 'immutable_input_output_sha256' in summary:
            maps.append(summary['immutable_input_output_sha256'])
        for mapping in maps:
            for name, expected in mapping.items():
                path = Path(name); path = path if path.is_absolute() else directory / path
                actual = bind(path, expected)
                # The cold audit combines input/output locks. Only generated
                # directory-local outputs enter the comparable output count.
                if path.resolve().is_relative_to(directory.resolve()):
                    artifacts[str(path.resolve())] = actual
        studies[relative] = {'status': summary['status'], 'summary_sha256': summary_sha,
                             'verification_sha256': report_sha, 'optimal_solves': 0, 'diagnostic_only': True}
    if len(studies) != 18 or len(artifacts) != 1074:
        raise ValueError('Diagnostic extension count differs')
    for path, expected in bound.items():
        if digest(path) != expected:
            raise ValueError(f'Artifact changed during verification: {path}')
    report = {'status': 'verified', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'base_package_sha256': base_sha, 'verifier_sha256': digest(Path(__file__)), 'studies': studies,
              'verified_optimal_solves': total, 'rechecked_unique_immutable_outputs': len(artifacts),
              'rechecked_artifacts': len(bound), 'promotion_eligible': False, 'production_strategy': 'single:catboost',
              'limitations': base['limitations'] + ['Two additional diagnostics supply no new optimizer solves or model-promotion evidence.']}
    with output.open('x') as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False); file.write('\n')
    print({key: report[key] for key in ['status', 'verified_optimal_solves', 'rechecked_unique_immutable_outputs', 'rechecked_artifacts']})


if __name__ == '__main__':
    main()
