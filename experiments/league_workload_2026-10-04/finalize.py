"""Wait for terminal workload evidence and verify once; never fit or retry failure."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parent
EXPECTED_VERIFIER = '08077e2075c0aab2f1b6c76c3ef7de7aac366442246f532ecf8ab3c8b3340b7a'


def main():
    study = ROOT / 'seed0'; output = study / 'verification.json'
    if output.exists():
        raise ValueError('Preserve earlier verification')
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        try:
            summary = json.loads((study / 'summary.json').read_bytes())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5); continue
        if summary.get('status') in ('complete', 'selection_rejected'):
            break
        if summary.get('status') == 'incomplete':
            raise ValueError('Workload run failed; refuse numerical verification')
        time.sleep(5)
    else:
        raise TimeoutError('No terminal evidence within two hours; do not retry')
    script = ROOT / 'verify.py'
    if hashlib.sha256(script.read_bytes()).hexdigest() != EXPECTED_VERIFIER:
        raise ValueError('Reviewed verifier bytes changed')
    spec = importlib.util.spec_from_file_location('independent_workload_verifier', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.verify(study)
    report['verifier_sha256'] = EXPECTED_VERIFIER
    report['finalizer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with output.open('x') as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False); file.write('\n')
    print({'status': report['status'], 'optimal_solves': report['optimal_solves']})


if __name__ == '__main__':
    main()
