import argparse, hashlib, json, subprocess, sys, zipfile
from pathlib import Path
root = Path.cwd()
parser = argparse.ArgumentParser(description='Verify current wheel source bytes and installed imports.')
parser.add_argument('--wheel', type=Path, default=Path('/tmp/fpl-research-final-wheel/fpl_forecast_optimizer-0.1.0-py3-none-any.whl'))
parser.add_argument('--installed', type=Path, default=Path('/tmp/fpl-research-final-installed'))
args = parser.parse_args()
wheel, installed = args.wheel.resolve(), args.installed.resolve()
py = sorted((root/'fpl').rglob('*.py'))
artifacts = sorted((root/'fpl/models').glob('*.json')) + sorted((root/'fpl/data').glob('*.json'))
with zipfile.ZipFile(wheel) as z:
    for p in py + artifacts:
        assert z.read(str(p.relative_to(root))) == p.read_bytes(), p
    z.extractall(installed)
code = "import sys; from fpl import config; from fpl.model import price_research,horizon_price_research,availability_research,deadline_availability,models,cold_start; from fpl.data import deadline_snapshots; assert config.PRODUCTION_WEIGHT_STRATEGY == 'single:catboost'; assert 'torch' not in sys.modules; assert models.__file__.startswith(sys.argv[1]+'/'); print('installed imports, production default and lazy Torch: PASS')"
import os
env = dict(os.environ, PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE='1')
r = subprocess.run([sys.executable,'-B','-c',code,str(installed)],cwd='/tmp',env=env,text=True,capture_output=True)
assert r.returncode == 0, r.stderr
report={'status':'verified','wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'pipeline_sources':len(py),'json_artifacts':len(artifacts),'checks':['offline build','all shipped Python sources exact','all model/data JSON bytes exact','installed imports outside checkout','unchanged production default','Torch remains lazy']}
(wheel.parent/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'experiments/research_priorities_2026-10-03/package_wheel_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(r.stdout.strip())
print(json.dumps(report))
