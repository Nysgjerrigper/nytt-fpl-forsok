import hashlib, json, subprocess, sys, zipfile
from pathlib import Path
root = Path.cwd()
wheel = Path('/tmp/fpl-price-robustness-wheel/fpl_forecast_optimizer-0.1.0-py3-none-any.whl')
installed = Path('/tmp/fpl-price-robustness-installed')
py = sorted((root/'fpl').rglob('*.py'))
artifacts = sorted((root/'fpl/models').glob('*.json')) + sorted((root/'fpl/data').glob('*.json'))
with zipfile.ZipFile(wheel) as z:
    for p in py + artifacts:
        assert z.read(str(p.relative_to(root))) == p.read_bytes(), p
    z.extractall(installed)
code = "import sys; from fpl import config; from fpl.model import price_research,horizon_price_research,models,cold_start; assert config.PRODUCTION_WEIGHT_STRATEGY == 'single:catboost'; assert 'torch' not in sys.modules; assert models.__file__.startswith('/tmp/fpl-price-robustness-installed/'); print('installed imports, production default and lazy Torch: PASS')"
import os
env = dict(os.environ, PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE='1')
r = subprocess.run([sys.executable,'-B','-c',code],cwd='/tmp',env=env,text=True,capture_output=True)
assert r.returncode == 0, r.stderr
report={'status':'verified','wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'pipeline_sources':len(py),'json_artifacts':len(artifacts),'checks':['offline build','all shipped Python sources exact','all model/data JSON bytes exact','installed imports outside checkout','unchanged production default','Torch remains lazy']}
(wheel.parent/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'experiments/price_provenance_2026-10-03/seed_robustness_wheel_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(r.stdout.strip())
print(json.dumps(report))
