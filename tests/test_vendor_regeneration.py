"""The local solver must remain reproducible from unchanged pinned upstream bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from fpl import config


def test_generator_reproduces_reviewed_patch_and_manifest_offline(tmp_path):
    source = config.ROOT / 'fpl/milp/_vendor/solio'
    target = tmp_path / 'fpl/milp/_vendor/solio'
    target.mkdir(parents=True)
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy2(config.ROOT / 'scripts/vendor_open_fpl.py', scripts / 'vendor_open_fpl.py')
    for name in ('upstream_solver.py.txt', 'LICENSE'):
        shutil.copy2(source / name, target / name)
    original = {name: (target / name).read_bytes() for name in ('upstream_solver.py.txt', 'LICENSE')}
    subprocess.run([sys.executable, str(scripts / 'vendor_open_fpl.py')], check=True, cwd=tmp_path)
    for name in ('solver.py', 'PATCH.diff', 'MANIFEST.json', '__init__.py'):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    for name, content in original.items():
        assert (target / name).read_bytes() == content
    manifest = json.loads((target / 'MANIFEST.json').read_text())
    assert manifest['commit'] == config.SOLIO_UPSTREAM_COMMIT
    assert manifest['upstream_sha256'] == hashlib.sha256(original['upstream_solver.py.txt']).hexdigest()
    assert manifest['local_sha256'] == hashlib.sha256((target / 'solver.py').read_bytes()).hexdigest()


def test_generator_rejects_changed_upstream_instead_of_relabelling_it(tmp_path):
    target = tmp_path / 'fpl/milp/_vendor/solio'
    target.mkdir(parents=True)
    (target / 'upstream_solver.py.txt').write_text('changed upstream source')
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy2(config.ROOT / 'scripts/vendor_open_fpl.py', scripts / 'vendor_open_fpl.py')
    result = subprocess.run([sys.executable, str(scripts / 'vendor_open_fpl.py')], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'pinned revision' in result.stderr
    assert not (target / 'solver.py').exists()
