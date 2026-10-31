"""Stdlib-only preflight checks: all training/subprocesses forbidden."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/availability_research_2026-10-03/run_rmse_robustness.py'
CODE = SCRIPT.read_bytes()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v))

class Preflight(unittest.TestCase):

    def fixture(self):
        tmp = tempfile.TemporaryDirectory(prefix='fpl_rmse_preflight_')
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name)
        root = repo / 'experiments/availability_research_2026-10-03'
        root.mkdir(parents=True)
        driver = root / SCRIPT.name
        driver.write_bytes(CODE)
        verifier = root / 'verify_loss_study.py'
        verifier.write_text('# stub\n')
        dataset = repo / 'dataset.csv'
        dataset.write_bytes(b'frozen input')
        reg = root / 'rmse_seed0/registration.json'
        lock = reg.with_name('selection_lock.json')
        summary = reg.with_name('summary.json')
        report = reg.with_name('verification.json')
        archived = reg.parent / 'evaluated_sources/fpl/test.py'
        archived.parent.mkdir(parents=True)
        archived.write_bytes(b'archived source')
        forecasts = reg.parent / 'forecasts.csv'
        forecasts.write_bytes(b'preserved baseline predictions')
        write(reg, {'seed': 0, 'model_name': 'catboost_rmse', 'code_sha256': {'fpl/test.py': sha(archived)}, 'source_archive': 'evaluated_sources'})
        write(lock, {'locked': True})
        write(summary, {'status': 'complete', 'advanced': True, 'promotion_eligible': False, 'immutable_output_sha256': {'forecasts.csv': sha(forecasts)}})
        write(report, {'status': 'verified', 'summary_sha256': sha(summary), 'registration_sha256': sha(reg)})
        parent_path = root / 'rmse_robustness_registration.json'
        parent = {'model_name': 'catboost_rmse', 'seeds': [1, 2], 'comparison_by_seed': {'1': {'block_len': 3, 'n_boot': 10000, 'seed': 1}, '2': {'block_len': 3, 'n_boot': 10000, 'seed': 2}}, 'driver_sha256': sha(driver), 'verification_script_sha256': sha(verifier), 'seed0_registration_sha256': sha(reg), 'seed0_selection_lock_sha256': sha(lock), 'input_sha256': {}, 'dataset': str(dataset), 'dataset_sha256': sha(dataset), 'expected_code_sha256': {'fpl/test.py': 'sourcehash'}, 'versions': {'python': 'synthetic'}}
        write(parent_path, parent)
        run = Mock(side_effect=AssertionError('FORBIDDEN STUDY CALL'))
        experiment = types.ModuleType('fpl.experiment')
        experiment.log_result = Mock()
        experiment._library_versions = lambda: parent['versions']
        research = types.ModuleType('fpl.model.research_package')
        research.file_hash = sha
        research.source_hashes = lambda: parent['expected_code_sha256']
        availability = types.ModuleType('fpl.model.availability_research')
        availability.run = run
        config = types.ModuleType('fpl.config')
        config.ROOT = repo
        fpl = types.ModuleType('fpl')
        fpl.config = config
        fpl.experiment = experiment
        model = types.ModuleType('fpl.model')
        model.availability_research = availability
        model.research_package = research
        modules = {'fpl': fpl, 'fpl.model': model, 'fpl.config': config, 'fpl.experiment': experiment, 'fpl.model.research_package': research, 'fpl.model.availability_research': availability}
        module = types.ModuleType('driver_test')
        module.__file__ = str(driver)
        with patch.dict(sys.modules, modules):
            exec(compile(CODE, str(SCRIPT), 'exec'), module.__dict__)
        module.__file__ = str(driver)
        module.subprocess = types.SimpleNamespace(run=Mock(side_effect=AssertionError('FORBIDDEN SUBPROCESS')))
        return types.SimpleNamespace(**locals())

    def reject(self, f):
        with self.assertRaises(ValueError):
            f.module.main()
        f.run.assert_not_called()
        f.module.subprocess.run.assert_not_called()
        self.assertFalse((f.root / 'rmse_robustness_input_lock.json').exists())
        self.assertFalse((f.root / 'rmse_seed1').exists())
        self.assertFalse((f.root / 'rmse_seed2').exists())

    def test_existing_lock_and_outputs_refused_without_overwrite(self):
        for name in ('rmse_robustness_input_lock.json', 'rmse_robustness_summary.json', 'rmse_seed1', 'rmse_seed2'):
            with self.subTest(name=name):
                f = self.fixture()
                p = f.root / name
                p.write_bytes(b'preserved')
                with self.assertRaises(ValueError):
                    f.module.main()
                self.assertEqual(p.read_bytes(), b'preserved')
                f.run.assert_not_called()
                f.experiment.log_result.assert_not_called()

    def test_bad_model(self):
        f = self.fixture()
        f.parent['model_name'] = 'catboost_mae'
        write(f.parent_path, f.parent)
        self.reject(f)

    def test_invalid_seed_lists(self):
        for seeds in ([2, 1], [1], [1, 2, 3], [True, 2], [1, False], [1.0, 2], None, '1,2'):
            with self.subTest(seeds=seeds):
                f = self.fixture()
                f.parent['seeds'] = seeds
                write(f.parent_path, f.parent)
                self.reject(f)

    def test_wrong_driver_hash(self):
        f = self.fixture()
        f.parent['driver_sha256'] = '0' * 64
        write(f.parent_path, f.parent)
        self.reject(f)

    def test_wrong_verifier_hash(self):
        f = self.fixture()
        f.parent['verification_script_sha256'] = '0' * 64
        write(f.parent_path, f.parent)
        self.reject(f)

    def test_seed_zero_must_be_complete_advanced_verified_and_rmse(self):
        for name, field, value in (('summary', 'status', 'running'), ('summary', 'advanced', False), ('report', 'status', 'running'), ('reg', 'seed', 1), ('reg', 'model_name', 'catboost_mae')):
            with self.subTest(name=name, field=field):
                f = self.fixture()
                p = getattr(f, name)
                d = json.loads(p.read_bytes())
                d[field] = value
                write(p, d)
                if name == 'reg':
                    f.parent['seed0_registration_sha256'] = sha(p)
                    write(f.parent_path, f.parent)
                self.reject(f)

    def test_optimized_compile_refuses_before_import_and_main(self):
        module = types.ModuleType('optimized_driver')
        module.__file__ = str(SCRIPT)
        with self.assertRaisesRegex(RuntimeError, 'normal Python'):
            exec(compile(CODE, str(SCRIPT), 'exec', optimize=1), module.__dict__)

    def test_seed_zero_boolean_identity_rejected_before_execution(self):
        f = self.fixture()
        registration = json.loads(f.reg.read_bytes())
        registration['seed'] = False
        write(f.reg, registration)
        f.parent['seed0_registration_sha256'] = sha(f.reg)
        write(f.parent_path, f.parent)
        write(f.report, {'status': 'verified', 'summary_sha256': sha(f.summary), 'registration_sha256': sha(f.reg)})
        self.reject(f)

    def test_baseline_artifact_and_archived_source_bytes_bound_before_execution(self):
        for name in ('forecasts', 'archived'):
            with self.subTest(name=name):
                f = self.fixture()
                getattr(f, name).write_bytes(b'changed baseline evidence')
                self.reject(f)
if __name__ == '__main__':
    unittest.main(verbosity=2)
