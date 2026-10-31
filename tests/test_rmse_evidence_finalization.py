"""Read-only sealed-finalizer review using stdlib fixtures and fake subprocesses."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest

SEALED = Path(__file__).resolve().parents[1] / 'experiments/availability_research_2026-10-03/finalize_rmse_evidence.py'
CODE = SEALED.read_bytes()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))

class Review(unittest.TestCase):
    def fixture(self, optimize=0):
        temp = tempfile.TemporaryDirectory(prefix='fpl_finalizer_case_')
        self.addCleanup(temp.cleanup)
        repo = Path(temp.name)
        directory = repo/'experiments/availability_research_2026-10-03'
        package_dir = repo/'experiments/research_priorities_2026-10-03'
        package_dir.mkdir(parents=True)
        script = directory/'finalize_rmse_evidence.py'
        script.parent.mkdir(parents=True)
        script.write_bytes(CODE)
        source = repo/'fpl/config.py'
        source.parent.mkdir()
        source.write_text('production = "single:catboost"\n')
        inp = repo/'input.bytes'
        inp.write_bytes(b'preserved data')
        summary = directory/'rmse_seed0/summary.json'
        verification = summary.with_name('verification.json')
        write(summary, {'status':'selection_rejected','advanced':False,'promotion_eligible':False,
                        'elapsed_seconds':12.5,'comparisons':{}})
        write(verification, {'status':'verified'})
        protocol = directory/'rmse_pilot_registration.json'
        write(protocol, {'expected_code_sha256':{'fpl/config.py':sha(source)}})
        report_path = directory/'rmse_pilot_verification.json'
        report = {'status':'verified','protocol_sha256':sha(protocol),
                  'input_sha256':{str(inp):sha(inp)},'summary_sha256':sha(summary),
                  'verification_sha256':sha(verification),'pilot_status':'selection_rejected','advanced':False}
        write(report_path, report)
        module = types.ModuleType('sealed_test')
        module.__file__ = str(script)
        exec(compile(CODE, str(SEALED), 'exec', optimize=optimize), module.__dict__)
        module.__file__ = str(script)
        calls = []
        package = package_dir/'package_with_rmse_availability_verification.json'
        ledger = package_dir/'compute_timings_with_rmse_verified.json'
        final = directory/'rmse_evidence_finalization.json'
        def stub(args, *, check):
            self.assertTrue(check)
            self.assertEqual(json.loads(report_path.read_bytes())['status'], 'verified')
            self.assertFalse(final.exists())
            calls.append(args)
            if len(calls) == 1:
                self.assertEqual(args[-2:], ['--include-availability','--include-rmse-availability'])
                self.assertFalse(ledger.exists())
                write(package, {'studies':{'availability_research_2026-10-03/rmse_seed0':
                                         {'summary_sha256':report['summary_sha256']}},
                                'verified_optimal_solves':100})
            else:
                self.assertTrue(package.exists())
                write(ledger, {'status':'verified','seconds':12.5})
            return types.SimpleNamespace(returncode=0)
        module.subprocess = types.SimpleNamespace(run=stub)
        return types.SimpleNamespace(**locals())
    def run_main(self, f):
        with contextlib.redirect_stdout(io.StringIO()):
            f.module.main()
    def rejected(self, f, error):
        with self.assertRaises(error): self.run_main(f)
        self.assertEqual(f.calls, [])
        self.assertFalse(f.package.exists())
        self.assertFalse(f.ledger.exists())
        self.assertFalse(f.final.exists())
    def test_unverified_parent(self):
        f=self.fixture(); f.report['status']='running'; write(f.report_path,f.report)
        self.rejected(f,ValueError)
    def test_changed_input(self):
        f=self.fixture(); f.inp.write_bytes(b'changed')
        self.rejected(f,AssertionError)
    def test_changed_source(self):
        f=self.fixture(); f.source.write_bytes(b'changed')
        self.rejected(f,AssertionError)
    def test_preexisting_outputs(self):
        for name in ('package','ledger','final'):
            with self.subTest(output=name):
                f=self.fixture(); path=getattr(f,name); path.write_bytes(b'preserve')
                with self.assertRaises(ValueError):self.run_main(f)
                self.assertEqual(f.calls,[]);self.assertEqual(path.read_bytes(),b'preserve')
    def test_success_requires_verified_parent_then_package_ledger_and_final(self):
        f=self.fixture();self.run_main(f)
        self.assertEqual(len(f.calls),2)
        result=json.loads(f.final.read_bytes())
        self.assertEqual(result['status'],'verified')
        for key,path in [('pilot_verification_sha256',f.report_path),('package_verification_sha256',f.package),
                         ('compute_ledger_sha256',f.ledger),('finalizer_sha256',f.script)]:
            self.assertEqual(result[key],sha(path))
        self.assertFalse(result['promotion_eligible'])
    def test_optimized_mode_fails_closed_at_entry(self):
        f=self.fixture(optimize=1);f.inp.write_bytes(b'changed input')
        self.rejected(f,ValueError)
    def test_inflight_mutations_fail_closed_after_each_child(self):
        for stage in (1,2):
            for name in ('source','inp','protocol','summary','verification','report_path'):
                with self.subTest(stage=stage,artifact=name):
                    f=self.fixture();old_run=f.module.subprocess.run
                    def mutate(args,*,check):
                        result=old_run(args,check=check)
                        if len(f.calls)==stage:
                            getattr(f,name).write_bytes(b'changed in flight')
                        return result
                    f.module.subprocess.run=mutate
                    with self.assertRaises(ValueError):self.run_main(f)
                    self.assertEqual(len(f.calls),stage)
                    self.assertTrue(f.package.exists())
                    self.assertEqual(f.ledger.exists(),stage==2)
                    self.assertFalse(f.final.exists())
    def test_package_mutation_during_ledger_fails_closed(self):
        f=self.fixture();old_run=f.module.subprocess.run
        def mutate(args,*,check):
            result=old_run(args,check=check)
            if len(f.calls)==2:f.package.write_bytes(b'changed package')
            return result
        f.module.subprocess.run=mutate
        with self.assertRaisesRegex(ValueError,'Package report bytes changed'):self.run_main(f)
        self.assertEqual(len(f.calls),2)
        self.assertTrue(f.ledger.exists())
        self.assertFalse(f.final.exists())

if __name__=='__main__':unittest.main(verbosity=2)
