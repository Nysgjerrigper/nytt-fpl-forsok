"""Synthetic-only coverage audit contracts; no real outcomes or priors read."""
import importlib.util,json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('coverage',Path(__file__).resolve().parents[1]/'experiments/registered_cold_outcomes_2026-10-04/audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def priors():
 return pd.DataFrame({'player_id':[10,11,12,10],'origin_gw':[137,137,137,138],
   'position':['MID','DEF','GK','MID'],'team':['A','B','C','A'],'status':['a','i','u','a'],
   'predicted_total_points':[1.,2.,3.,1.], 'prior_std':[2.,3.,4.,2.]})
def raw():
 return pd.DataFrame({'player_id':[10,10,11,10,10],'GW_global':[137,137,137,138,184],
   'fixture':[1,2,3,4,5],'position':['MID','MID','MID','MID',None],
   'team':['A','A','Other','A',None],'total_points':[3.,2.,0.,-1.,np.inf], 'minutes':[60.,30.,0.,90.,np.nan]})
class Coverage(unittest.TestCase):
 def fixture(self):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
  repo=root/'repo';(repo/'fpl').mkdir(parents=True);(repo/'fpl/synthetic.py').write_text('# fixed\n')
  rows=pd.DataFrame({'player_id':range(10,201),'origin_gw':[137]*191,'position':['MID']*191,
                    'team':['A']*191,'status':['a']*191,'predicted_total_points':[1.]*191})
  prior=root/'prior.csv';rows.to_csv(prior,index=False)
  dataset=root/'dataset.csv';raw().to_csv(dataset,index=False)
  reg=root/'registration.json';reg.write_text(json.dumps({'dataset':str(dataset),'dataset_sha256':m.digest(dataset)}))
  audit=root/'audit.json';audit.write_text(json.dumps({'status':'complete','promotion_eligible':False,
    'export_sha256':m.digest(prior),'registration_sha256':m.digest(reg),'covered_player_origin_rows':191}))
  return {'priors_path':prior,'audit_path':audit,'registration_path':reg,'dataset_path':dataset,
          'dataset_sha256':m.digest(dataset),'output_dir':root/'output','repository':repo}
 def test_dgw_zero_absence_identity_and_order(self):
  p=priors();out=m.link_outcomes(p,raw());pd.testing.assert_frame_equal(out[p.columns],p)
  self.assertEqual(out.actual_total_points.tolist()[:2],[5.,0.]);self.assertEqual(out.actual_minutes.tolist()[:2],[90.,0.])
  self.assertEqual(out.actual_fixture_count.tolist()[:2],[2.,1.]);self.assertTrue(out.iloc[1].position_mismatch);self.assertTrue(out.iloc[1].team_mismatch)
  self.assertFalse(out.iloc[2].outcome_observed)
  self.assertTrue(out.iloc[2][['actual_total_points','actual_minutes','actual_fixture_count','position_mismatch','team_mismatch']].isna().all())
  self.assertEqual(out.iloc[3].actual_total_points,-1.)
  counts=m.coverage_counts(out);self.assertEqual(counts['observed_rows'],3);self.assertEqual(counts['unobserved_rows'],1)
  self.assertEqual(counts['positive_minutes_rows'],2);self.assertEqual(counts['positive_points_rows'],1);self.assertEqual(counts['observed_total_points'],4.)
 def test_future_outcomes_and_saved_prior_values_do_not_enter_labels(self):
  out=m.link_outcomes(priors(),raw());changed=raw();changed.loc[4,['minutes','total_points']]=[-999,999]
  pd.testing.assert_frame_equal(out,m.link_outcomes(priors(),changed))
  p=priors();p['predicted_total_points']=9999
  revised=m.link_outcomes(p,raw());pd.testing.assert_frame_equal(out[m.LABEL_COLUMNS],revised[m.LABEL_COLUMNS])
 def test_duplicate_exact_fixture_fails_but_future_duplicate_excluded(self):
  with self.assertRaisesRegex(ValueError,'fixture'):m.link_outcomes(priors(),pd.concat([raw(),raw().iloc[:1]]))
  m.link_outcomes(priors(),pd.concat([raw(),raw().iloc[-1:]]))
 def test_absent_labels_all_missing_not_zero(self):
  out=m.link_outcomes(priors(),raw().iloc[:0]);self.assertEqual(len(out),4)
  self.assertTrue(out[['actual_total_points','actual_minutes','actual_fixture_count']].isna().all().all())
  self.assertIsNone(m.coverage_counts(out)['observed_total_points'])
 def test_invalid_targets_and_prior_keys(self):
  for name,value in [('minutes',np.nan),('total_points',np.inf),('minutes',-1.)]:
   with self.subTest(name=name,value=value):
    r=raw();r.loc[0,name]=value
    with self.assertRaises(ValueError):m.link_outcomes(priors(),r)
  p=priors();p['player_id']=p.player_id.astype(object);p.loc[0,'player_id']=True
  with self.assertRaises(ValueError):m.link_outcomes(p,raw())
  with self.assertRaises(ValueError):m.link_outcomes(pd.concat([priors(),priors().iloc[:1]]),raw())
 def test_registration_exists_before_frozen_parse_and_191_retained(self):
  args=self.fixture();actual=m.read_frozen
  def load(path,sha):
   registration=args['output_dir']/'registration.json';self.assertTrue(registration.exists())
   registered=json.loads(registration.read_text());self.assertEqual(registered['window'],[137,183]);self.assertTrue(registered['coverage_only'])
   self.assertFalse(registered['forecast_scores']);self.assertFalse(registered['model_selection']);self.assertFalse(registered['milp'])
   self.assertTrue((args['output_dir']/'evaluated_sources/fpl/synthetic.py').exists())
   return actual(path,sha)
  with patch.object(m,'read_frozen',side_effect=load):report=m.run(**args)
  self.assertEqual(report['status'],'complete');self.assertEqual(report['counts']['player_origin_rows'],191)
  self.assertEqual(len(pd.read_csv(args['output_dir']/'registered_outcomes.csv')),191)
  for path,sha in report['immutable_input_output_sha256'].items():self.assertEqual(m.digest(Path(path)),sha)
 def test_input_hash_and_existing_output_refusals_before_outcomes(self):
  for artifact in ('priors_path','registration_path','dataset_path'):
   with self.subTest(artifact=artifact):
    args=self.fixture();args[artifact].write_bytes(b'changed')
    with patch.object(m,'read_frozen') as load:
     with self.assertRaises(ValueError):m.run(**args)
     load.assert_not_called()
    self.assertFalse(args['output_dir'].exists())
  args=self.fixture();args['output_dir'].mkdir();sentinel=args['output_dir']/'old';sentinel.write_bytes(b'keep')
  with self.assertRaises(ValueError):m.run(**args)
  self.assertEqual(sentinel.read_bytes(),b'keep')
 def test_inflight_dataset_or_source_drift_incomplete(self):
  for artifact in ('dataset','source'):
   with self.subTest(artifact=artifact):
    args=self.fixture();actual=m.read_frozen
    def mutate(path,sha):
     frame=actual(path,sha)
     target=path if artifact=='dataset' else args['repository']/'fpl/synthetic.py';target.write_bytes(b'changed')
     return frame
    with patch.object(m,'read_frozen',side_effect=mutate):
     with self.assertRaises(ValueError):m.run(**args)
    self.assertEqual(json.loads((args['output_dir']/'incomplete.json').read_text())['status'],'incomplete')
    self.assertFalse((args['output_dir']/'registered_outcomes.csv').exists())
    self.assertFalse((args['output_dir']/'summary.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
