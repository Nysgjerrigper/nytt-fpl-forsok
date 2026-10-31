"""Synthetic EPL-only workload checks; no source publication or model claims."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('workload',Path(__file__).resolve().parents[1]/'experiments/league_workload_2026-10-04/league_workload.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def data():
 return pd.DataFrame({'player_id':[1,1,1,1,2,1,2], 'team':['A','A','A','A','A','A','B'],
  'GW_global':[1,2,3,4,4,4,4],'fixture':[11,12,13,14,14,15,16],
  'kickoff_time':['2024-01-01T12:00:00Z','2024-01-08T12:00:00Z','2024-01-15T12:00:00Z',
                  '2024-01-22T12:00:00Z','2024-01-22T12:00:00Z','2024-01-25T12:00:00Z','2024-01-22T12:00:00Z']},index=[9,2,9,1,1,8,1])
class Workload(unittest.TestCase):
 def test_dgw_dedup_index_order_and_exact_boundaries(self):
  raw=data();before=raw.copy();out=m.build_league_workload(raw,4)
  self.assertEqual(out.index.tolist(),raw.index.tolist());self.assertEqual(list(out),m.COLUMNS)
  pd.testing.assert_frame_equal(raw,before)
  self.assertTrue(out.iloc[0].isna().all());self.assertTrue(out.iloc[6].isna().all())
  for pos in (3,4,5):self.assertEqual(out.iloc[pos].tolist(),[7.,1.,2.])
  self.assertEqual(out.iloc[2].tolist(),[7.,1.,2.])
 def test_nonmonotone_round_and_clock_excludes_not_yet_played_and_future_round(self):
  r=pd.DataFrame({'player_id':[1]*4,'team':['A']*4,'GW_global':[1,2,3,4], 'fixture':[1,2,3,4],
    'kickoff_time':['2024-01-01T12:00:00Z','2024-02-01T12:00:00Z','2024-01-15T12:00:00Z','2024-01-10T12:00:00Z']})
  out=m.build_league_workload(r,4)
  self.assertEqual(out.iloc[2].tolist(),[14.,0.,1.]);self.assertEqual(out.iloc[3].tolist(),[9.,0.,1.])
 def test_same_clock_prior_round_is_not_past_history(self):
  r=data().iloc[:2].copy();r.iloc[1,r.columns.get_loc('kickoff_time')]=r.iloc[0].kickoff_time
  self.assertTrue(m.build_league_workload(r,2).iloc[1].isna().all())
 def test_fractional_days_and_zero_recent_counts(self):
  r=data().iloc[:2].copy();r.iloc[1,r.columns.get_loc('kickoff_time')]='2024-01-16T00:00:00Z'
  self.assertEqual(m.build_league_workload(r,2).iloc[1].tolist(),[14.5,0.,0.])
 def test_future_timestamp_perturbation_capped_before_parse(self):
  r=data().astype(object);original=m.build_league_workload(r,3)
  r.loc[r.GW_global.gt(3),'kickoff_time']=[None,'not a timestamp',np.inf,'2024-01-22T12:00:00']
  pd.testing.assert_frame_equal(original,m.build_league_workload(r,3))
 def test_conflicting_team_fixture_kickoffs_rejected(self):
  r=data();r.iloc[4,r.columns.get_loc('kickoff_time')]='2024-01-22T13:00:00Z'
  with self.assertRaisesRegex(ValueError,'Conflicting'):m.build_league_workload(r,4)
 def test_naive_nonfinite_and_invalid_clocks(self):
  for bad in (None,np.inf,'2024-01-01','2024-01-01T12:00:00','2024-13-01T12:00:00Z'):
   with self.subTest(bad=bad):
    r=data().astype(object);r.iloc[0,r.columns.get_loc('kickoff_time')]=bad
    with self.assertRaisesRegex(ValueError,'timezone'):m.build_league_workload(r,4)
 def test_invalid_identity_team_bound_and_schema(self):
  for column,bad in [('player_id',True),('player_id',0),('GW_global',False),('GW_global',1.5),('fixture',np.nan),('team',' '),('team',None),('team',9)]:
   with self.subTest(column=column,bad=bad):
    r=data().astype(object);r.iloc[0,r.columns.get_loc(column)]=bad
    with self.assertRaises(ValueError):m.build_league_workload(r,4)
  for maximum in (0,True,1.5):
   with self.assertRaises(ValueError):m.build_league_workload(data(),maximum)
  with self.assertRaises(ValueError):m.build_league_workload(data().drop(columns='fixture'),4)
 def test_timezone_normalization_and_empty_cap(self):
  r=data();r.iloc[1,r.columns.get_loc('kickoff_time')]='2024-01-08T14:00:00+02:00'
  pd.testing.assert_frame_equal(m.build_league_workload(data(),4),m.build_league_workload(r,4))
  out=m.build_league_workload(data().iloc[1:],1);self.assertTrue(out.empty)
  self.assertEqual(list(out),m.COLUMNS)
if __name__=='__main__':unittest.main(verbosity=2)
