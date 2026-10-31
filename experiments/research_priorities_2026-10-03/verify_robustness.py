import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from fpl.milp.compare_backtests import compare
from fpl.model.research_package import validate_paired_forecasts, select_ablation
from fpl import experiment
base=Path('experiments/research_priorities_2026-10-03')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
parent=read(base/'robustness_summary.json');protocol=read(base/'robustness_protocol.json')
assert parent['status']=='complete' and parent['protocol_sha256']==sha(base/'robustness_protocol.json')
assert sha(base/'robustness_launcher.py')==protocol['launcher_sha256']
assert sha(protocol['dataset'])==protocol['dataset_sha256']
records={}
for seed in (1,2):
 d=base/f'selection_seed{seed}';s=read(d/'summary.json');r=read(d/'registration.json');lock=read(d/'selection_lock.json')
 assert s['status']=='complete' and not s['promotion_eligible']
 assert sha(d/'summary.json')==parent['seeds'][str(seed)]['summary_sha256']
 assert sha(d/'registration.json')==s['registration_sha256']
 assert r['seed']==seed and r['parameters']==protocol['parameters'][str(seed)]
 for k in ['code_sha256','dataset_sha256','feature_variants','cutoffs','tuned_artifact_sha256','horizon','retrain_every','gap_rel','chips','scoring_protocol','points_accounting']:
  assert r[k]==protocol[k],k
 for path,digest in s['immutable_output_sha256'].items(): assert sha(d/path)==digest,path
 for path,digest in r['code_sha256'].items(): assert sha(d/r['source_archive']/path)==digest,path
 for path,digest in r['tuned_artifact_sha256'].items(): assert sha(path)==digest,path
 assert sha(d/lock['selection_summary'])==lock['selection_summary_sha256']
 assert lock['selected_ablation']==s['selected_ablation']==select_ablation(s['selection'])
 solves=0
 def verify(result,n,start,phase):
  global solves
  pred=d/phase/result['predictions'];squad=d/phase/result['squads'];prefix=str(pred).replace('.predictions.csv','')
  assert sha(pred)==result['prediction_sha256'] and sha(squad)==result['squad_sha256']
  adm=read(prefix+'.admission.json');assert adm['seed']==seed
  model='catboost_rmse' if result['variant']=='rmse_full' else 'catboost_mae'
  assert adm['parameters']==r['parameters'][model]
  assert adm['feature_columns']==r['feature_variants'][result['variant']]
  df=pd.read_csv(squad);assert list(df.gameweek)==list(range(start,start+n))
  assert (df.scoring_protocol=='autosubs_v2').all() and (df.points_accounting=='net_after_transfer_hits').all()
  assert np.array_equal(df.actual_total_points.to_numpy(),(df.gross_actual_total_points-4*df.alpha).to_numpy())
  assert float(df.actual_total_points.sum())==result['total_points']
  assert float(df.gross_actual_total_points.sum())==result['gross_points']
  assert float(4*df.alpha.sum())==result['transfer_hit_points']
  count=Path(prefix+'.log').read_text().count('Status: Optimal (');assert count==n
  solves+=count
 for res in s['selection'].values():verify(res,16,137,'selection')
 for name,paths in s['evaluation'].items():
  for mode,res in paths.items():verify(res,31,153,'evaluation')
 for candidate,modes in s['comparisons'].items():
  for mode,expected in modes.items():
   a=s['evaluation'][candidate][mode];b=s['evaluation']['mae_full'][mode]
   validate_paired_forecasts(d/'evaluation'/a['predictions'],d/'evaluation'/b['predictions'])
   actual=compare(str(d/'evaluation'/a['squads']),str(d/'evaluation'/b['squads']),seed=seed)
   assert actual==expected,(seed,candidate,mode)
 assert solves==282
 report={'status':'verified','verified_at_utc':datetime.now(timezone.utc).isoformat(),'summary_sha256':sha(d/'summary.json'),'registration_sha256':sha(d/'registration.json'),'source_files':len(r['code_sha256']),'immutable_outputs':len(s['immutable_output_sha256']),'optimal_solves':solves,'checks':['frozen full bytes','parent preregistration and launcher','all archived sources','all tuned parameters','selection lock and earlier-only ranking','all scored output hashes','predicted metadata and paired universe','net gross hits arithmetic','optimal solve counts','recomputed full paired bootstrap/sign-test reports'],'promotion_eligible':False}
 (d/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True));records[str(seed)]=report
 experiment.log_result('research_package_seed_verification',{'seed':seed,'summary_sha256':report['summary_sha256']},{'status':'VERIFIED','promotion_eligible':False,'optimal_solves':solves})
 print(seed,report['optimal_solves'],report['source_files'],report['immutable_outputs'])
report={'status':'verified','protocol_sha256':sha(base/'robustness_protocol.json'),'summary_sha256':sha(base/'robustness_summary.json'),'elapsed_seconds':parent['elapsed_seconds'],'seeds':records,'promotion_eligible':False}
(base/'robustness_verification.json').write_text(json.dumps(report,indent=2,sort_keys=True))
(base/'verify_robustness.py').write_bytes(Path(__file__).read_bytes())
runtime=read(base/'robustness_runtime.json');runtime.update(classification_at_recording='completed: exec session 84806 observed exit_code 0',classified_at_utc=datetime.now(timezone.utc).isoformat(),seed1_status='complete',seed2_status='complete',training_running=False)
(base/'robustness_runtime.json').write_text(json.dumps(runtime,indent=2,sort_keys=True))
