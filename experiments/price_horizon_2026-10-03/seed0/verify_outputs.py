"""Independent completed price-proxy artifact, budget, and paired-result checks."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fpl.milp.compare_backtests import compare
from fpl.model.research_package import validate_paired_forecasts
from fpl import experiment
base=Path('experiments/price_horizon_2026-10-03/seed0')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
r=read(base/'registration.json');s=read(base/'summary.json');lock=read(base/'input_lock.json')
assert s['status']=='complete' and not s['promotion_eligible']
assert sha(base/'registration.json')==s['registration_sha256']==lock['registration_sha256']
assert sha(r['dataset'])==r['dataset_sha256']
for p,digest in r['cached_input_sha256'].items(): assert sha(p)==digest,p
for p,digest in s['immutable_output_sha256'].items(): assert sha(p)==digest,p
for p,digest in r['code_sha256'].items(): assert sha(base/'source_archive'/p)==digest,p
raw=pd.read_csv(r['dataset'],usecols=['player_id','GW_global','value'])
raw=raw.drop_duplicates(['player_id','GW_global'])
old=read(Path(r['source_registration']['source_run'])/'summary.json')
oldprice=read(Path(r['source_run'])/'summary.json')
count=0;budgets=0;reports={}; selected_proxy_diagnostics={}
for phase,(start,end) in r['windows'].items():
 cat=pd.read_csv(Path(r['source_run'])/f'{phase}_origin_prices.csv')
 assert sha(Path(r['source_run'])/f'{phase}_origin_prices.csv')==r['origin_price_catalog_sha256'][phase]==lock['origin_price_catalog_sha256'][phase]
 resultset=s['phases'][phase];directory=base/phase
 forecasts={name:pd.read_csv(directory/x['predictions']) for name,x in resultset.items()}
 info=forecasts['mae_horizon'].drop_duplicates('player_id')[['player_id','name']].copy();info['name']=info.name.astype(str).str.strip()
 assert info.name.is_unique and not info.name.str.contains(', ',regex=False).any()
 name_ids=info.set_index('name').player_id.to_dict()
 def decode(value):
  if pd.isna(value) or value=='':return []
  return [int(name_ids[name]) for name in value.split(', ')]
 pool=set(forecasts['mae_horizon'].player_id)
 phase_raw=raw.loc[raw.GW_global<=end]
 catindexed=cat.set_index(['origin_gw','player_id']).sort_index()
 independent=[]
 for origin in range(start,end+1):
  latest=(phase_raw.loc[(phase_raw.GW_global<origin)&phase_raw.player_id.isin(pool)]
          .sort_values('GW_global').groupby('player_id',sort=True).tail(1))
  latest=latest.assign(origin_gw=origin).rename(columns={'GW_global':'price_observed_gw'})
  independent.append(latest[['origin_gw','player_id','value','price_observed_gw']])
 expected=pd.concat(independent).set_index(['origin_gw','player_id']).sort_index()
 pd.testing.assert_frame_equal(catindexed,expected,check_dtype=False)
 for arm,result in resultset.items():
  predpath=directory/result['predictions'];squadpath=directory/result['squads'];prefix=directory/f'{arm}_origin'
  assert sha(predpath)==result['prediction_sha256'] and sha(squadpath)==result['squad_sha256']
  admission=read(prefix.with_suffix('.admission.json'))
  assert admission['origin_price_catalog_sha256']==r['origin_price_catalog_sha256'][phase]
  assert admission['origin_price_policy']==result['origin_price_policy']
  assert admission['cached_replay'] and admission['feature_columns']==r['feature_columns']
  variant,model,model_prefix=r['arms'][arm];policy='origin-current-proxy'
  assert admission['parameters']==r['parameters'][model]
  source_result=old['selection'][variant] if phase=='selection' else old['evaluation'][variant]['origin']
  oldpath=Path(r['source_registration']['source_run'])/('selection' if phase=='selection' else 'evaluation')/source_result['predictions']
  original=pd.read_csv(oldpath);original=original.loc[original.GW.between(start,end)&original.origin_gw.between(start,end)].reset_index(drop=True)
  pd.testing.assert_frame_equal(forecasts[arm].reset_index(drop=True),original,check_exact=False,rtol=1e-14,atol=1e-14)
  replay=model_prefix+'_replay'
  assert result['prediction_sha256']==oldprice['phases'][phase][replay]['prediction_sha256']
  assert result['prediction_sha256']==oldprice['phases'][phase][model_prefix+'_asof']['prediction_sha256']
  scored=pd.read_csv(squadpath)
  assert list(scored.gameweek)==list(range(start,end+1))
  assert (scored.scoring_protocol=='autosubs_v2').all() and (scored.points_accounting=='net_after_transfer_hits').all()
  assert (scored.origin_price_policy==policy).all() and (scored.price_eligibility=='strictly_prior_observed_price').all()
  assert np.array_equal(scored.actual_total_points.to_numpy(),(scored.gross_actual_total_points-4*scored.alpha).to_numpy())
  assert scored.actual_total_points.sum()==result['total_points']
  assert scored.gross_actual_total_points.sum()==result['gross_points']
  assert 4*scored.alpha.sum()==result['transfer_hit_points']
  optimal=prefix.with_suffix('.log').read_text().count('Status: Optimal (');assert optimal==end-start+1;count+=optimal
  # Independent budget reconstruction uses only current-round prices (future
  # prices may influence planning; the separate synthetic tests cover invariance).
  replayprices=(forecasts[arm].drop_duplicates(['player_id','GW']).pivot(index='player_id',columns='GW',values='value').ffill(axis=1).bfill(axis=1).fillna(50.))
  bank=1000.;selected_ages=[];absent_horizon=0;absent_current=0
  for row in scored.itertuples():
   gw=int(row.gameweek);known=set(cat.loc[cat.origin_gw==gw].player_id)
   squad=decode(row.squad);incoming=decode(row.transfers_in);outgoing=decode(row.transfers_out)
   observed=cat.loc[cat.origin_gw==gw].set_index('player_id').price_observed_gw
   selected_ages.extend(gw-int(observed[p]) for p in squad)
   origin_rows=forecasts[arm].loc[forecasts[arm].origin_gw==gw]
   absent_horizon+=len(set(squad)-set(origin_rows.player_id))
   absent_current+=len(set(squad)-set(origin_rows.loc[origin_rows.GW==gw].player_id))
   assert set(squad).issubset(known) and set(incoming).issubset(known) and set(outgoing).issubset(known)
   prices=cat.loc[cat.origin_gw==gw].set_index('player_id').value if policy=='origin-asof' else replayprices[gw]
   expected_bank=(1000.-sum(prices[p] for p in squad) if gw==start else bank+sum(prices[p] for p in outgoing)-sum(prices[p] for p in incoming))
   assert np.isclose(row.budget_end,expected_bank,atol=1e-7), (phase,arm,gw,row.budget_end,expected_bank)
   bank=float(row.budget_end);budgets+=1
  selected_proxy_diagnostics[f'{phase}/{arm}']={'selected_player_rounds':len(selected_ages),'eligibility_observation_age_mean':float(np.mean(selected_ages)),'eligibility_observation_age_max':max(selected_ages),'older_than_38_rounds':sum(age>38 for age in selected_ages),'absent_horizon_fixture_player_rounds':absent_horizon,'absent_current_fixture_player_rounds':absent_current}
 for label,a,b in [('mae_horizon_vs_replay','mae_horizon','mae_replay'),('rmse_horizon_vs_replay','rmse_horizon','rmse_replay'),('mae_horizon_vs_asof','mae_horizon','mae_asof'),('rmse_horizon_vs_asof','rmse_horizon','rmse_asof'),('rmse_horizon_vs_mae_horizon','rmse_horizon','mae_horizon')]:
  aa=resultset[a];bb=resultset[b] if b in resultset else oldprice['phases'][phase][b]
  control_dir=directory if b in resultset else Path(r['source_run'])/phase
  validate_paired_forecasts(directory/aa['predictions'],control_dir/bb['predictions'])
  actual=compare(str(directory/aa['squads']),str(control_dir/bb['squads']),**r['comparison'])
  assert actual==s['comparisons'][phase][label]
 reports[phase]={name:x['total_points'] for name,x in resultset.items()}
assert count==94 and budgets==94
report={'status':'verified','verified_at_utc':datetime.now(timezone.utc).isoformat(),'summary_sha256':sha(base/'summary.json'),'registration_sha256':sha(base/'registration.json'),'source_files':len(r['code_sha256']),'immutable_outputs':len(s['immutable_output_sha256']),'optimal_solves':count,'independently_reconstructed_current_budgets':budgets,'results':reports,'selected_proxy_diagnostics':selected_proxy_diagnostics,'elapsed_seconds':s['elapsed_seconds'],'checks':['original/current archive, frozen data, source and catalog locks','catalog latest prior observations rebuilt independently','unchanged numerical cached forecasts and identical policy replay CSVs','all output/admission/result/log hashes','known-price eligibility and current budgets','net gross hit arithmetic','all paired bootstrap/sign-test reports'],'limitations':r['limitations']+['Horizon freeze holds the reconstructed current vector fixed; current publication, roster and filling gaps remain.'],'promotion_eligible':False}
(base/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True));(base/'verify_outputs.py').write_bytes(Path(__file__).read_bytes())
runtime=read(base/'runtime.json');runtime.update(classification='completed: exec session1528 observed exit_code0',classified_at_utc=datetime.now(timezone.utc).isoformat(),training_jobs=0,solver_running=False)
(base/'runtime.json').write_text(json.dumps(runtime,indent=2,sort_keys=True))
experiment.log_result('horizon_price_research_verification',{'summary_sha256':report['summary_sha256']},{'status':'VERIFIED','promotion_eligible':False,'optimal_solves':count})
print(json.dumps({'optimal':count,'budget_checks':budgets,'points':reports,'elapsed':s['elapsed_seconds']},indent=2))
