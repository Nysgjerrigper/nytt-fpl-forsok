from pathlib import Path
import json,hashlib,time,traceback,sys
import pandas as pd
from fpl.milp.origin_prices import load_origin_captures
from fpl.milp.solio import run_schedule
from fpl.milp.optimize import write_decision_snapshot
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((root/'plan.json').read_text())
for name,h in plan['inputs'].items():
 assert sha(root/name)==h, name
for name,h in plan['code_sha256'].items():
 assert sha(Path(name))==h, name
out=root/'first';out.mkdir(exist_ok=False)
t0=time.monotonic()
try:
 caps=load_origin_captures(root/'captures/captures.json',season='2023-24',origins=list(range(137,153)))
 for arm in ['control','candidate']:
  assert time.monotonic()-t0 < 2100
  frame=pd.read_csv(root/f'{arm}-predictions.csv')
  frame=frame.drop(columns=[x for x in frame if x.startswith('actual_') or x=='minutes'])
  frame.to_csv(out/f'{arm}-decision-input.csv',index=False)
  decisions=run_schedule(frame,season='2023-24',start_gw=137,max_gw=152,horizon=3,bank=1000,free_transfers=1,chip_inventory=None,forced_chips=None,time_limit=60,origin_price_captures=caps)
  write_decision_snapshot(decisions,out/f'{arm}-decisions.json')
  decisions.to_csv(out/f'{arm}-decisions.csv',index=False)
  assert len(decisions)==16
  for row in decisions.to_dict('records'):
   assert set(row['squad']) <= set(caps[row['gameweek']]['metadata'].index)
  print(arm,'completed',len(decisions),'decisions',flush=True)
 result={'status':'DECISIONS_COMPLETED_UNSETTLED','elapsed_seconds':time.monotonic()-t0,'model_fits':0,'decisions':32,'plan_sha256':sha(root/'plan.json')}
except BaseException as e:
 result={'status':'FAILED','elapsed_seconds':time.monotonic()-t0,'error':repr(e),'traceback':traceback.format_exc(),'plan_sha256':sha(root/'plan.json')}
 (out/'failed.json').write_text(json.dumps(result,indent=2)+'\n')
 raise
else:
 result['output_sha256']={str(x.relative_to(out)):sha(x) for x in sorted(out.rglob('*')) if x.is_file()}
 (out/'decisions-complete.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result),flush=True)
