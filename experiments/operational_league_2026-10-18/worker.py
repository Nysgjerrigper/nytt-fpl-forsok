"""Generate operational reports from preserved artifacts; never solve, fit or score."""
from pathlib import Path
import json
from fpl.artifacts import sha256_file,write_json_new
from fpl.operational_reports import import_operations,read_operations,render_operations
from fpl.playground import league,render_league
r=Path('/Users/perandersmolberg/.codex/visualizations/2026/10/17/01a1493e-578c-7060-900d-b428843d8b30/OperationalLeagueReporting_20261018')
rows=[]
for slug in ['ownership-control','ownership-candidate','gw139-diagnostic']:
 path=import_operations(r/f'{slug}.json',runs_dir=r/'runs')
 card=read_operations(path)
 assert (path/'REPORT.md').read_text()==render_operations(card)
 assert import_operations(r/f'{slug}.json',runs_dir=r/'runs')==path
 rows.append({'id':slug,'directory':str(path),'status':card['operations']['status'],'counts':card['operations']['counts'],'acceptance':card['operations'].get('solver_result_accepted'),'report_receipt_sha256':sha256_file(path/'operational-complete.json')})
groups=league(r/'runs')
assert len(groups)==3
for group in groups:
 for row in group['entries']:
  assert all(row[k] is None for k in ['rank','net_or_gross_points','delta','ci'])
write_json_new(r/'league.json',groups)
(r/'LEAGUE.md').write_text(render_league(groups))
write_json_new(r/'verification.json',{'schema_version':1,'source':'PRESERVED_ARTIFACT_REPORTING','entries':rows,'report_reuse_verified':True,'human_machine_agreement':True,'all_entries_unranked':True,'model_fits':0,'solver_calls':0,'settlement_calls':0,'promotion_eligible':False})
print(json.dumps(rows,indent=2))
