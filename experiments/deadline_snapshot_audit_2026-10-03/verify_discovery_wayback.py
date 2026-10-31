"""Independently bind repository captures to Internet Archive clock and original bytes."""
from datetime import datetime,timezone
import base64,gzip,hashlib,json,time,urllib.parse,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RESOURCE='https://fantasy.premierleague.com/api/bootstrap-static/'
request_count=0
last_request=0.

def get(url):
 global last_request,request_count
 pause=1.5-(time.monotonic()-last_request)
 if pause>0: time.sleep(pause)
 last_request=time.monotonic()
 request_count+=1
 with urllib.request.urlopen(url,timeout=30) as response:
  body=response.read(3_000_001)
  metadata={'http_status':response.status,'final_url':response.url,'headers':{k:v for k,v in response.headers.items() if k.lower() in ('memento-datetime','content-encoding','x-archive-orig-date','x-archive-orig-content-encoding','content-type')}}
 if len(body)>3_000_000: raise ValueError('bounded archive response exceeded')
 return body,metadata

query=urllib.parse.urlencode({'url':RESOURCE,'from':'20200901','to':'20240201','output':'json','fl':'timestamp,original,statuscode,mimetype,digest','filter':'statuscode:200'})
index_url='https://web.archive.org/cdx/search/cdx?'+query
body,metadata=get(index_url)
(ROOT/'inventory/discovery_wayback_index.json').write_bytes(body)
rows=json.loads(body); columns=rows[0]; indexed={}
for row in rows[1:]:
 record=dict(zip(columns,row)); indexed.setdefault(record['timestamp'],[]).append(record)
output=ROOT/'discovery_wayback_verification.json'
report={'status':'running','index':{'url':index_url,'sha256':hashlib.sha256(body).hexdigest(),'rows':len(rows)-1,**metadata},'captures':[],'attempt_errors':[]}
if output.exists():
 previous=json.loads(output.read_text())
 if previous['index']['sha256']!=report['index']['sha256']: raise ValueError('archive index changed; preserve original audit')
 report['captures']=previous['captures']
 report['attempt_errors']=previous.get('attempt_errors',[])
 for record in report['captures']:
  if hashlib.sha256((ROOT/record['original_payload']).read_bytes()).hexdigest()!=record['original_payload_sha256']: raise ValueError('previous archive proof changed')
  if hashlib.sha256((ROOT/record['manifest']).read_bytes()).hexdigest()!=record['manifest_sha256']: raise ValueError('previous source manifest changed')
completed={(r['season'],r['event']) for r in report['captures']}
failures=[]
for path in sorted((ROOT/'captures/armband/data/captures').glob('*/GW*/manifest.json')):
 manifest=json.loads(path.read_text()); provenance=manifest['backfill']; stamp=provenance['wayback_timestamp']
 if provenance['season'] not in ('2020-21','2021-22','2022-23','2023-24') or (provenance['season']=='2023-24' and manifest['event']>22): continue
 if (provenance['season'],manifest['event']) in completed: continue
 matching=[r for r in indexed.get(stamp,[]) if r['digest']==provenance['digest'] and r['statuscode']=='200']
 if not matching: raise ValueError(f'no independently hosted CDX match for{path}')
 url=provenance['source']
 try: raw,headers=get(url)
 except Exception as exc:
  error={'season':provenance['season'],'event':manifest['event'],'url':url,'error':f'{type(exc).__name__}: {exc}'}
  report['attempt_errors'].append(error); failures.append(error)
  report['status']='incomplete'
  output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print('unverified',provenance['season'],manifest['event'],type(exc).__name__,flush=True)
  continue
 original_path=ROOT/'captures/wayback'/f'{stamp}.payload'
 original_path.parent.mkdir(parents=True,exist_ok=True)
 if original_path.exists() and original_path.read_bytes()!=raw: raise ValueError('archived raw response changed')
 original_path.write_bytes(raw)
 digest=base64.b32encode(hashlib.sha1(raw).digest()).decode().rstrip('=')
 if digest!=provenance['digest']: raise ValueError(f'archive payload digest mismatch for{stamp}')
 uncompressed=gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw
 stored=gzip.decompress((path.parent/'bootstrap-static.json.gz').read_bytes())
 if uncompressed!=stored: raise ValueError(f'archive/repository body mismatch for{stamp}')
 record={'season':provenance['season'],'event':manifest['event'],'captured_at':manifest['captured_at'],'deadline':manifest['event_deadline'],'wayback_timestamp':stamp,'source_url':url,'manifest':str(path.relative_to(ROOT)),'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'original_payload':str(original_path.relative_to(ROOT)),'original_payload_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':hashlib.sha256(uncompressed).hexdigest(),'cdx_digest':digest,'index_records':matching,**headers}
 report['captures'].append(record)
 output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print('verified',provenance['season'],manifest['event'],flush=True)
report.update(status='verified' if not failures else 'incomplete',verified_at_utc=datetime.now(timezone.utc).isoformat(),request_count=request_count)
output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('Archive clock and bytes verified for',len(report['captures']),'captures; current missing',len(failures),flush=True)
