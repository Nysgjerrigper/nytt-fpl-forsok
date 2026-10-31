"""Retrieve discovery-window archive candidates only, pinned to audited upstream bytes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,re,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO='unsaltedpw/fpl-armband'; REV='4cbbe8b206686a6def6751966c640fbc673333a7'
items=json.loads((ROOT/'inventory/armband/tree.json').read_text())['tree']
jobs=[]
for item in items:
 path=item['path']
 if item['type']!='blob': continue
 match=re.fullmatch(r'data/captures/(2020-21|2021-22|2022-23|2023-24)/GW(\d+)-[^/]+/(bootstrap-static.json.gz|manifest.json)',path)
 wanted=match and (match[1]!='2023-24' or int(match[2])<=22)
 calendar=path in {f'data/captures/{s}/deadlines.json' for s in ['2020-21','2021-22','2022-23']}
 if wanted or calendar: jobs.append(item)

def get(item):
 url=f'https://raw.githubusercontent.com/{REPO}/{REV}/{item["path"]}'
 with urllib.request.urlopen(url,timeout=45) as response: body=response.read(2_000_001)
 if len(body)>2_000_000: raise ValueError('bounded file size exceeded')
 sha=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
 if sha!=item['sha']: raise ValueError('Git blob mismatch')
 dest=ROOT/'captures/armband'/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and dest.read_bytes()!=body: raise ValueError('preserve original bytes')
 dest.write_bytes(body)
 return {'path':item['path'],'url':url,'local_path':str(dest.relative_to(ROOT)),'git_blob_sha':sha,'sha256':hashlib.sha256(body).hexdigest(),'size':len(body)}
with ThreadPoolExecutor(max_workers=6) as pool: records=list(pool.map(get,jobs))
(ROOT/'discovery_download_manifest.json').write_text(json.dumps({'repository':REPO,'revision':REV,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'files':records,'scope':'Discovery<=GW136 archive candidates only; no fitting/evaluation.'},indent=2,sort_keys=True)+'\n')
print('Downloaded',len(records),'discovery/calendar files,',sum(x['size'] for x in records),'bytes')
