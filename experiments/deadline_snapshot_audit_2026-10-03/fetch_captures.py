"""Bounded retrieval of pinned pre-deadline candidates, excluding spent/prospective outcomes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,re,urllib.request
from pathlib import Path
SOURCES={'fplcache':('Randdalf/fplcache','de78720734e8c24954433fbf91bc356188cfc669'),
         'armband':('unsaltedpw/fpl-armband','4cbbe8b206686a6def6751966c640fbc673333a7')}

ROOT=Path(__file__).resolve().parent
jobs=[]
for label,(repo,revision) in SOURCES.items():
 tree=json.loads((ROOT/'inventory'/label/'tree.json').read_text())['tree']
 for item in tree:
  path=item['path']
  if item['type']!='blob': continue
  documents=(path in {'README.md','LICENSE','cache.py','.github/workflows/cache.yml','docs/backfill.md'} or (label=='armband' and path.startswith(('internal/capture/','internal/backfill/','internal/wayback/')) and path.endswith('.go')))
  calendar=label=='armband' and path in {'data/captures/2023-24/deadlines.json','data/captures/2024-25/deadlines.json'}
  match=re.fullmatch(r'data/captures/(2023-24|2024-25)/GW(\d+)-[^/]+/(bootstrap-static.json.gz|manifest.json)',path)
  capture=match and ((match[1]=='2023-24' and 23<=int(match[2])<=38) or (match[1]=='2024-25' and 1<=int(match[2])<=31))
  if not (documents or calendar or capture): continue
  destination=ROOT/('source_documents' if documents else 'captures')/label/path
  jobs.append((repo,revision,item,destination))

def get(job):
 repo,revision,item,dest=job
 url=f'https://raw.githubusercontent.com/{repo}/{revision}/{item["path"]}'
 with urllib.request.urlopen(url,timeout=45) as response: body=response.read(2_000_001)
 if len(body)>2_000_000: raise ValueError('bounded per-file download exceeded')
 gitsha=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
 if gitsha!=item['sha']: raise ValueError('Git blob identity mismatch')
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and dest.read_bytes()!=body: raise ValueError('preserved bytes differ; refuse overwrite')
 dest.write_bytes(body)
 return {'repository':repo,'revision':revision,'path':item['path'],'local_path':str(dest.relative_to(ROOT)),'url':url,'git_blob_sha':gitsha,'sha256':hashlib.sha256(body).hexdigest(),'size':len(body)}

with ThreadPoolExecutor(max_workers=6) as pool: records=list(pool.map(get,jobs))
manifest={'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sources':SOURCES,'files':records,'scope':'2023/24 GW23–38 and2024/25 GW1–31 candidate captures; source docs, no model/outcome evaluation.'}
(ROOT/'download_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print('Downloaded and byte-verified',len(records),'files,',sum(x['size'] for x in records),'bytes')
