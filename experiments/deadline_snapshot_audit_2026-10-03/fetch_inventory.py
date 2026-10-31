"""Retrieve pinned public archive inventory only; never execute upstream code."""
import hashlib,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
SOURCES={'fplcache':('Randdalf/fplcache','de78720734e8c24954433fbf91bc356188cfc669'),
         'armband':('unsaltedpw/fpl-armband','4cbbe8b206686a6def6751966c640fbc673333a7')}
for label,(repo,revision) in SOURCES.items():
 dest=ROOT/'inventory'/label
 dest.mkdir(parents=True,exist_ok=True)
 url=f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'
 request=urllib.request.Request(url,headers={'User-Agent':'FPL-research-source-audit','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(request,timeout=45) as response: body=response.read(16_000_001)
 if len(body)>16_000_000: raise ValueError('bounded inventory size exceeded')
 tree=json.loads(body)
 if tree.get('truncated'): raise ValueError('incomplete GitHub tree')
 (dest/'tree.json').write_bytes(body)
 entries=tree['tree']
 print(label,revision,len(entries))
 if label=='fplcache':
  candidates=[x for x in entries if x['type']=='blob' and x['path'].startswith(('cache/2023/','cache/2024/'))]
  print('candidate snapshots',len(candidates),'first',candidates[:2],'last',candidates[-2:])
 else:
  candidates=[x for x in entries if x['type']=='blob' and any(s in x['path'] for s in ('captures/2023-24/','captures/2024-25/'))]
  print('candidate captures',len(candidates),'sample',candidates[:6])
 (dest/'retrieval.json').write_text(json.dumps({'repository':repo,'revision':revision,'url':url,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(body).hexdigest(),'entries':len(entries),'candidate_count':len(candidates)},indent=2)+'\n')
