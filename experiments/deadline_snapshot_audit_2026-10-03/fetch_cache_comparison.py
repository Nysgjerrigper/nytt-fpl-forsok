"""Retrieve independent repository captures using conservative pre-deadline minute bounds."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import hashlib,json,re,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO='Randdalf/fplcache'
REV='de78720734e8c24954433fbf91bc356188cfc669'
entries=json.loads((ROOT/'inventory/fplcache/tree.json').read_text())['tree']
items=[]
for item in entries:
 match=re.fullmatch(r'cache/(\d{4})/(\d+)/(\d+)/(\d{2})(\d{2})\.json\.xz',item['path'])
 if item['type']!='blob' or not match: continue
 year,month,day,hour,minute=map(int,match.groups())
 if year not in (2024,2025): continue
 at=datetime(year,month,day,hour,minute,tzinfo=timezone.utc)
 items.append((at,item))
items.sort(key=lambda item:item[0])
selected=[]
for season,first,last in [('2023-24',23,38),('2024-25',1,31)]:
 calendar=json.loads((ROOT/f'captures/armband/data/captures/{season}/deadlines.json').read_text())
 for gw in range(first,last+1):
  deadline=datetime.fromisoformat(calendar['deadlines'][str(gw)].replace('Z','+00:00'))
  candidates=[(at,item) for at,item in items if at+timedelta(minutes=1)<=deadline and deadline-at<timedelta(days=1)]
  if not candidates: raise ValueError(f'no bounded candidate for{season}/GW{gw}')
  at,item=candidates[-1]
  selected.append({'season':season,'event':gw,'deadline':deadline.isoformat(),'filename_clock_assuming_utc':at.isoformat(),'item':item})
(ROOT/'cache_comparison_selection.json').write_text(json.dumps({'clock_assumption':'Naive datetime.today on GitHub Ubuntu runner interpreted as UTC; not independently proven by filenames. Minute upper bound is used.','repository':REPO,'revision':REV,'captures':selected},indent=2)+'\n')

def get(record):
 item=record['item']; url=f'https://raw.githubusercontent.com/{REPO}/{REV}/{item["path"]}'
 with urllib.request.urlopen(url,timeout=45) as response: body=response.read(2_000_001)
 if len(body)>2_000_000: raise ValueError('bounded size exceeded')
 gitsha=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
 if gitsha!=item['sha']: raise ValueError('Git blob mismatch')
 path=ROOT/'captures/fplcache'/item['path']; path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.read_bytes()!=body: raise ValueError('refuse overwritten bytes')
 path.write_bytes(body)
 return {**record,'url':url,'sha256':hashlib.sha256(body).hexdigest(),'local_path':str(path.relative_to(ROOT))}
with ThreadPoolExecutor(max_workers=6) as pool: files=list(pool.map(get,selected))
(ROOT/'cache_comparison_downloads.json').write_text(json.dumps({'captures':files},indent=2)+'\n')
print('Retrieved',len(files),'bounded independent repository candidates')
