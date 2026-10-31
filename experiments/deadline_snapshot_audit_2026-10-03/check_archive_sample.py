"""Read one independently hosted Internet Archive index/body; no retries or writes upstream."""
import gzip,hashlib,json,urllib.parse,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
p=next((root/'captures/armband/data/captures/2024-25').glob('GW01-*/manifest.json'))
m=json.loads(p.read_text())
stamp=m['backfill']['wayback_timestamp']
query=urllib.parse.urlencode({'url':'https://fantasy.premierleague.com/api/bootstrap-static/','from':stamp,'to':stamp,'output':'json','fl':'timestamp,original,statuscode,mimetype,digest','filter':'statuscode:200'})
urls={'cdx':'https://web.archive.org/cdx/search/cdx?'+query,'raw':m['backfill']['source']}
records={}
for label,url in urls.items():
 try:
  with urllib.request.urlopen(url,timeout=25) as response:
   body=response.read(3_000_001); status=response.status
  if len(body)>3_000_000: raise ValueError('bounded archive response exceeded')
  dest=root/'inventory'/f'wayback_sample_{label}'
  dest.write_bytes(body)
  record={'url':url,'status':status,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
  if label=='raw':
   original=gzip.decompress((p.parent/'bootstrap-static.json.gz').read_bytes())
   raw=gzip.decompress(body) if body.startswith(b'\x1f\x8b') else body
   record['uncompressed_matches_github_capture']=raw==original
  records[label]=record
 except Exception as exc: records[label]={'url':url,'error':f'{type(exc).__name__}: {exc}'}
(root/'wayback_sample_check.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
