"""Preserve independent index metadata for the one collected Understat capture."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://web.archive.org/cdx/search/cdx?url=understat.com/team/Liverpool/2020&output=json&filter=statuscode:200&filter=mimetype:text/html&from=20201101041833&to=20201101041833&fl=timestamp,original,digest,length,statuscode,mimetype'


def main():
    """Bind the replay digest and immutable capture clock to raw archive index bytes."""
    output = ROOT / 'cdx_verification.json'; raw = ROOT / 'raw/cdx.json'
    if output.exists() or raw.exists():
        raise ValueError('Preserve existing metadata verification')
    capture_bytes = (ROOT / 'capture.json').read_bytes(); capture = json.loads(capture_bytes)
    with urllib.request.urlopen(urllib.request.Request(URL, headers={'User-Agent': 'FPL-Research-Archive-Audit/1.0'}), timeout=45) as response:
        payload = response.read(200_001)
        if response.status != 200 or len(payload) > 200_000:
            raise ValueError('Invalid bounded CDX response')
    table = json.loads(payload)
    if not isinstance(table, list) or len(table) < 2:
        raise ValueError('CDX lacks the expected capture')
    records = [dict(zip(table[0], row, strict=True)) for row in table[1:]]
    matches = [row for row in records if row.get('timestamp') == '20201101041833' and row.get('digest') == capture['cdx_digest']
               and row.get('original') in ['https://understat.com/team/Liverpool/2020', 'http://understat.com/team/Liverpool/2020']
               and row.get('statuscode') == '200' and row.get('mimetype') == 'text/html']
    if not matches:
        raise ValueError('Replay payload lacks independently matching CDX identity/digest')
    decoded = (ROOT / 'raw/decoded.html').read_bytes()
    if hashlib.sha256(decoded).hexdigest() != capture['decoded_sha256']:
        raise ValueError('Decoded capture bytes changed')
    with raw.open('xb') as file:
        file.write(payload)
    report = {'status': 'verified', 'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'query_url': URL,
              'cdx_sha256': hashlib.sha256(payload).hexdigest(), 'capture_report_sha256': hashlib.sha256(capture_bytes).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'matched_records': matches,
              'training_admitted': False, 'coverage': 'One historical team-page capture only.'}
    with output.open('x') as file:
        json.dump(report, file, indent=2, sort_keys=True); file.write('\n')
    print({'status': report['status'], 'matched_records': len(matches), 'training_admitted': False})


if __name__ == '__main__':
    main()
