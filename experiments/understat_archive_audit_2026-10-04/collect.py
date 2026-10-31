"""Collect one independently indexed historical Understat HTML payload; no execution."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import gzip
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://web.archive.org/web/20201101041833id_/https://understat.com/team/Liverpool/2020'
CDX_DIGEST = 'ROLN4L5ZQQGBKLVTNMPY5C23R7GAHZUQ'
EXPECTED_SHA256 = 'c185d7a7c599af3bf7d9c2d6ea7caf1af757b66ef730349eaeceba8626aba7da'


def main():
    """Preserve transfer and decoded bytes separately, accepting only indexed bytes."""
    output = ROOT / 'capture.json'
    if output.exists() or (ROOT / 'raw').exists():
        raise ValueError('Preserve existing capture evidence; refuse overwrite')
    request = urllib.request.Request(URL, headers={'User-Agent': 'FPL-Research-Archive-Audit/1.0', 'Accept-Encoding': 'gzip'})
    with urllib.request.urlopen(request, timeout=45) as response:
        transfer = response.read(2_000_001)
        headers = dict(response.headers.items())
        if response.status != 200 or len(transfer) > 2_000_000:
            raise ValueError('Invalid bounded archive response')
        encoding = response.headers.get('Content-Encoding', '').lower()
    if encoding not in ('', 'identity', 'gzip'):
        raise ValueError('Unrecognized archive content encoding')
    decoded = gzip.decompress(transfer) if encoding == 'gzip' else transfer
    if len(decoded) > 2_000_000:
        raise ValueError('Decoded HTML exceeds collection bound')
    b32 = lambda data: base64.b32encode(hashlib.sha1(data).digest()).decode()
    sha256 = lambda data: hashlib.sha256(data).hexdigest()
    if b32(decoded) != CDX_DIGEST or sha256(decoded) != EXPECTED_SHA256:
        raise ValueError('Archive payload differs from independently inspected CDX digest/bytes')
    raw = ROOT / 'raw'; raw.mkdir()
    for name, data in [('transfer.bin', transfer), ('decoded.html', decoded)]:
        with (raw / name).open('xb') as file:
            file.write(data)
    report = {'status': 'verified_capture', 'collected_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_url': 'https://understat.com/team/Liverpool/2020', 'replay_url': URL,
              'cdx_query': 'https://web.archive.org/cdx/search/cdx?url=understat.com/team/Liverpool/2020&output=json&filter=statuscode:200&filter=mimetype:text/html&from=2020&to=2021&fl=timestamp,original,digest,length&collapse=timestamp:6&limit=12',
              'capture_utc': '2020-11-01T04:18:33Z', 'cdx_digest': CDX_DIGEST,
              'content_encoding': encoding, 'response_headers': headers,
              'transfer_bytes': len(transfer), 'transfer_sha1_base32': b32(transfer), 'transfer_sha256': sha256(transfer),
              'decoded_bytes': len(decoded), 'decoded_sha1_base32': b32(decoded), 'decoded_sha256': sha256(decoded),
              'collector_sha256': sha256(Path(__file__).read_bytes()), 'cdx_matches_decoded': True,
              'training_admitted': False, 'limitations': ['One team/season capture only; no complete historical deadline coverage.',
              'Team-page player statistics are season aggregates, not player-match history.',
              'No league/season IDs verified inside records; URL scope only.',
              'Match datetime timezone and exact identity joins require separate validation.',
              'No upstream code executed or fitted.']}
    with output.open('x') as file:
        json.dump(report, file, indent=2, sort_keys=True); file.write('\n')
    print({key: report[key] for key in ['status', 'decoded_bytes', 'decoded_sha256', 'training_admitted']})


if __name__ == '__main__':
    main()
