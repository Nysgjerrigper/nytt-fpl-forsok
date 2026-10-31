"""Static parsing of archived Understat JSON assignments; no JS execution or admission.

Values retain their source structure. Event datetime strings have no inferred timezone.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_HTML_BYTES = 2_000_000
_NAMES = ('datesData', 'playersData', 'statisticsData')
_DECLARATION = re.compile(r'\b(?:var|let|const)\s+(datesData|playersData|statisticsData)\b\s*=\s*JSON\s*\.\s*parse\s*\(')
_ESCAPE = {'\\': '\\', "'": "'", '"': '"', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\v'}


def _literal(text: str, start: int) -> tuple[str, int]:
    """Decode a quoted JS literal with an explicit, restricted escape alphabet."""
    quote = text[start]
    result: list[str] = []
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == quote:
            return ''.join(result), i + 1
        if ch in '\r\n':
            raise ValueError('Unescaped newline in JavaScript string literal')
        if ch != '\\':
            result.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(text):
            raise ValueError('Unterminated JavaScript escape')
        escape = text[i]
        if escape in _ESCAPE:
            result.append(_ESCAPE[escape])
            i += 1
        elif escape in ('x', 'u'):
            width = 2 if escape == 'x' else 4
            digits = text[i + 1:i + 1 + width]
            if len(digits) != width or re.fullmatch(r'[0-9a-fA-F]+', digits) is None:
                raise ValueError('Invalid hexadecimal JavaScript escape')
            result.append(chr(int(digits, 16)))
            i += width + 1
        else:
            raise ValueError(f'Unsupported JavaScript escape: {escape!r}')
    raise ValueError('Unterminated JavaScript string literal')


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite JSON numeric value')
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _reject_constant(value: str) -> None:
    raise ValueError(f'Nonfinite JSON constant: {value}')


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently taking their last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON object key: {key}')
        result[key] = value
    return result


def _entries(value: Any, name: str, required: set[str]) -> None:
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list')
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f'{name} entry lacks required keys: {sorted(required)}')
        identifier = item['id']
        if isinstance(identifier, (dict, list)) or identifier is None or isinstance(identifier, bool):
            raise ValueError(f'{name} invalid ID')
        key = str(identifier)
        if key in seen:
            raise ValueError(f'{name} duplicate ID: {identifier}')
        seen.add(key)


def parse_understat_capture(html: bytes | str) -> dict[str, Any]:
    """Extract three static JSON collections from bounded archived HTML.

    No JavaScript is executed. This checks structure only, supplies no external-data
    admission, and preserves raw naive event datetime strings and numeric source values.
    """
    if not isinstance(html, (bytes, str)):
        raise TypeError('HTML must be bytes or str')
    encoded = html if isinstance(html, bytes) else html.encode('utf-8')
    if len(encoded) > MAX_HTML_BYTES:
        raise ValueError('Archived HTML exceeds byte limit')
    text = encoded.decode('utf-8')
    values: dict[str, Any] = {}
    i = 0
    while i < len(text):
        if text.startswith('//', i):
            end = text.find('\n', i + 2)
            i = len(text) if end < 0 else end + 1
            continue
        if text.startswith('/*', i):
            end = text.find('*/', i + 2)
            if end < 0:
                raise ValueError('Unterminated JavaScript comment')
            i = end + 2
            continue
        match = _DECLARATION.match(text, i)
        if match:
            name = match.group(1)
            if name in values:
                raise ValueError(f'Duplicate assignment: {name}')
            j = match.end()
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] != "'":
                raise ValueError(f'{name} requires a single-quoted JSON literal')
            literal, j = _literal(text, j)
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] != ')':
                raise ValueError(f'{name} malformed JSON.parse call')
            try:
                values[name] = json.loads(literal, parse_constant=_reject_constant,
                                         object_pairs_hook=_unique_object)
                _finite(values[name])
            except (json.JSONDecodeError, RecursionError) as exc:
                raise ValueError(f'{name} invalid JSON') from exc
            i = j + 1
            continue
        if text[i] in ('"', "'", '`'):
            # Skip unrelated strings/templates as opaque content. Their escapes are
            # not data; do not interpret interpolation or apparent assignments inside.
            quote = text[i]
            i += 1
            while i < len(text):
                if text[i] == '\\':
                    i += 2
                elif text[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            continue
        i += 1
    missing = set(_NAMES) - values.keys()
    if missing:
        raise ValueError(f'Missing assignments: {sorted(missing)}')
    _entries(values['datesData'], 'datesData', {'id', 'isResult', 'h', 'a', 'goals', 'xG', 'datetime'})
    _entries(values['playersData'], 'playersData', {'id', 'games', 'time', 'xG', 'xA'})
    if not isinstance(values['statisticsData'], dict):
        raise ValueError('statisticsData must be an object')
    for fixture in values['datesData']:
        if type(fixture['isResult']) is not bool:
            raise ValueError('Fixture isResult must be a boolean')
        if not isinstance(fixture['datetime'], str):
            raise ValueError('Fixture datetime must be a source string')
    values['completed_match_count'] = sum(item['isResult'] for item in values['datesData'])
    values['upcoming_match_count'] = len(values['datesData']) - values['completed_match_count']
    return values
