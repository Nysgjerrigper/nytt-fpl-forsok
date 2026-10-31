"""Synthetic static extraction regression tests; no external data or execution."""
import json
import random

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/understat_archive_audit_2026-10-04"))

from understat_capture_parser import MAX_HTML_BYTES, parse_understat_capture


def fixture(**changes):
    return dict(id='10', isResult=True, h={'title': 'Home'}, a={'title': 'Away'},
                goals={'h': '1', 'a': '0'}, xG={'h': '1.25', 'a': '0.5'},
                datetime='2024-08-17 15:00:00', **changes)


def payloads():
    return {'datesData': [fixture()], 'playersData': [{'id': '9', 'games': '2', 'time': '120', 'xG': '1.2', 'xA': '0.4'}], 'statisticsData': {'season': '2024'}}


def literal(value):
    return json.dumps(value, ensure_ascii=False).replace('\\', '\\\\').replace("'", "\\'")


def html(values=None):
    return '<html><script>\n' + '\n'.join(f"var {name} = JSON.parse('{literal(value)}');" for name, value in (values or payloads()).items()) + '\n</script></html>'


def test_collections_keep_source_values_datetimes_and_counts():
    values = payloads()
    upcoming = fixture(); upcoming.update(id='11', isResult=False)
    values['datesData'].append(upcoming)
    result = parse_understat_capture(html(values).encode())
    for name in values:
        assert result[name] == values[name]
    assert result['completed_match_count'] == result['upcoming_match_count'] == 1
    assert result['datesData'][0]['datetime'] == '2024-08-17 15:00:00'
    assert result['playersData'][0]['time'] == '120'


@pytest.mark.parametrize('declaration', ['var', 'let', 'const'])
def test_whitespace_and_declarations(declaration):
    source = html().replace('var ', declaration+'\n').replace(' = JSON.parse(', ' \n= JSON \n. parse ( \n')
    assert parse_understat_capture(source)['completed_match_count'] == 1


@pytest.mark.parametrize('value', ["O'Brien", 'ÆΩ雪', '\\', '"quoted"', '\b\f\n\r\t\v', "'); throw Error('boom'); //"])
def test_source_strings_are_data(value):
    values = payloads(); values['statisticsData']['name'] = value
    assert parse_understat_capture(html(values))['statisticsData']['name'] == value


@pytest.mark.parametrize('escape,expected', [(r'\x41', 'A'), (r'\u0041', 'A'), (r'\u96ea', '雪'), (r'\/', '/'), (r"\'", "'"), (r'\v', '\v')])
def test_explicit_escape_decoder(escape, expected):
    # Vertical tab must remain JSON-escaped after the JS decoding layer.
    content = escape if escape != r'\v' else r'\\u000b'
    source = html().replace(literal({'season': '2024'}), '{"name":"'+content+'"}')
    assert parse_understat_capture(source)['statisticsData']['name'] == expected


@pytest.mark.parametrize('bad', [r'\q', r'\0', r'\xG1', r'\x1', r'\uXYZ0', r'\u123', '\\\n'])
def test_invalid_escapes_rejected(bad):
    source = html().replace(literal({'season': '2024'}), '{"name":"'+bad+'"}')
    with pytest.raises(ValueError):
        parse_understat_capture(source)


@pytest.mark.parametrize('source', ["var datesData=JSON.parse('[]", "var datesData=JSON.parse('[]\\", "var datesData=JSON.parse(\"[]\")", "var datesData=JSON.parse('[]', malicious())"])
def test_invalid_literal_call_rejected(source):
    with pytest.raises(ValueError):
        parse_understat_capture(source+'\n'+html())


def test_duplicate_assignment_rejected():
    with pytest.raises(ValueError, match='Duplicate assignment'):
        parse_understat_capture(html()+"const datesData=JSON.parse('[]');")


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', '1e999'])
def test_nonfinite_json_rejected(value):
    with pytest.raises(ValueError, match='Nonfinite'):
        parse_understat_capture(html().replace(literal({'season': '2024'}), '{"bad":'+value+'}'))


@pytest.mark.parametrize('name', ['datesData', 'playersData', 'statisticsData'])
def test_all_assignments_required(name):
    values = payloads(); del values[name]
    with pytest.raises(ValueError, match='Missing assignments'):
        parse_understat_capture(html(values))


@pytest.mark.parametrize('name', ['datesData', 'playersData'])
def test_duplicate_ids_rejected(name):
    values = payloads(); values[name].append(values[name][0].copy())
    with pytest.raises(ValueError, match='duplicate ID'):
        parse_understat_capture(html(values))


@pytest.mark.parametrize('name,key', [('datesData','xG'),('datesData','datetime'),('playersData','games'),('playersData','xA')])
def test_required_schema(name,key):
    values = payloads(); del values[name][0][key]
    with pytest.raises(ValueError, match='required keys'):
        parse_understat_capture(html(values))


@pytest.mark.parametrize('bad', [0, 1, 'true', None])
def test_result_requires_boolean(bad):
    values = payloads(); values['datesData'][0]['isResult'] = bad
    with pytest.raises(ValueError, match='boolean'):
        parse_understat_capture(html(values))


@pytest.mark.parametrize('name,value', [('datesData',{}),('playersData',{}),('statisticsData',[])])
def test_collection_types(name,value):
    values = payloads(); values[name] = value
    with pytest.raises(ValueError):
        parse_understat_capture(html(values))


def test_byte_bound_includes_utf8_multibyte_characters():
    source = html()
    padded = source+' '*(MAX_HTML_BYTES-len(source.encode()))
    assert parse_understat_capture(padded)['completed_match_count'] == 1
    with pytest.raises(ValueError, match='byte limit'):
        parse_understat_capture(padded+'雪')


def test_comment_string_decoys_not_assignments_or_execution(tmp_path):
    path = tmp_path/'executed'
    decoy = "var datesData=JSON.parse('[]');"
    source = f'/* {decoy} */\n// {decoy}\n"{decoy}";\n`{decoy}`;\n' + html()
    source += f";require('fs').writeFileSync('{path}', 'unsafe');"
    assert parse_understat_capture(source)['completed_match_count'] == 1
    assert not path.exists()


def test_deterministic_fuzz_roundtrips_escaped_content():
    rng = random.Random(0)
    alphabet = "abc'\"\\/\n\t\r\v雪Æ"
    for _ in range(80):
        values = payloads()
        values['statisticsData']['name'] = ''.join(rng.choice(alphabet) for _ in range(40))
        assert parse_understat_capture(html(values))['statisticsData'] == values['statisticsData']


@pytest.mark.parametrize('encoded,expected', [(r'\\', '\\'), (r"\'", "'"), (r'\"', '"'), (r'\/', '/'), (r'\b', '\b'), (r'\f', '\f'), (r'\n', '\n'), (r'\r', '\r'), (r'\t', '\t'), (r'\v', '\v'), (r'\x41', 'A'), (r'\u96ea', '雪')])
def test_every_permitted_escape_is_decoded_explicitly(encoded,expected):
    from understat_capture_parser import _literal
    decoded,end = _literal("'"+encoded+"'remaining",0)
    assert decoded == expected
    assert end == len(encoded)+2


def test_unterminated_comment_rejected():
    with pytest.raises(ValueError,match='Unterminated JavaScript comment'):
        parse_understat_capture('/*'+html())


def test_duplicate_json_object_keys_rejected():
    source = html().replace("JSON.parse('{\"season\": \"2024\"}')", "JSON.parse('{\"season\": \"2024\", \"season\": \"2025\"}')")
    with pytest.raises(ValueError, match='Duplicate JSON object key'):
        parse_understat_capture(source)
