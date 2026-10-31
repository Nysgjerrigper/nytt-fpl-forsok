"""Synthetic offline archive-chain admission; no external captures or requests."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fpl.data import deadline_snapshots as adapter


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'load_frozen_dataset_manifest', lambda: {'max_gw_global':231})
    monkeypatch.setattr(adapter.config, 'DEFAULT_START_SEASON', '2020-21')
    payload = {'events': [{'id':1, 'deadline_time':'2024-08-16T17:30:00Z', 'finished':True, 'is_next':False},
                          {'id':2, 'deadline_time':'2024-08-24T10:00:00Z', 'finished':False, 'is_next':True}],
               'teams':[{'id':1, 'name':'Spurs'}],
               'element_types':[{'id':i} for i in range(1,5)],
               'elements':[{'id':1, 'code':101, 'team':1, 'element_type':3, 'now_cost':55,
                            'status':'a', 'chance_of_playing_next_round':None, 'web_name':'Player',
                            'total_points':999, 'minutes':999}]}
    paths = {name:tmp_path / name for name in ('proof.json','index.json','manifest.json','raw.payload')}
    def build(payload_change=None, capture_change=None, manifest_change=None, index_change=None, proof_change=None):
        data=copy.deepcopy(payload)
        if payload_change: payload_change(data)
        body=json.dumps(data).encode(); raw=gzip.compress(body,mtime=0)
        paths['raw.payload'].write_bytes(raw)
        digest=base64.b32encode(hashlib.sha1(raw).digest()).decode()
        timestamp='20240824090000'
        source=f'https://web.archive.org/web/{timestamp}id_/https://fantasy.premierleague.com/api/bootstrap-static/'
        capture={'season':'2024-25','event':2,'wayback_timestamp':timestamp,'cdx_digest':digest,
                 'captured_at':'2024-08-24T09:00:00Z','deadline':'2024-08-24T10:00:00Z',
                 'source_url':source,'http_status':200,'headers':{'Content-Type':'application/json','content-encoding':'gzip'},
                 'original_payload':'raw.payload','original_payload_sha256':sha(raw),
                 'body_sha256':sha(body),'manifest':'manifest.json'}
        manifest={'captured_at':capture['captured_at'],'event':2,'event_deadline':capture['deadline'],
                  'files':[{'endpoint':'/bootstrap-static/','bytes':len(body),'sha256':sha(body)}],
                  'backfill':{'season':'2024-25','source':source,'wayback_timestamp':timestamp,'digest':digest}}
        if manifest_change: manifest_change(manifest)
        paths['manifest.json'].write_text(json.dumps(manifest))
        capture['manifest_sha256']=sha(paths['manifest.json'].read_bytes())
        if capture_change: capture_change(capture)
        index=[['timestamp','original','statuscode','mimetype','digest'],
               [timestamp,'https://fantasy.premierleague.com/api/bootstrap-static/','200','application/json',digest]]
        if index_change: index_change(index)
        paths['index.json'].write_text(json.dumps(index))
        proof={'status':'incomplete','index':{'sha256':sha(paths['index.json'].read_bytes())},'captures':[capture]}
        if proof_change: proof_change(proof)
        paths['proof.json'].write_text(json.dumps(proof))
        return sha(paths['proof.json'].read_bytes())
    def load(digest=None, origins=(154,), **kwargs):
        return adapter.load_deadline_rosters(paths['proof.json'],paths['index.json'],digest or build(),origins,**kwargs)
    return paths, build, load


def test_valid_incomplete_subset_preserves_identity_actual_time_and_null(archive):
    _,build,load=archive
    digest=build();frame=load(digest)
    assert len(frame)==1 and frame.player_id.iloc[0]==101 and frame.fpl_element_id.iloc[0]==1
    assert frame.origin_gw.iloc[0]==154 and frame.event.iloc[0]==2
    assert frame.team.iloc[0]=='Tottenham' and frame.position.iloc[0]=='MID'
    assert frame.value.iloc[0]==55 and np.isnan(frame.chance_of_playing_next_round.iloc[0])
    assert frame.snapshot_age_hours.iloc[0]==1. and frame.captured_at.iloc[0]=='2024-08-24T09:00:00Z'
    assert frame.attrs['proof_sha256']==digest and frame.attrs['index_sha256']
    assert 'total_points' not in frame and 'minutes' not in frame and 'price_observed_gw' not in frame


@pytest.mark.parametrize('name',['proof.json','index.json','manifest.json','raw.payload'])
def test_descriptor_byte_tampering_fails_before_parse(archive,name):
    paths,build,load=archive;digest=build()
    paths[name].write_bytes(paths[name].read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        load(digest)


@pytest.mark.parametrize('field',['cdx_digest','body_sha256','original_payload_sha256','manifest_sha256'])
def test_capture_chain_hash_mismatch(archive,field):
    _,build,load=archive
    digest=build(capture_change=lambda c:c.update({field:'X'*64}))
    with pytest.raises(ValueError):load(digest)


@pytest.mark.parametrize('change',[
    lambda d:d['events'][1].update(finished=True),
    lambda d:d['events'][1].update(is_next=False),
    lambda d:d['events'][0].update(is_next=True),
    lambda d:d['events'][1].update(deadline_time='2024-08-24T11:00:00Z'),
    lambda d:d['events'][0].update(deadline_time='2023-08-16T17:30:00Z'),
    lambda d:d['events'].append(copy.deepcopy(d['events'][1])),
    lambda d:d['teams'].append(copy.deepcopy(d['teams'][0])),
    lambda d:d['elements'].append(copy.deepcopy(d['elements'][0])),
    lambda d:d['elements'].append({**d['elements'][0],'id':2}),
    lambda d:d['elements'][0].update(code=True),
    lambda d:d['elements'][0].update(id=False),
    lambda d:d['elements'][0].update(team=2),
    lambda d:d['elements'][0].update(element_type=5),
    lambda d:d['elements'][0].update(now_cost=True),
    lambda d:d['elements'][0].update(now_cost=0),
    lambda d:d['elements'][0].update(chance_of_playing_next_round=35),
    lambda d:d['elements'][0].update(chance_of_playing_next_round=True),
    lambda d:d['elements'][0].update(status='unknown'),
])
def test_payload_semantics_refused_with_valid_byte_chain(archive,change):
    _,build,load=archive;digest=build(payload_change=change)
    with pytest.raises(ValueError):load(digest)


@pytest.mark.parametrize('chance',[0,25,50,75,100,None])
def test_valid_chance_values(archive,chance):
    _,build,load=archive;digest=build(payload_change=lambda d:d['elements'][0].update(chance_of_playing_next_round=chance))
    value=load(digest).chance_of_playing_next_round.iloc[0]
    assert np.isnan(value) if chance is None else value==chance


@pytest.mark.parametrize('field,value',[
    ('captured_at','2024-08-24T09:00:01Z'), ('captured_at','2024-08-24T09:00:00'),
    ('deadline','2024-08-24T09:00:00Z'), ('deadline','2024-08-24T08:00:00Z'),
    ('deadline','2024-08-26T10:00:00Z'), ('season','2024-26'),('event',True),('event',39),
    ('source_url','https://web.archive.org/web/20240824090000/https://fantasy.premierleague.com/api/bootstrap-static/'),
    ('original_payload','../raw.payload'), ('manifest','/tmp/manifest.json'),
])
def test_capture_identity_time_paths_fail_closed(archive,field,value):
    _,build,load=archive;digest=build(capture_change=lambda c:c.update({field:value}))
    with pytest.raises(ValueError):load(digest)


@pytest.mark.parametrize('field,value',[('event',3),('captured_at','2024-08-24T08:00:00Z'),
                                      ('event_deadline','2024-08-24T11:00:00Z')])
def test_manifest_identity_differs(archive,field,value):
    _,build,load=archive;digest=build(manifest_change=lambda m:m.update({field:value}))
    with pytest.raises(ValueError,match='manifest identity'):load(digest)


@pytest.mark.parametrize('change',[lambda m:m['backfill'].update(season='2023-24'),
                                   lambda m:m['files'][0].update(sha256='0'*64),
                                   lambda m:m['files'][0].update(bytes=1)])
def test_source_manifest_chain_mismatch(archive,change):
    _,build,load=archive;digest=build(manifest_change=change)
    with pytest.raises(ValueError):load(digest)


@pytest.mark.parametrize('column,value',[(0,'20240824080000'),(1,'https://example.com/api/bootstrap-static/'),
                                        (2,'404'),(3,'text/html'),(4,'WRONG')])
def test_exact_cdx_record_required(archive,column,value):
    _,build,load=archive;digest=build(index_change=lambda i:i[1].__setitem__(column,value))
    with pytest.raises(ValueError,match='CDX record'):load(digest)


def test_http_original_same_official_resource(archive):
    _,build,load=archive
    digest=build(index_change=lambda i:i[1].__setitem__(1,'http://fantasy.premierleague.com/api/bootstrap-static/'))
    assert len(load(digest))==1


@pytest.mark.parametrize('origins',[(True,), (0,), (154,154), (232,), (), (155,)])
def test_requested_origins_unique_covered_frozen(archive,origins):
    _,build,load=archive;digest=build()
    with pytest.raises(ValueError):load(digest,origins=origins)


def test_duplicate_capture_refused_but_disjoint_future_ignored(archive):
    _,build,load=archive
    digest=build(proof_change=lambda p:p['captures'].append(copy.deepcopy(p['captures'][0])))
    with pytest.raises(ValueError,match='exactly one'):load(digest)
    digest=build(proof_change=lambda p:p['captures'].append({'season':'2026-27','event':4}))
    assert len(load(digest))==1


def test_bounded_decompression(archive,monkeypatch):
    _,build,load=archive;digest=build()
    monkeypatch.setattr(adapter,'MAX_BODY_BYTES',20)
    with pytest.raises(ValueError,match='maximum bytes'):load(digest)


def test_stable_unique_roster_rows_even_with_fixture_like_extra_data(archive):
    _,build,load=archive
    def change(d):
        d['fixtures']=[{'event':2,'id':10},{'event':2,'id':11}]
        d['elements'].append({**d['elements'][0],'id':2,'code':102})
    frame=load(build(payload_change=change))
    assert len(frame)==2 and not frame.duplicated(['player_id','origin_gw']).any()


def test_unavailable_absent_capture_has_per_origin_reason(archive):
    _,build,load=archive
    with pytest.raises(adapter.SnapshotUnavailableError,match='origin 155.*no verified capture') as error:
        load(build(),origins=(155,))
    assert error.value.origin_gw==155 and error.value.reason=='no verified capture'


def test_unavailable_excessive_age_requires_valid_evidence(archive):
    paths,build,load=archive;digest=build()
    with pytest.raises(adapter.SnapshotUnavailableError,match='capture age'):
        load(digest,max_age_hours=.5)
    paths['raw.payload'].write_bytes(b'tampered')
    with pytest.raises(ValueError) as error:
        load(digest,max_age_hours=.5)
    assert not isinstance(error.value,adapter.SnapshotUnavailableError)


def test_unavailable_next_event_earlier_alignment(archive):
    _,build,load=archive
    def change(d):
        d['events'][0]['is_next']=True
        d['events'][1]['is_next']=False
    with pytest.raises(adapter.SnapshotUnavailableError,match='next event 1 precedes requested event 2'):
        load(build(payload_change=change))


@pytest.mark.parametrize('change',[
    lambda d:d['events'][1].update(finished=True),
    lambda d:d['events'][0].update(is_next=True),
    lambda d:d['elements'][0].update(code=True),
])
def test_fatal_evidence_is_never_unavailable(archive,change):
    _,build,load=archive
    with pytest.raises(ValueError) as error:
        load(build(payload_change=change),max_age_hours=.5)
    assert not isinstance(error.value,adapter.SnapshotUnavailableError)


@pytest.mark.parametrize('max_age',[0,-1,float('nan'),float('inf'),True])
def test_invalid_age_bound_fatal(archive,max_age):
    _,build,load=archive
    with pytest.raises(ValueError) as error:load(build(),max_age_hours=max_age)
    assert not isinstance(error.value,adapter.SnapshotUnavailableError)


def test_payload_missing_required_field_is_explicit_fatal(archive):
    _,build,load=archive
    digest=build(payload_change=lambda d:d.pop('elements'))
    with pytest.raises(ValueError,match='Invalid archive evidence') as error:load(digest)
    assert not isinstance(error.value,adapter.SnapshotUnavailableError)


def test_proof_unknown_status_fatal(archive):
    _,build,load=archive
    with pytest.raises(ValueError,match='proof must'):
        load(build(proof_change=lambda p:p.update(status='unverified')))


def test_payload_nonboolean_next_flag_fatal(archive):
    _,build,load=archive
    with pytest.raises(ValueError,match='flags must be boolean'):
        load(build(payload_change=lambda d:d['events'][0].update(is_next=0)))


def _managers(payload, label='Assistant Manager'):
    payload['element_types'].append({'id':5, 'singular_name':label,
                                     'singular_name_short':'AM', 'plural_name':'Assistant Managers'})
    for manager in range(20):
        payload['elements'].append({'id':1000+manager,'code':800000+manager,'team':1,
                                    'element_type':5,'now_cost':15,'status':'a',
                                    'chance_of_playing_next_round':None,'web_name':f'Manager {manager}'})


@pytest.mark.parametrize('label',['Manager','Assistant Manager','assistant MANAGER'])
def test_optional_manager_elements_validated_excluded_and_disclosed(archive,label):
    _,build,load=archive
    frame=load(build(payload_change=lambda d:_managers(d,label)))
    assert frame.player_id.tolist()==[101] and frame.position.tolist()==['MID']
    assert frame.attrs['excluded_nonplayer_count']==20
    assert frame.attrs['excluded_nonplayer_count_by_origin']=={154:20}


@pytest.mark.parametrize('change',[
    lambda d:_managers(d,'Forward'),
    lambda d:d['element_types'].append({'id':6,'singular_name':'Manager'}),
    lambda d:d['element_types'].remove(d['element_types'][0]),
])
def test_unknown_or_mislabelled_extra_type_fatal(archive,change):
    _,build,load=archive
    with pytest.raises(ValueError):load(build(payload_change=change))


@pytest.mark.parametrize('field,value',[('code',101),('id',1),('code',True),('team',9),
                                       ('now_cost',0),('status','x'),('chance_of_playing_next_round',10)])
def test_excluded_manager_still_has_all_roster_validation(archive,field,value):
    _,build,load=archive
    def change(d):
        _managers(d)
        d['elements'][-1][field]=value
    with pytest.raises(ValueError) as error:load(build(payload_change=change))
    assert not isinstance(error.value,adapter.SnapshotUnavailableError)


def test_without_managers_exclusion_counts_zero(archive):
    _,build,load=archive
    frame=load(build())
    assert frame.attrs['excluded_nonplayer_count']==0
    assert frame.attrs['excluded_nonplayer_count_by_origin']=={154:0}
