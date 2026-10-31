"""Mocked fixed research orchestration and origin workload alignment tests."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location('staged_workload_driver',Path(__file__).resolve().parents[1]/'experiments/league_workload_2026-10-04/run.py')
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)


def test_attach_preserves_duplicate_index_order_attrs_and_missing_context():
    frame = pd.DataFrame({'team':['B','A','B','C'],'GW_global':[3,3,3,4],'target':[1,2,3,4]},index=[8,8,3,2])
    frame.attrs['source'] = 'admitted'
    context = pd.DataFrame({'team':['A','B'],'GW_global':[3,3],m.COLUMNS[0]:[7.,2.],m.COLUMNS[1]:[1.,2.],m.COLUMNS[2]:[2.,3.]})
    result = m.attach_context(frame,context)
    pd.testing.assert_frame_equal(result[frame.columns],frame)
    assert result.index.equals(frame.index) and result.attrs == frame.attrs
    assert result[m.COLUMNS[0]].iloc[:3].tolist() == [2,7,2]
    assert result.iloc[3][list(m.COLUMNS)].isna().all()


def test_origin_uses_current_team_clock_not_horizon_round():
    context = pd.DataFrame({'team':['A','A'],'GW_global':[3,4],m.COLUMNS[0]:[2.,99.],m.COLUMNS[1]:[1.,99.],m.COLUMNS[2]:[2.,99.]})
    snapshot = pd.DataFrame({'team':['A','A','B'],'GW_global':[3,4,5]})
    result = m.attach_context(snapshot,context,origin=3)
    assert result[m.COLUMNS[0]].iloc[:2].tolist() == [2.,2.]
    assert result.iloc[2][list(m.COLUMNS)].isna().all()


def test_dgw_context_identical_nan_collapses(monkeypatch):
    raw = pd.DataFrame({'team':['A','A'],'GW_global':[3,3]},index=[6,6])
    aligned = pd.DataFrame({column:[np.nan,np.nan] for column in m.COLUMNS},index=raw.index)
    monkeypatch.setattr(m._helper,'build_league_workload',lambda *_:aligned)
    assert len(m.build_context(raw,3)) == 1
    aligned.loc[:,m.COLUMNS[0]] = [1.,2.]
    with pytest.raises(ValueError,match='conflict'):
        m.build_context(raw,3)


def test_context_alignment_and_collision_rejected(monkeypatch):
    raw = pd.DataFrame({'team':['A'],'GW_global':[3]},index=[6])
    monkeypatch.setattr(m._helper,'build_league_workload',lambda *_:pd.DataFrame(columns=m.COLUMNS))
    with pytest.raises(ValueError,match='alignment'): m.build_context(raw,3)
    frame = raw.assign(**{m.COLUMNS[0]:1})
    with pytest.raises(ValueError,match='collision'): m.attach_context(frame,pd.DataFrame())
    ctx = pd.DataFrame({'team':['A','A'],'GW_global':[3,3],**{c:[1.,1.] for c in m.COLUMNS}})
    with pytest.raises(ValueError,match='unique'):m.attach_context(raw,ctx)


@pytest.fixture
def mocked(tmp_path,monkeypatch):
    dataset = tmp_path/'frozen.csv'; dataset.write_text('frozen')
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    monkeypatch.setattr(m.provenance,'FROZEN_MANIFEST_PATH',manifest)
    cutoffs = dict(discovery_max_gw=2,selection_min_gw=3,selection_max_gw=4,backtest_min_gw=5,backtest_max_gw=6)
    monkeypatch.setattr(m,'protocol_cutoffs',lambda:cutoffs)
    codefile = tmp_path/'source.py';codefile.write_text('source')
    code = {'fpl/source.py':m.research_package.file_hash(codefile)}
    monkeypatch.setattr(m.research_package,'source_hashes',lambda:dict(code))
    def archive(directory,expected):
        destination=directory/'fpl/source.py'; destination.parent.mkdir(parents=True); destination.write_bytes(codefile.read_bytes())
    monkeypatch.setattr(m.cnn_research,'_archive_sources',archive)
    params = {'catboost_mae':{'MID':{'catboost_mae':dict(random_seed=0,thread_count=5,depth=6)}}}
    monkeypatch.setattr(m.research_package,'resolve_parameters',lambda *_:(params,{}))
    monkeypatch.setattr(m.experiment,'_library_versions',lambda:{'mock':'1'})
    logs=[];monkeypatch.setattr(m.experiment,'log_result',lambda *args:logs.append(args))
    base=[f'f{i}' for i in range(115)]
    loads=[];calls=[];comparisons=[]
    output=tmp_path/'study'
    def load(end,path):
        assert (output/'registration.json').exists()
        assert logs[0][0]=='league_workload_registration'
        if end==6:assert (output/'selection_lock.json').exists()
        loads.append(end)
        raw=pd.DataFrame({'player_id':[1]*end,'team':['A']*end,'GW_global':range(1,end+1),
            'fixture':range(11,11+end),'value':[50.]*end,'kickoff_time':[f'2024-08-{i:02}T15:00:00Z' for i in range(1,end+1)]})
        raw.attrs['frozen_dataset_sha256']=m.research_package.file_hash(dataset)
        frame=pd.concat([raw.copy(),pd.DataFrame(1.,index=raw.index,columns=base)],axis=1)
        return raw,frame,base
    monkeypatch.setattr(m,'load_research_features',load)
    monkeypatch.setattr(m,'build_live_snapshot',lambda hist:pd.DataFrame({'player_id':[1],'team':['A']}))
    state={'candidate_points':11.,'mutate':None}
    def evaluate(raw,frame,cols,arm,parameters,start,end,origin,directory,seed,**kwargs):
        assert (directory/'columns.json').exists()
        calls.append(dict(arm=arm,columns=cols,parameters=parameters,start=start,end=end,origin=origin,seed=seed,kwargs=kwargs))
        if arm==m.ARMS[1] and origin:
            snapshot=kwargs['origin_snapshot_builder'](raw.loc[raw.GW_global<start],start)
            assert snapshot[m.COLUMNS[0]].iloc[0]==1.
            assert kwargs['extra_player_features']==list(m.COLUMNS)
            missing=kwargs['origin_snapshot_builder'](raw,end+1)
            assert missing[list(m.COLUMNS)].isna().all().all()
        prefix=directory/f'{arm}_{"origin" if origin else "standard"}'
        predictions=prefix.with_suffix('.predictions.csv');squads=prefix.with_suffix('.squads.csv')
        pd.DataFrame({'player_id':[1],'position':['MID'],'GW':[start],'fixture':[11],
             'value':[50.],'minutes':[90.],'actual_total_points':[2.],'predicted_total_points':[3. if arm==m.ARMS[0] else 4.]}).to_csv(predictions,index=False)
        squads.write_text('gameweek,actual_total_points\n3,10\n')
        for suffix in ('.admission.json','.result.json','.log'):prefix.with_suffix(suffix).write_text('{}')
        if state['mutate'] is not None:state['mutate'](len(calls),output,directory)
        return dict(total_points=10. if arm==m.ARMS[0] else state['candidate_points'],predictions=predictions.name,squads=squads.name,
                    prediction_sha256=m.research_package.file_hash(predictions),squad_sha256=m.research_package.file_hash(squads))
    monkeypatch.setattr(m.research_package,'evaluate_variant',evaluate)
    def compare(*args,**kwargs):comparisons.append((args,kwargs));return {'delta':1.,'ci_low':-1.,'ci_high':3.}
    monkeypatch.setattr(m.research_package.compare_backtests,'compare',compare)
    return dict(dataset=dataset,output=output,loads=loads,calls=calls,comparisons=comparisons,state=state,params=params,code=code,logs=logs)


@pytest.mark.parametrize('points',[9.,10.])
def test_selection_rejects_loss_tie_without_reference_loading(mocked,points):
    mocked['state']['candidate_points']=points
    result=m.run(mocked['dataset'],mocked['output'])
    assert result['status']=='selection_rejected'
    assert mocked['loads']==[4] and len(mocked['calls'])==2
    assert result['evaluation']=={}
    assert not (mocked['output']/'reference').exists()
    lock=json.loads((mocked['output']/'selection_lock.json').read_text())
    assert lock['finalists']==[] and not lock['advanced']


def test_positive_gate_namespace_capacity_chronology_and_outputs(mocked):
    result=m.run(mocked['dataset'],mocked['output'])
    assert result['status']=='complete' and mocked['loads']==[4,6]
    assert len(mocked['calls'])==6 and len(mocked['comparisons'])==3
    for call in mocked['calls']:
        assert call['parameters']==mocked['params'] and call['seed']==0
        assert call['kwargs']['model_name']=='catboost_mae' and call['kwargs']['retrain_every']==4
        assert len(call['columns'])==(115 if call['arm']==m.ARMS[0] else 118)
        if call['origin']:assert call['kwargs']['origin_price_policy']=='origin-current-proxy'
        else:assert 'origin_price_catalog' not in call['kwargs'] and 'origin_snapshot_builder' not in call['kwargs']
    for _,kwargs in mocked['comparisons']:assert kwargs==dict(n_boot=10000,block_len=3,seed=0)
    for path,digest in result['immutable_output_sha256'].items():assert m.research_package.file_hash(Path(path))==digest
    assert result['registration_sha256']==m.research_package.file_hash(mocked['output']/'registration.json')


@pytest.mark.parametrize('target',['registration.json','selection/context.csv','source_archive/fpl/source.py','selection/mae_full_origin.predictions.csv','selection/mae_full_origin.admission.json','selection_lock.json'])
def test_mutation_fails_closed(mocked,target):
    def mutate(n,output,directory):
        if n==(3 if target=='selection_lock.json' else 2):
            path=output/target;path.write_bytes(path.read_bytes()+b' ')
    mocked['state']['mutate']=mutate
    with pytest.raises(ValueError,match='Sealed workload bytes changed'):m.run(mocked['dataset'],mocked['output'])
    assert json.loads((mocked['output']/'summary.json').read_text())['status']=='incomplete'


def test_dataset_mutation_fails_closed(mocked):
    mocked['state']['mutate']=lambda n,*_:mocked['dataset'].write_text('changed') if n==1 else None
    with pytest.raises(ValueError,match='Sealed'):m.run(mocked['dataset'],mocked['output'])
    assert len(mocked['calls'])==1


def test_no_overwrite(mocked):
    mocked['output'].mkdir();(mocked['output']/'sentinel').write_text('keep')
    with pytest.raises(ValueError,match='already exists'):m.run(mocked['dataset'],mocked['output'])
    assert (mocked['output']/'sentinel').read_text()=='keep' and not mocked['calls']


def test_same_paired_pool_required_before_gate(mocked,monkeypatch):
    monkeypatch.setattr(m.research_package,'validate_paired_forecasts',lambda *_:(_ for _ in ()).throw(ValueError('changed universe')))
    with pytest.raises(ValueError,match='changed universe'):m.run(mocked['dataset'],mocked['output'])
    assert mocked['loads']==[4] and not (mocked['output']/'selection_lock.json').exists()


def test_capacity_runtime_override_rejected_before_feature_load(mocked,monkeypatch):
    mocked['params']['catboost_mae']['MID']['catboost_mae']['thread_count']=4
    with pytest.raises(ValueError,match='five threads'):m.run(mocked['dataset'],mocked['output'])
    assert mocked['loads']==[] and mocked['calls']==[]


def test_source_drift_after_fit_rejected(mocked):
    mocked['state']['mutate']=lambda n,*_:mocked['code'].update({'fpl/source.py':'0'*64}) if n==1 else None
    with pytest.raises(ValueError,match='pipeline sources changed'):m.run(mocked['dataset'],mocked['output'])
    assert len(mocked['calls'])==1


def test_phase_schema_drift_refused_before_reference_fit(mocked,monkeypatch):
    original=m.load_research_features
    def load(end,path):
        raw,frame,columns=original(end,path)
        return raw,frame,list(reversed(columns)) if end==6 else columns
    monkeypatch.setattr(m,'load_research_features',load)
    with pytest.raises(ValueError,match='ordered base features differ'):m.run(mocked['dataset'],mocked['output'])
    assert len(mocked['calls'])==2 and mocked['loads']==[4,6]


def test_future_rows_in_frozen_loader_fail_before_fit(mocked,monkeypatch):
    original=m.load_research_features
    def load(end,path):
        raw,frame,columns=original(end,path)
        raw.loc[0,'GW_global']=end+1
        return raw,frame,columns
    monkeypatch.setattr(m,'load_research_features',load)
    with pytest.raises(ValueError,match='cutoff mismatch'):m.run(mocked['dataset'],mocked['output'])
    assert mocked['calls']==[]


def test_snapshot_future_history_rejected(mocked,monkeypatch):
    original=m.research_package.evaluate_variant
    def evaluate(*args,**kwargs):
        if 'origin_snapshot_builder' in kwargs:
            with pytest.raises(ValueError,match='precede origin'):
                kwargs['origin_snapshot_builder'](args[0],args[6])
        return original(*args,**kwargs)
    monkeypatch.setattr(m.research_package,'evaluate_variant',evaluate)
    m.run(mocked['dataset'],mocked['output'])


def test_registration_mutation_before_features_fails_closed(mocked,monkeypatch):
    original=m.experiment.log_result
    def log(name,*args):
        original(name,*args)
        if name=='league_workload_registration':
            path=mocked['output']/'registration.json';path.write_text(path.read_text()+' ')
    monkeypatch.setattr(m.experiment,'log_result',log)
    with pytest.raises(ValueError,match='Sealed'):m.run(mocked['dataset'],mocked['output'])
    assert mocked['loads']==[] and mocked['calls']==[]
