"""Staged grouped-interval runner tests; synthetic exports and frozen loaders only."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpl.model import grouped_interval_research as runner


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload))


@pytest.fixture
def study(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.config,'ROOT',tmp_path)
    monkeypatch.setattr(runner.config,'GROUPED_INTERVAL_MIN_CALIBRATION_ROWS',4,raising=False)
    monkeypatch.setattr(runner.config,'GROUPED_INTERVAL_MINUTES_THRESHOLDS',(.2,.8),raising=False)
    monkeypatch.setattr(runner.config,'GROUPED_INTERVAL_BOOTSTRAP_SAMPLES',100,raising=False)
    monkeypatch.setattr(runner.config,'GROUPED_INTERVAL_BLOCK_LENGTH',3,raising=False)
    monkeypatch.setattr(runner.config,'INTERVAL_ALPHA',.2);monkeypatch.setattr(runner.config,'INTERVAL_SEED',0)
    source=tmp_path/'fpl/sample.py';source.parent.mkdir();source.write_text('current source')
    dataset=tmp_path/'frozen';dataset.write_text('frozen bytes')
    manifest=tmp_path/'manifest';manifest.write_text('frozen manifest');monkeypatch.setattr(runner.provenance,'FROZEN_MANIFEST_PATH',manifest)
    base=tmp_path/'base';out=tmp_path/'run'
    old=base/'evaluated_sources/fpl/sample.py';old.parent.mkdir(parents=True);old.write_text('old source')
    splits={'selection':{'train_max_gw':1,'calibration_min_gw':2,'calibration_max_gw':5,'evaluation_min_gw':6,'evaluation_max_gw':8},
            'reference':{'train_max_gw':5,'calibration_min_gw':6,'calibration_max_gw':9,'evaluation_min_gw':10,'evaluation_max_gw':12}}
    raw=pd.DataFrame([{'player_id':pid,'position':position,'GW_global':gw,runner.FEATURE:propensity}
      for gw in range(1,14) for pid,position,propensity in [(1,'MID',.1),(2,'DEF',.5),(3,'GK',np.nan),(4,'FWD',.9)]])
    phases={}
    for phase,split in splits.items():
        for kind,low,high in [('calibration',split['calibration_min_gw'],split['calibration_max_gw']),('predictions',split['evaluation_min_gw'],split['evaluation_max_gw'])]:
            frame=raw.loc[raw.GW_global.between(low,high),runner.KEYS].copy()
            for field,value in {'actual_total_points':2.,'minutes':90.,'raw_lower':0.,'median':2.,'raw_upper':4.,
                                'calibrated_lower':2.,'calibrated_upper':2.,'adjustment':-2.}.items():frame[field]=value
            path=base/phase/f'{kind}.csv';path.parent.mkdir(exist_ok=True);frame.to_csv(path,index=False)
        phases[phase]={'split':split,'calibration_sha256':runner.research_package.file_hash(base/phase/'calibration.csv'),
                       'predictions_sha256':runner.research_package.file_hash(base/phase/'predictions.csv')}
    reg={'dataset_sha256':runner.research_package.file_hash(dataset),'alpha':.2,'seed':0,'splits':splits,
         'code_sha256':{'fpl/sample.py':runner.research_package.file_hash(old)}}
    write(base/'registration.json',reg)
    summary={'status':'complete','registration_sha256':runner.research_package.file_hash(base/'registration.json'),'phases':phases}
    write(base/'summary.json',summary)
    def verify():
        write(base/'verification.json',{'status':'VERIFIED','summary_sha256':runner.research_package.file_hash(base/'summary.json'),
                                       'registration_sha256':runner.research_package.file_hash(base/'registration.json')})
    verify()
    logs=[];loads=[];fitted=[]
    monkeypatch.setattr(runner.experiment,'_library_versions',lambda:{'mock':'1'})
    monkeypatch.setattr(runner.experiment,'log_result',lambda *args:logs.append(args))
    def loader(cap,path):
        assert logs[0][0]=='grouped_interval_research_registration'
        registration=json.loads((out/'registration.json').read_text())
        assert registration['splits']==splits and registration['min_group_size']==4
        assert len(registration['input_sha256'])==10 # Three proofs + old source + dataset+manifest + four raw CSVs.
        loads.append(cap)
        frame=raw.loc[raw.GW_global<=cap].copy();frame.attrs['frozen_dataset_sha256']=runner.research_package.file_hash(dataset)
        return frame.copy(),frame,[runner.FEATURE]
    monkeypatch.setattr(runner,'load_research_features',loader)
    fit=runner.fit_grouped_calibration
    def captured(y,lower,upper,positions,groups,**kwargs):
        fitted.append(np.asarray(y).copy());return fit(y,lower,upper,positions,groups,**kwargs)
    monkeypatch.setattr(runner,'fit_grouped_calibration',captured)
    return SimpleNamespace(dataset=dataset,base=base,out=out,raw=raw,source=source,old=old,manifest=manifest,
                           splits=splits,logs=logs,loads=loads,fitted=fitted,verify=verify)


def run(study):return runner.run(study.dataset,study.out,base_dir=study.base)


def test_registration_before_features_fixed_bothphases_fullkeys_and_metadata(study):
    summary=run(study)
    assert summary['status']=='complete' and study.loads==[8,12] and len(study.fitted)==2
    assert all((values==2).all() for values in study.fitted)
    assert set(summary['phases'])=={'selection','reference'}
    for phase in summary['phases']:
        frame=pd.read_csv(study.out/phase/'predictions.csv');base=pd.read_csv(study.base/phase/'predictions.csv')
        pd.testing.assert_frame_equal(frame[base.columns],base)
        assert set(frame.minutes_group)=={'low','mid','high','missing'}
        assert 'post_outcome_played_diagnostic_only' in summary['phases'][phase]['metrics']
    for path,digest in summary['immutable_output_sha256'].items():assert runner.research_package.file_hash(Path(path))==digest


def test_grouping_boundaries_missing_and_invalid_observations():
    assert runner.minutes_groups(pd.Series([0,.199,.2,.799,.8,1,np.nan]),(.2,.8)).tolist()==['low','low','mid','mid','high','high','missing']
    for values in ([-.1],[1.1],[np.inf],['wrong']):
        with pytest.raises(ValueError):runner.minutes_groups(pd.Series(values),(.2,.8))


def test_dgw_consistency_nan_equal_and_join_keeps_base_order():
    base=pd.DataFrame({'player_id':[2,1],'position':['MID','MID'],'GW_global':[7,7]},index=[9,2])
    frame=pd.DataFrame({'player_id':[1,1,2,2],'position':['MID']*4,'GW_global':[7]*4,runner.FEATURE:[np.nan,np.nan,.5,.5]})
    joined=runner.join_minutes(base,frame,(.2,.8))
    assert joined.player_id.tolist()==[2,1] and joined.index.tolist()==[9,2]
    assert joined.minutes_group.tolist()==['mid','missing']
    frame.loc[1,runner.FEATURE]=.2
    with pytest.raises(ValueError,match='DGW'):runner.join_minutes(base,frame,(.2,.8))


@pytest.mark.parametrize('kind',['missing','duplicate'])
def test_join_refuses_missing_feature_and_duplicate_base_round(kind):
    base=pd.DataFrame({'player_id':[1],'position':['MID'],'GW_global':[7]})
    frame=base.assign(mins60_rate_roll5=.5)
    if kind=='missing':frame['player_id']=2
    else:base=pd.concat([base,base],ignore_index=True)
    with pytest.raises(ValueError):runner.join_minutes(base,frame,(.2,.8))


def test_evaluation_outcomes_do_not_fit_cells(study):
    path=study.base/'reference/predictions.csv';frame=pd.read_csv(path);frame['actual_total_points']=999.
    frame.to_csv(path,index=False)
    summary_path=study.base/'summary.json';summary=json.loads(summary_path.read_text())
    summary['phases']['reference']['predictions_sha256']=runner.research_package.file_hash(path);write(summary_path,summary);study.verify()
    summary=run(study)
    assert all(np.max(values)<999 for values in study.fitted)
    assert summary['phases']['reference']['metrics']['all']['control']['interval_score']>0


@pytest.mark.parametrize('artifact',['dataset','manifest','current_source','archive','registration','raw_csv','earlier_output','calibration_lock'])
def test_drift_fails_closed_without_overwriting_evidence(study,monkeypatch,artifact):
    apply=runner.apply_grouped_calibration;calls=[]
    def changed(*args,**kwargs):
        result=apply(*args,**kwargs);calls.append(1)
        if len(calls)==3: # Reference calibration application, after selection files are sealed.
            paths={'dataset':study.dataset,'manifest':study.manifest,'current_source':study.source,
                   'archive':study.out/'source_archive/fpl/sample.py','registration':study.out/'registration.json',
                   'raw_csv':study.base/'selection/calibration.csv','earlier_output':study.out/'selection/predictions.csv',
                   'calibration_lock':study.out/'selection/calibration_lock.json'}
            path=paths[artifact];path.write_bytes(path.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(runner,'apply_grouped_calibration',changed)
    with pytest.raises(ValueError):run(study)
    assert json.loads((study.out/'summary.json').read_text())['status']=='incomplete'


def test_base_verification_required_before_feature_loading(study):
    write(study.base/'verification.json',{'status':'incomplete'})
    with pytest.raises(ValueError,match='independently verified'):run(study)
    assert not study.loads and not study.fitted


def test_no_overwrite(study):
    study.out.mkdir();marker=study.out/'keep';marker.write_text('preserve')
    with pytest.raises(ValueError,match='already exists'):run(study)
    assert marker.read_text()=='preserve' and not study.logs


def test_position_control_numeric_reconstruction_must_match(study):
    path=study.base/'selection/predictions.csv';frame=pd.read_csv(path);frame['adjustment']=-1.
    frame.to_csv(path,index=False);summary_path=study.base/'summary.json';summary=json.loads(summary_path.read_text())
    summary['phases']['selection']['predictions_sha256']=runner.research_package.file_hash(path);write(summary_path,summary);study.verify()
    with pytest.raises(ValueError,match='control differs'):run(study)
    assert not study.fitted


def test_moving_block_statistic_weights_player_rounds_and_preserves_round_counts():
    gw=np.array([1,2,2,3,4,4,4]);delta=np.array([1,2,2,3,4,4,4],dtype=float)
    report=runner.paired_interval_score_ci(gw,delta,n_boot=20,block_len=3,seed=0)
    sums=np.array([1,4,3,12]);counts=np.array([1,2,1,3])
    starts=np.random.default_rng(0).integers(0,2,size=(20,2))
    indices=(starts[:,:,None]+np.arange(3)).reshape(20,-1)[:,:4]
    reference=sums[indices].sum(1)/counts[indices].sum(1)
    assert report['mean_interval_score_delta']==pytest.approx(delta.mean())
    assert report['ci_low']==pytest.approx(np.quantile(reference,.025)) and report['ci_high']==pytest.approx(np.quantile(reference,.975))
    assert report['player_rounds']==7 and report['rounds']==4 and not any('p_value' in key for key in report)


def test_noncontiguous_gameweeks_refused():
    with pytest.raises(ValueError,match='contiguous'):runner.paired_interval_score_ci(np.array([1,3,4]),np.ones(3),n_boot=10,block_len=3,seed=0)


def test_calibration_lock_written_before_any_application(study,monkeypatch):
    apply=runner.apply_grouped_calibration;calls=[]
    def check(*args,**kwargs):
        phase='selection' if len(calls)<2 else 'reference'
        lock=json.loads((study.out/phase/'calibration_lock.json').read_text())
        assert lock['min_group_size']==4 and lock['group_counts']
        calls.append(1)
        return apply(*args,**kwargs)
    monkeypatch.setattr(runner,'apply_grouped_calibration',check)
    run(study);assert len(calls)==4


def test_join_refuses_existing_generated_features():
    base=pd.DataFrame({'player_id':[1],'position':['MID'],'GW_global':[7],runner.FEATURE:[.2]})
    with pytest.raises(ValueError):runner.join_minutes(base,base,(.2,.8))


def test_feature_loader_future_rows_fail_before_calibration(study,monkeypatch):
    def broken(cap,path):
        frame=study.raw.copy();frame.attrs['frozen_dataset_sha256']=runner.research_package.file_hash(path)
        return frame.copy(),frame,[]
    monkeypatch.setattr(runner,'load_research_features',broken)
    with pytest.raises(ValueError,match='feature cap'):run(study)
    assert not study.fitted


@pytest.mark.parametrize('column,value',[('player_id',True),('GW_global',6.5),('minutes',-1.),
                                         ('actual_total_points',np.nan),('raw_lower',5.),('position','Manager')])
def test_base_csv_identity_numeric_and_bounds_invalid(column,value):
    frame=pd.DataFrame({'player_id':[1],'position':['MID'],'GW_global':[7], 'actual_total_points':[2.],
                        'minutes':[90.],'raw_lower':[0.],'median':[2.],'raw_upper':[4.],
                        'calibrated_lower':[2.],'calibrated_upper':[2.],'adjustment':[-2.]})
    frame[column]=value
    with pytest.raises(ValueError):runner._validate_base(frame,6,8)


def test_final_summary_mutation_by_completion_logger_is_incomplete(study, monkeypatch):
    def mutate(name, params, metrics):
        study.logs.append((name, params, metrics))
        if name == 'grouped_interval_research_completed':
            assert params['summary_sha256'] == runner.research_package.file_hash(study.out / 'summary.json')
            (study.out / 'summary.json').write_text('tampered final summary')
    monkeypatch.setattr(runner.experiment, 'log_result', mutate)
    with pytest.raises(ValueError, match='Final grouped interval summary changed'):
        run(study)
    summary = json.loads((study.out / 'summary.json').read_text())
    assert summary['status'] == 'incomplete' and set(summary['phases']) == {'selection', 'reference'}
    assert str((study.out / 'summary.json').resolve()) not in summary['immutable_output_sha256']
