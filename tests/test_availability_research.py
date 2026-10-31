"""Mocked availability registration, chronology and immutable input enforcement."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fpl import config
from fpl.model import availability_research as runner


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def study(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'ROOT',tmp_path)
    source=tmp_path/'fpl/test.py';source.parent.mkdir();source.write_text('preserved source')
    dataset=tmp_path/'frozen.csv';dataset.write_text('frozen bytes')
    manifest=tmp_path/'frozen_manifest.json';write(manifest,{'frozen_cutoff':231})
    monkeypatch.setattr(runner.provenance,'FROZEN_MANIFEST_PATH',manifest)
    audit=tmp_path/'audit';out=tmp_path/'run'
    cutoffs={'discovery_max_gw':1,'selection_min_gw':2,'selection_max_gw':3,'backtest_min_gw':4,'backtest_max_gw':5}
    monkeypatch.setattr(runner,'protocol_cutoffs',lambda:cutoffs)
    proof_names=('discovery_wayback_verification.json','wayback_verification.json')
    capture_paths=[]
    for name,origins,indexname in [(proof_names[0],[1],'discovery_wayback_index.json'),(proof_names[1],[2,3,4,5],'wayback_index.json')]:
        index=audit/'inventory'/indexname;write(index,[])
        captures=[]
        for origin in origins:
            capture={'season':'2020-21','event':origin}
            for key,suffix in [('original_payload','payload'),('manifest','json')]:
                path=audit/f'{origin}.{suffix}';path.write_text('preserved capture')
                capture[key]=path.name;capture[f'{key}_sha256']=runner.research_package.file_hash(path)
                capture_paths.append(path)
            captures.append(capture)
        write(audit/name,{'status':'incomplete','captures':captures,'index':{'sha256':runner.research_package.file_hash(index)}})
    loaded_rosters=[]
    def roster(proof,index,digest,origins):
        assert runner.research_package.file_hash(proof)==digest
        origin=origins[0];loaded_rosters.append(origin)
        if origin==2:raise runner.deadline_snapshots.SnapshotUnavailableError(origin,'no verified capture')
        frame=pd.DataFrame({'player_id':[1,2],'origin_gw':origin,'status':['a','d'],
                            'chance_of_playing_next_round':[100.,50.],'snapshot_age_hours':1.,
                            'captured_at':'2024-01-01T09:00:00Z','deadline':'2024-01-01T10:00:00Z'})
        frame.attrs['excluded_nonplayer_count']=0
        return frame
    monkeypatch.setattr(runner.deadline_snapshots,'load_deadline_rosters',roster)
    raw=pd.DataFrame([{'player_id':pid,'GW_global':gw,'position':'MID','value':50.,'total_points':2.,
                       'minutes':90.,'fixture':gw*10+pid,'base':float(gw)} for gw in range(1,7) for pid in (1,2)])
    loaded=[]
    def features(end,path):
        loaded.append(end);frame=raw.loc[raw.GW_global<=end].copy()
        frame.attrs['frozen_dataset_sha256']=runner.research_package.file_hash(dataset)
        return frame.copy(),frame,['base']
    monkeypatch.setattr(runner,'load_research_features',features)
    tuned=tmp_path/'tuned.json';tuned.write_text('tracked parameters')
    params={name:{pos:{name:{'depth':3 if name=='catboost_mae' else 7,'random_seed':0}}
                  for pos in config.ONFIELD_POSITIONS} for name in ('catboost_mae','catboost_rmse')}
    resolved={'seed':0,'calls':[]}
    def resolve(mode,seed):
        assert mode=='selection'
        resolved['seed']=seed
        resolved['calls'].append((mode,seed))
        for model,positions in params.items():
            for position in config.ONFIELD_POSITIONS:
                positions[position][model]['random_seed']=seed
        return params,{'tuned.json':runner.research_package.file_hash(tuned)}
    monkeypatch.setattr(runner.research_package,'resolve_parameters',resolve)
    logs=[];evaluated=[];compares=[];net={'mae_full':10.,'mae_availability':11.,'rmse_full':20.,'rmse_availability':21.}
    monkeypatch.setattr(runner.experiment,'_library_versions',lambda:{'mock':'1'})
    monkeypatch.setattr(runner.experiment,'log_result',lambda *args:logs.append(args))
    monkeypatch.setattr(runner,'build_live_snapshot',lambda hist:hist[['player_id']].drop_duplicates().copy())
    def evaluate(rawframe,frame,columns,arm,parameters,start,end,origin,directory,seed,**kwargs):
        assert logs[0][0]=='availability_research_registration' and seed==resolved['seed']
        assert logs[0][1]['seed']==seed
        model=kwargs['model_name']
        assert model==logs[0][1]['model_name']
        assert set(parameters)=={model}
        assert all(p[model]['random_seed']==seed for p in parameters[model].values())
        assert loaded_rosters==[1,2,3,4,5] and loaded==[3,5]
        assert (out/'selection_origin_prices.csv').is_file() and (out/'reference_origin_prices.csv').is_file()
        assert (out/'rosters.csv').is_file() and (out/'coverage_mask.csv').is_file()
        assert len(frame)==len(rawframe) and (frame.total_points==2).all()
        assert frame.loc[frame.GW_global==2,runner.availability_columns()].isna().all().all()
        assert parameters=={model:params[model]} and kwargs['retrain_every']==4
        assert columns==(['base']+runner.availability_columns() if arm.endswith('_availability') else ['base'])
        if directory.name=='reference':
            lock=json.loads((out/'selection_lock.json').read_text());assert lock['advanced']
            assert lock['selection_summary_sha256']==runner.research_package.file_hash(out/'selection_summary.json')
        if origin:
            assert kwargs['origin_price_policy']=='origin-current-proxy'
            assert kwargs['origin_price_catalog']==out/f'{directory.name}_origin_prices.csv'
            if arm.endswith('_availability'):
                assert kwargs['extra_player_features']==runner.availability_columns()
                snapshot=kwargs['origin_snapshot_builder'](rawframe.loc[rawframe.GW_global<3],3)
                assert (snapshot.GW_global==3).all() and snapshot.archive_status_a.tolist()==[1.,0.]
                missing=kwargs['origin_snapshot_builder'](rawframe.loc[rawframe.GW_global<2],2)
                assert missing[runner.availability_columns()].isna().all().all()
            else:assert 'origin_snapshot_builder' not in kwargs
        else:assert 'origin_price_policy' not in kwargs and 'origin_snapshot_builder' not in kwargs
        predictions=[]
        for gw in range(start,end+1):
            for pid in (1,2):
                predictions.append({'player_id':pid,'position':'MID','GW':gw,'fixture':gw*10+pid,
                                    'value':50.,'minutes':90.,'actual_total_points':2.,'predicted_total_points':float(arm.endswith('_availability'))})
        exported=pd.DataFrame(predictions)
        if origin:exported['origin_gw']=exported.GW
        prefix=f'{arm}_{"origin" if origin else "standard"}'
        pred=f'{prefix}.predictions.csv';squad=f'{prefix}.squads.csv'
        exported.to_csv(directory/pred,index=False)
        pd.DataFrame({'gameweek':range(start,end+1),'actual_total_points':2.}).to_csv(directory/squad,index=False)
        result={'predictions':pred,'squads':squad,'prediction_sha256':runner.research_package.file_hash(directory/pred),
                'squad_sha256':runner.research_package.file_hash(directory/squad),'total_points':net[arm]}
        for suffix in ('admission.json','result.json'):write(directory/f'{prefix}.{suffix}',result)
        (directory/f'{prefix}.log').write_text('mock optimal checks delegated')
        evaluated.append((directory.name,arm,origin))
        return result
    monkeypatch.setattr(runner.research_package,'evaluate_variant',evaluate)
    def compare(a,b,**kwargs):
        assert kwargs=={'n_boot':10000,'block_len':3,'seed':resolved['seed']}
        compares.append((a,b));return {'total_diff':1.}
    monkeypatch.setattr(runner.research_package.compare_backtests,'compare',compare)
    return SimpleNamespace(dataset=dataset,audit=audit,out=out,raw=raw,loaded=loaded,logs=logs,
                           evaluated=evaluated,compares=compares,net=net,source=source,tuned=tuned,captures=capture_paths,manifest=manifest,resolved=resolved)


def run(study):return runner.run(study.dataset,study.out,audit_dir=study.audit)


def test_registered_equal_capacity_gate_origin_freezing_missing_rows_preserved(study):
    summary=run(study)
    assert summary['status']=='complete' and summary['advanced'] and not summary['promotion_eligible']
    assert len(study.evaluated)==6 and len(study.compares)==3
    assert study.evaluated[:2]==[('selection','mae_full',True),('selection','mae_availability',True)]
    reg=json.loads((study.out/'registration.json').read_text())
    assert reg['coverage'][1]['available'] is False and reg['coverage'][1]['reason']=='no verified capture'
    assert reg['coverage_counts']['selection']['fixture_rows']==6
    assert reg['coverage_counts']['selection']['missing_fixture_rows']==2
    assert reg['coverage_counts']['selection']['evaluation_covered_fixture_rows']==2
    assert reg['coverage_counts']['selection']['evaluation_missing_fixture_rows']==2
    assert reg['max_capture_age_hours']==config.DEADLINE_SNAPSHOT_MAX_AGE_HOURS
    assert str(study.manifest.resolve()) in reg['input_sha256']
    assert any('history-based' in text for text in reg['limitations'])
    for path,digest in summary['immutable_output_sha256'].items():assert runner.research_package.file_hash(Path(path))==digest
    assert study.logs[-1][1]['summary_sha256']==runner.research_package.file_hash(study.out/'summary.json')


@pytest.mark.parametrize('net',[9.,10.])
def test_no_reference_when_candidate_loses_or_ties(study,net):
    study.net['mae_availability']=net
    summary=run(study)
    assert summary['status']=='selection_rejected' and not summary['evaluation']
    assert len(study.evaluated)==2 and len(study.compares)==1
    assert json.loads((study.out/'selection_lock.json').read_text())['finalists']==[]


@pytest.mark.parametrize('artifact',['source','dataset','tuned','proof','roster','catalog','coverage','earlier_output','lock','frozen_manifest'])
def test_sealed_input_or_output_mutation_fails_closed(study,monkeypatch,artifact):
    evaluate=runner.research_package.evaluate_variant
    def changed(*args,**kwargs):
        result=evaluate(*args,**kwargs)
        mutate_after=3 if artifact=='lock' else 1
        if len(study.evaluated)==mutate_after:
            paths={'source':study.source,'dataset':study.dataset,'tuned':study.tuned,
                   'proof':study.audit/'wayback_verification.json','roster':study.out/'rosters.csv',
                   'catalog':study.out/'reference_origin_prices.csv','coverage':study.out/'coverage_mask.csv',
                   'earlier_output':study.out/'selection/mae_full_origin.predictions.csv','lock':study.out/'selection_lock.json',
                   'frozen_manifest':study.manifest}
            path=paths[artifact];path.write_bytes(path.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(runner.research_package,'evaluate_variant',changed)
    with pytest.raises(ValueError):run(study)
    summary=json.loads((study.out/'summary.json').read_text())
    assert summary['status']=='incomplete'
    assert study.logs[-1][0]=='availability_research_incomplete'


def test_fatal_archive_error_not_hidden_as_missing_signal(study,monkeypatch):
    monkeypatch.setattr(runner.deadline_snapshots,'load_deadline_rosters',lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('invalid archive hash')))
    with pytest.raises(ValueError,match='invalid archive hash'):run(study)
    assert study.loaded==[] and study.evaluated==[]


def test_missing_proof_fails_before_fit(study):
    (study.audit/'discovery_wayback_verification.json').unlink()
    with pytest.raises(FileNotFoundError):run(study)
    assert study.evaluated==[] and study.loaded==[]


def test_existing_directory_never_overwritten(study):
    study.out.mkdir();marker=study.out/'keep';marker.write_text('preserve')
    with pytest.raises(ValueError,match='already exists'):run(study)
    assert marker.read_text()=='preserve' and not study.evaluated


def test_cli_explicit_audit_and_dataset(monkeypatch):
    calls=[];monkeypatch.setattr(runner,'run',lambda *args,**kwargs:calls.append((args,kwargs)))
    runner.main(['--dataset','frozen','--output-dir','new','--audit-dir','audit'])
    assert calls==[((Path('frozen'),Path('new')),{'audit_dir':Path('audit')})]


def test_reference_predictions_must_preserve_paired_metadata(study,monkeypatch):
    evaluate=runner.research_package.evaluate_variant
    def changed(*args,**kwargs):
        result=evaluate(*args,**kwargs)
        if args[3]=='mae_availability' and args[8].name=='reference' and args[7]:
            path=args[8]/result['predictions']
            frame=pd.read_csv(path);frame.loc[0,'minutes']=0.
            frame.to_csv(path,index=False)
            result['prediction_sha256']=runner.research_package.file_hash(path)
        return result
    monkeypatch.setattr(runner.research_package,'evaluate_variant',changed)
    with pytest.raises(ValueError,match='paired forecasts differ'):run(study)
    assert json.loads((study.out/'summary.json').read_text())['status']=='incomplete'


def test_selection_input_mismatch_cannot_advance(study,monkeypatch):
    evaluate=runner.research_package.evaluate_variant
    def changed(*args,**kwargs):
        result=evaluate(*args,**kwargs)
        if args[3]=='mae_availability':
            path=args[8]/result['predictions']
            frame=pd.read_csv(path);frame.loc[0,'value']=60.
            frame.to_csv(path,index=False)
            result['prediction_sha256']=runner.research_package.file_hash(path)
        return result
    monkeypatch.setattr(runner.research_package,'evaluate_variant',changed)
    with pytest.raises(ValueError,match='paired forecasts differ'):run(study)
    assert len(study.evaluated)==2 and not (study.out/'selection_lock.json').exists()


@pytest.mark.parametrize('seed',[0,1,2])
def test_requested_seed_propagates_same_capacity_to_every_arm_and_ci(study,seed):
    summary=runner.run(study.dataset,study.out,audit_dir=study.audit,seed=seed)
    assert summary['status']=='complete' and len(study.evaluated)==6
    assert study.resolved['calls']==[('selection',seed)]
    reg=json.loads((study.out/'registration.json').read_text())
    assert reg['seed']==seed and reg['comparison']['seed']==seed
    for position in config.ONFIELD_POSITIONS:
        assert reg['parameters']['catboost_mae'][position]['catboost_mae']=={'depth':3,'random_seed':seed}
    assert 'no new HPO' in reg['parameter_protocol']
    assert any('not independent pooled samples' in text for text in reg['limitations'])
    assert len(study.compares)==3


@pytest.mark.parametrize('seed',[1,2])
@pytest.mark.parametrize('net',[9.,10.])
def test_nonzero_seeds_keep_strict_selection_gate(study,seed,net):
    study.net['mae_availability']=net
    summary=runner.run(study.dataset,study.out,audit_dir=study.audit,seed=seed)
    assert summary['status']=='selection_rejected' and summary['evaluation']=={}
    assert len(study.evaluated)==2


@pytest.mark.parametrize('seed',[True,False,-1,3,1.0,'1',None])
def test_invalid_seed_refused_without_artifacts_or_input_reads(study,seed):
    with pytest.raises(ValueError,match='seed must be an integer'):
        runner.run(study.dataset,study.out,audit_dir=study.audit,seed=seed)
    assert not study.out.exists() and not study.loaded and not study.logs
    assert study.resolved['calls']==[] and study.evaluated==[]


def test_cli_explicit_nonzero_seed(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner,'run',lambda *args,**kwargs:calls.append((args,kwargs)))
    runner.main(['--dataset','frozen','--output-dir','new','--audit-dir','audit','--seed','2'])
    assert calls==[((Path('frozen'),Path('new')),{'audit_dir':Path('audit'),'seed':2})]


@pytest.mark.parametrize('seed',[0,1,2])
def test_rmse_namespaces_capacity_arm_names_and_seed_are_exact(study,seed):
    summary=runner.run(study.dataset,study.out,audit_dir=study.audit,seed=seed,model_name='catboost_rmse')
    assert summary['status']=='complete' and set(summary['selection'])=={'rmse_full','rmse_availability'}
    assert set(summary['evaluation'])=={'rmse_full','rmse_availability'}
    assert [arm for _,arm,_ in study.evaluated]==['rmse_full','rmse_availability','rmse_full','rmse_full','rmse_availability','rmse_availability']
    reg=json.loads((study.out/'registration.json').read_text())
    assert reg['model_name']=='catboost_rmse' and set(reg['parameters'])=={'catboost_rmse'}
    assert reg['arms']==['rmse_full','rmse_availability']
    for position in config.ONFIELD_POSITIONS:
        assert reg['parameters']['catboost_rmse'][position]['catboost_rmse']=={'depth':7,'random_seed':seed}
    assert reg['feature_variants']['rmse_full']==['base']
    assert reg['feature_variants']['rmse_availability']==['base']+runner.availability_columns()
    assert 'catboost_rmse' in reg['parameter_protocol'] and 'RMSE' in reg['hypothesis']
    assert len(runner.ARMS)==2 and runner.ARMS==('mae_full','mae_availability')


@pytest.mark.parametrize('net',[19.,20.])
def test_rmse_strict_gate_never_uses_mae_or_forecast_accuracy(study,net,monkeypatch):
    study.net['rmse_availability']=net
    study.net['mae_full']=-999.  # A mistaken cross-loss comparison would advance.
    evaluate=runner.research_package.evaluate_variant
    def diagnostic(*args,**kwargs):
        result=evaluate(*args,**kwargs)
        result['diagnostics']=[{'mae':0.,'rmse':0.,'total_calibration':1.}]
        return result
    monkeypatch.setattr(runner.research_package,'evaluate_variant',diagnostic)
    summary=runner.run(study.dataset,study.out,audit_dir=study.audit,model_name='catboost_rmse')
    assert summary['status']=='selection_rejected' and not summary['evaluation']
    assert len(study.evaluated)==2
    assert json.loads((study.out/'selection_lock.json').read_text())['finalists']==[]


@pytest.mark.parametrize('model',['catboost','sequence_cnn','MAE','RMSE',None,True,0,[]])
def test_invalid_model_refused_before_artifacts_and_inputs(study,model):
    with pytest.raises(ValueError,match='model_name must'):
        runner.run(study.dataset,study.out,audit_dir=study.audit,model_name=model)
    assert not study.out.exists() and not study.loaded and not study.logs and not study.resolved['calls']


def test_explicit_mae_preserves_default_arms_and_protocol(study):
    summary=runner.run(study.dataset,study.out,audit_dir=study.audit,model_name='catboost_mae')
    assert summary['status']=='complete' and set(summary['selection'])==set(runner.ARMS)
    reg=json.loads((study.out/'registration.json').read_text())
    assert reg['arms']==list(runner.ARMS) and reg['model_name']=='catboost_mae'


def test_cli_loss_objective_override_preserves_seed(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner,'run',lambda *args,**kwargs:calls.append((args,kwargs)))
    runner.main(['--dataset','frozen','--output-dir','new','--audit-dir','audit','--model-name','catboost_rmse','--seed','1'])
    assert calls==[((Path('frozen'),Path('new')),{'audit_dir':Path('audit'),'seed':1,'model_name':'catboost_rmse'})]


@pytest.mark.parametrize('advance', [True, False])
def test_final_summary_mutation_by_completion_logger_is_incomplete(study, monkeypatch, advance):
    if not advance:
        study.net['mae_availability'] = study.net['mae_full']
    def mutate(name, params, metrics):
        study.logs.append((name, params, metrics))
        if name == 'availability_research_completed':
            assert params['summary_sha256'] == runner.research_package.file_hash(study.out / 'summary.json')
            (study.out / 'summary.json').write_text('tampered final summary')
    monkeypatch.setattr(runner.experiment, 'log_result', mutate)
    with pytest.raises(ValueError, match='Final availability summary changed'):
        run(study)
    summary = json.loads((study.out / 'summary.json').read_text())
    assert summary['status'] == 'incomplete'
    assert summary['advanced'] is advance
    assert str((study.out / 'summary.json').resolve()) not in summary['immutable_output_sha256']
