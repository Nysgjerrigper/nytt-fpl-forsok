"""Registered replay rejects bad certificates/drift without retraining or overwrites."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from fpl.model import research_package as rp


@pytest.fixture(params=['run.py','run_v2.py','run_v3.py'])
def replay(tmp_path, monkeypatch, request):
    path=Path(__file__).resolve().parents[1]/'experiments/corrected_optimizer_replay_2026-10-04'/request.param
    spec=importlib.util.spec_from_file_location('replay_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if hasattr(module,'RECOVERY'):monkeypatch.setattr(module,'RECOVERY',None)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'OUTPUT',tmp_path/'new_replay')
    monkeypatch.setattr(module,'START',1);monkeypatch.setattr(module,'END',3)
    monkeypatch.setattr(module,'SEEDS',(0,))
    source=tmp_path/'fpl/fake.py';source.parent.mkdir();source.write_text('# archived source\n')
    monkeypatch.setattr(rp,'source_hashes',lambda:{'fpl/fake.py':rp.file_hash(source)})
    dataset=tmp_path/'frozen.csv';dataset.write_text('sealed input')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    monkeypatch.setattr(module,'DATASET',dataset);monkeypatch.setattr(module,'FROZEN_MANIFEST_PATH',manifest)
    monkeypatch.setattr(module,'load_frozen_research_dataset',lambda path:pd.DataFrame({'GW_global':[1]}))
    verifier=tmp_path/'experiments/league_workload_2026-10-04/verify_v2.py'
    verifier.parent.mkdir(parents=True);verifier.write_text('def budgets_and_points(predictions,squads,*args,**kwargs):\n    return len(squads)\n')
    base=tmp_path/'experiments/research_priorities_2026-10-03/selection_seed0'
    (base/'evaluation').mkdir(parents=True)
    (base/'registration.json').write_text(json.dumps({'parameters':{'catboost_mae':{},'catboost_rmse':{}}}))
    evaluation={}
    for arm in module.ARMS:
        evaluation[arm]={}
        for mode in module.MODES:
            pairs=[(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)] if mode=='origin' else [(1,1),(2,2),(3,3)]
            frame=pd.DataFrame({'player_id':1,'position':'MID','GW':[gw for _,gw in pairs],
                'fixture':[gw*10 for _,gw in pairs],'minutes':90,'value':50.,
                'actual_total_points':3.,'predicted_total_points':2.})
            if mode=='origin':frame['origin_gw']=[origin for origin,_ in pairs]
            prediction=base/'evaluation'/f'{arm}_{mode}.predictions.csv';frame.to_csv(prediction,index=False)
            evaluation[arm][mode]={'predictions':prediction.name,'prediction_sha256':rp.file_hash(prediction),'total_points':9.,'diagnostics':[]}
    (base/'summary.json').write_text(json.dumps({'evaluation':evaluation}))
    monkeypatch.setattr(module.experiment,'log_result',lambda *a,**k:None)
    monkeypatch.setattr(module.experiment,'_library_versions',lambda:{})
    def evaluate(raw,frame,columns,arm,parameters,start,end,origin,directory,seed,**kwargs):
        assert (module.OUTPUT/'registration.json').is_file(), 'Must register before scoring'
        forecasts=kwargs['forecast_provider']()
        prefix=f'{arm}_{"origin" if origin else "standard"}'
        forecasts.to_csv(directory/f'{prefix}.predictions.csv',index=False)
        squads=pd.DataFrame({'gameweek':[1,2,3],'actual_total_points':3.,
            'solver_status':'Optimal','solver_solution_status':'Optimal Solution Found',
            'solver_native_status':'Optimal','solver_gap_rel':0.,'solver_gap_abs':0.,
            'solver_mip_gap':0.,'solver_primal_bound':1.,'solver_dual_bound':1.,
            'scoring_protocol':'autosubs_v2','points_accounting':'net_after_transfer_hits',
            'gross_actual_total_points':3.,'alpha':0,'q_start':[1,2,2],
            'transfers_in':None,'chip_played':None})
        if getattr(module,'bad_gap',False):squads['solver_mip_gap']=.001
        if getattr(module,'roundoff_gap',False):
            squads['solver_mip_gap']=1.6e-16
            squads['solver_dual_bound']=1.+2.2e-16
        squads.to_csv(directory/f'{prefix}.squads.csv',index=False)
        return {'variant':arm,'predictions':f'{prefix}.predictions.csv','squads':f'{prefix}.squads.csv','squad_sha256':rp.file_hash(directory/f'{prefix}.squads.csv'),'total_points':9.,'diagnostics':[]}
    monkeypatch.setattr(rp,'evaluate_variant',evaluate)
    if hasattr(module,'score_case'):
        module.actual_score_case=module.score_case
        monkeypatch.setattr(module,'score_case',lambda case,directory:evaluate(pd.DataFrame(),pd.DataFrame(),[],case['arm'],case['parameters'],1,3,case['mode']=='origin',directory,case['seed'],forecast_provider=lambda:pd.read_csv(case['prediction'])))
    monkeypatch.setattr(rp.predict,'walk_forward_predictions',lambda *a,**k:pytest.fail('Replay refitted'))
    monkeypatch.setattr(rp.predict,'origin_based_predictions',lambda *a,**k:pytest.fail('Replay refitted'))
    return module


def test_saved_forecast_replay_seals_inputs_and_refuses_overwrite(replay):
    report=replay.run()
    assert report['status']=='complete' and report['optimal_solves']==12
    assert len(report['results'])==4 and report['promotion_eligible'] is False
    assert report['comparisons']['0']['origin']['total_diff']==0
    before={path:path.read_bytes() for path in replay.OUTPUT.rglob('*') if path.is_file()}
    with pytest.raises(ValueError,match='preserve'):replay.run()
    assert all(path.read_bytes()==data for path,data in before.items())


def test_nonzero_achieved_gap_rejected(replay):
    replay.bad_gap=True
    with pytest.raises(ValueError,match='Nonzero requested/achieved gap|exceeds floating-point roundoff'):replay.run()
    assert json.loads((replay.OUTPUT/'summary.json').read_text())['status']=='incomplete'


def test_completion_summary_drift_fails_closed(replay,monkeypatch):
    def log(name,*args):
        if name=='corrected_optimizer_replay_completed':
            (replay.OUTPUT/'summary.json').write_text('{}')
    monkeypatch.setattr(replay.experiment,'log_result',log)
    with pytest.raises(ValueError,match='Final replay summary changed'):replay.run()
    assert json.loads((replay.OUTPUT/'summary.json').read_text())['status']=='incomplete'


def test_roundoff_does_not_relax_requested_gaps(replay):
    replay.roundoff_gap=True
    if hasattr(replay,'RECOVERY') or hasattr(replay,'score_case'):
        report=replay.run()
        assert report['status']=='complete'
        registration=json.loads((replay.OUTPUT/'registration.json').read_text())
        assert registration['gap_rel']==registration['gap_abs']==0
        assert registration['roundoff_relative_tolerance']<1e-12
    else:
        with pytest.raises(ValueError,match='Nonzero requested/achieved gap'):replay.run()


def test_direct_replay_preserves_forecast_bytes(replay,tmp_path,monkeypatch):
    if not hasattr(replay,'actual_score_case'):
        pytest.skip('Byte-preserving path introduced in v3')
    prediction=tmp_path/'tiny.csv'
    pd.DataFrame({'player_id':[1,1,1],'GW':[1,2,3],'position':'MID','fixture':[10,20,30],
        'predicted_total_points':[1.23456789012345e-18]*3,'actual_total_points':3.,
        'minutes':90.,'value':50.}).to_csv(prediction,index=False)
    source_bytes=prediction.read_bytes()
    def solve(args):
        assert Path(args.predictions_csv).read_bytes()==source_bytes
        pd.DataFrame({'gameweek':[1,2,3],'actual_total_points':3.,'alpha':0,
            'scoring_protocol':'autosubs_v2','chip_played':None}).to_csv(args.output,index=False)
    monkeypatch.setattr(replay.optimize,'run',solve)
    directory=tmp_path/'case';directory.mkdir()
    case={'arm':'mae_full','mode':'standard','prediction':str(prediction),'parameters':{},
          'seed':0,'forecast_diagnostics':[]}
    result=replay.actual_score_case(case,directory)
    assert (directory/result['predictions']).read_bytes()==source_bytes
    assert result['total_points']==9


def test_independent_verifier_reconstructs_and_rejects_drift(replay,monkeypatch):
    if not hasattr(replay,'actual_score_case'):
        pytest.skip('Independent certificate verification covers v3')
    replay.run()
    source=Path(__file__).resolve().parents[1]/'experiments/corrected_optimizer_replay_2026-10-04/verify.py'
    local=replay.ROOT/'experiments/corrected_optimizer_replay_2026-10-04/verify.py'
    local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(source.read_bytes())
    spec=importlib.util.spec_from_file_location('verifier_under_test',local)
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    report=verifier.verify(replay.OUTPUT)
    assert report['status']=='verified' and report['independently_reconstructed_rounds']==12
    squad=replay.OUTPUT/'seed0/mae_full_standard.squads.csv'
    squad.write_text(squad.read_text()+'\n')
    with pytest.raises(ValueError,match='Evidence drift'):verifier.verify(replay.OUTPUT)
