"""Verify the preserved pilot's requested and fitted recipes without fitting again."""
from collections import defaultdict
import json
from pathlib import Path
import sys

from fpl import experiment
from fpl.artifacts import sha256_file, write_json_new
from fpl.research_runner import read_result


def audit(bundle: Path, output: Path) -> None:
    """Audit eight paired receipt cells; preserve original registered/output bytes."""
    original = read_result(bundle)
    plan = json.loads((bundle / 'plan.json').read_bytes())
    if plan['window'] != {'start_gw':141, 'end_gw':144, 'horizon':3, 'retrain_every':4, 'phase':'selection_diagnostic'}:
        raise ValueError('this audit targets the preserved four-round pilot only')
    pinned = {name:sha256_file(bundle / name) for name in ['complete.json', 'plan.json']}
    pairs = defaultdict(dict)
    for number in range(1,17):
        files = [f'fit-start-{number}.json', f'fit-complete-{number}.json']
        start, complete = [json.loads((bundle / name).read_bytes()) for name in files]
        for name in files:
            pinned[name] = sha256_file(bundle / name)
        if {k:v for k,v in complete.items() if k not in {'fitted_params','seconds'}} != start:
            raise ValueError('fit start/completion metadata differs')
        recipe = plan['arms'][start['arm']]
        if (start['features'] != recipe['features'] or start['model'] != recipe['model'] or
                start['params'] != recipe['params_by_pos'][start['position']][start['model']]):
            raise ValueError('fit receipt differs from registered recipe')
        key = (start['protocol'], start['position'], start['train_min_gw'], start['train_max_gw'],
               start['training_rows'], start['seed'])
        if start['arm'] in pairs[key]:
            raise ValueError('duplicate fit cell')
        pairs[key][start['arm']] = complete
    if len(pairs) != 8:
        raise ValueError('expected two protocols times four positions')
    audited = []
    for key, arms in sorted(pairs.items()):
        if set(arms) != {'candidate','control'}:
            raise ValueError('missing paired arm')
        a, b = arms['control'], arms['candidate']
        if a['features'] != b['features']:
            raise ValueError('unmatched ordered features')
        requested = {k:[a['params'].get(k),b['params'].get(k)] for k in a['params'].keys()|b['params'].keys()
                     if a['params'].get(k) != b['params'].get(k)}
        effective = {k:[a['fitted_params'].get(k),b['fitted_params'].get(k)]
                     for k in a['fitted_params'].keys()|b['fitted_params'].keys()
                     if a['fitted_params'].get(k) != b['fitted_params'].get(k)}
        if requested != {'loss_function':['MAE','RMSE']}:
            raise ValueError('requested differences exceed loss function')
        if effective != {'loss_function':['MAE','RMSE'], 'eval_metric':['MAE','RMSE'],
                          'leaf_estimation_method':['Exact','Newton']}:
            raise ValueError('unexpected effective fitted differences')
        audited.append({'protocol':key[0], 'position':key[1], 'train_min_gw':key[2], 'train_max_gw':key[3],
            'training_rows':key[4], 'seed':key[5], 'feature_count':len(a['features']),
            'requested_differences':requested, 'effective_differences':effective})
    read_result(bundle)
    if any(sha256_file(bundle/name) != digest for name,digest in pinned.items()):
        raise ValueError('input changed during audit')
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema_version':1, 'status':'MATCHED_CAPACITY_RECIPE_VERIFIED',
        'original_bundle':str(bundle), 'input_sha256':pinned, 'script_sha256':sha256_file(Path(__file__)),
        'paired_cells':audited, 'fit_receipts':16, 'fit_start_receipts':16, 'model_fits':0, 'optimizer_solves':0,
        'promotion_eligible':False, 'interpretation':'Matched explicit non-loss hyperparameters and features. CatBoost loss-dependent effective leaf estimation and evaluation metric also differ. No separately tuned capacity contrast in this pilot.',
        'limitations':'Existing four-round, one-seed selection diagnostic; no new score or causal/promotion evidence. Older separately tuned recovered studies remain capacity-confounded.'}
    write_json_new(output/'result.json',result)
    experiment.log_result('pilot-recipe-description-audit',{'scope':'preserved_recipe_provenance','original_receipt_sha256':pinned['complete.json']},
        {'status':result['status'],'paired_cells':8,'model_fits':0,'optimizer_solves':0,'promotion_eligible':False},results_path=output/'registry.csv')
    write_json_new(output/'complete.json', {'output_sha256':{name:sha256_file(output/name) for name in ['result.json','registry.csv']}})
    print(result['status'], 'eight paired cells; zero fits/solves')


if __name__ == '__main__':
    audit(Path(sys.argv[1]),Path(sys.argv[2]))
