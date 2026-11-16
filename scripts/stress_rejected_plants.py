"""Predict and validate rejected generation plants, without fitting any models."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,re,ast,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import FEATURES,assess_controller
from data_gen import Plant,PIDGains,PlantSimulator,FeatureExtractor,PIDOptimizer

OUT=Path('results/rejected_plants')
MODELS=Path('results/dataset_20000')
NAMES=['linear','ridge','polynomial_ridge','random_forest','xgboost','standard_mlp']


def recover_rejections(log_text):
    pattern=r'(Discarded|Failed) Plant\(zeta=([^,]+), wn=([^,]+), tau=([^,]+), K=([^\)]+)\): ([^\r\n]*)'
    rows=[]
    for status,z,w,t,k,details in re.findall(pattern,log_text):
        reasons=ast.literal_eval(details) if status=='Discarded' else {'exception':details}
        rows.append({'rejection_id':len(rows),'zeta':float(z),'wn':float(w),'tau':float(t),'K':float(k),
                     'status':status,'original_reason':'+'.join(sorted(reasons)),
                     'original_details':json.dumps(reasons,sort_keys=True)})
    return pd.DataFrame(rows)


def validate(task):
    name,index,parameters,gains,weight=task
    plant=Plant(*parameters);vector=np.array(gains)
    strict=assess_controller(plant,vector,weight)
    result={'model':name,'rejection_id':index,**strict,
            **{f'predicted_{key}':float(value) for key,value in zip(['Kp','Ki','Kd'],vector)},
            'extended_attempted':False,'extended_accepted':strict['accepted'],
            'extended_reason':strict['reason'],'extended_itae':strict['itae'],
            'extended_objective':strict['objective'],'extended_overshoot_pct':strict['overshoot_pct'] if strict['accepted'] else None}
    if strict['reason']=='resolution_limit':
        result['extended_attempted']=True
        sim=PlantSimulator();opt=PIDOptimizer(sim,gain_penalty=weight,max_validation_points=262145)
        metrics,reason=opt._validate_candidate(plant,PIDGains(*vector),sim.get_dynamic_t_eval(plant))
        result.update(extended_accepted=metrics is not None,extended_reason=reason)
        if metrics is not None:
            result.update(extended_itae=metrics['itae'],extended_overshoot_pct=metrics['overshoot_pct'],
                          extended_objective=opt.objective_terms(plant,vector,metrics['itae'])['objective'])
    return result


if __name__=='__main__':
    from MLP import MLPModel
    import torch
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    log=Path('data/generation_20000.log');source=Path('data/pid_dataset_20000.csv')
    metadata=json.loads(source.with_suffix('.meta.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==metadata['dataset_sha256']
    plants=recover_rejections(log.read_text(encoding='utf-8'))
    assert len(plants)==metadata['attempts']-metadata['samples']
    assert not plants.duplicated(['zeta','wn','tau','K']).any()
    accepted=pd.read_csv(source)[['zeta','wn','tau','K']].to_numpy()
    for r in plants.itertuples():
        vector=np.array([r.zeta,r.wn,r.tau,r.K])
        assert not np.any(np.all(np.abs(accepted-vector)<1e-12,axis=1))
    sim=PlantSimulator();extractor=FeatureExtractor()
    for i,r in plants.iterrows():
        t,y=sim.observe_open_loop(Plant(r.zeta,r.wn,r.tau,r.K))
        for key,value in extractor.extract(t,y).items():plants.loc[i,key]=value
        plants.loc[i,'observation_duration']=t[-1]
    assert np.isfinite(plants[FEATURES].to_numpy()).all()
    plants.to_csv(OUT/'recovered_plants.csv',index=False)
    hashes={};tasks=[]
    for name in NAMES:
        path=MODELS/(name+('.pt' if name=='standard_mlp' else '.joblib'))
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        model=MLPModel.load(path) if name=='standard_mlp' else joblib.load(path)['pipeline']
        prediction=model.predict(plants[FEATURES])
        for r,g in zip(plants.itertuples(),prediction):
            tasks.append((name,r.rejection_id,[r.zeta,r.wn,r.tau,r.K],g.tolist(),metadata['gain_penalty']))
    provenance={'plants':len(plants),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
        'gain_penalty':metadata['gain_penalty'],
        'source_dataset_sha256':metadata['dataset_sha256'],'model_hashes':hashes,'training_performed':False,
        'accepted_dataset_overlap':0,'strict_max_points':65537,'extended_max_points':262145,
        'extended_rule':'only strict resolution-limit predictions are retried; tolerances and horizon limits unchanged',
        'interpretation':'selection-biased stress set, not an independent random test set; no optimized target gains available'}
    (OUT/'protocol.json').write_text(json.dumps(provenance,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for result in pool.map(validate,tasks,chunksize=2):
            rows.append(result)
            if len(rows)%50==0:
                pd.DataFrame(rows).to_csv(OUT/'predictions_partial.csv',index=False)
                print(f'{len(rows)}/{len(tasks)} predictions checked',flush=True)
    results=pd.DataFrame(rows).merge(plants,on='rejection_id',validate='many_to_one')
    results.to_csv(OUT/'predictions.csv',index=False)
    for name in NAMES:
        path=MODELS/(name+('.pt' if name=='standard_mlp' else '.joblib'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
    summary=results.groupby('model',sort=False).agg(plants=('accepted','size'),
        strict_passes=('accepted','sum'),stable=('stable','sum'),
        extended_attempts=('extended_attempted','sum'),passes_with_extended_budget=('extended_accepted','sum'))
    summary.to_csv(OUT/'summary.csv');print(summary.to_string())
