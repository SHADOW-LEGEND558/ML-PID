"""Shared final-study model specifications, fitting and controller evaluation.

Scientific runners reconstruct selected models in memory from the fixed split;
they never need an original checkpoint or a Python-model archive.
"""
from pathlib import Path
from time import perf_counter
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from benchmark import FEATURES,TARGETS,PLANT_COLUMNS,load_frame,gain_metrics
from target_normalization import EmpiricalTargetRegressor,evaluate_task
from provenance import matches_recorded_hash

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/dataset_final_10000'
OUT=ROOT/'temp/model_training'
MODEL_NAMES=['linear','ridge','polynomial_ridge','random_forest','xgboost','standard_mlp']

def summarize(frame):
    truth=frame[['target_'+g for g in TARGETS]].to_numpy()
    predicted=frame[['predicted_'+g for g in TARGETS]].to_numpy()
    ratios=frame.objective_ratio.dropna()
    return {**gain_metrics(truth,predicted),'plants':len(frame),'passes':int(frame.accepted.sum()),
        'success_pct':100*float(frame.accepted.mean()),'stable_pct':100*float(frame.stable.mean()),
        'bounds_violation_pct':100*float((~frame.within_bounds).mean()),
        'median_J_ratio':float(ratios.median()),'p90_J_ratio':float(ratios.quantile(.9)),
        'ratio_pairs':len(ratios),'reference_passes':int(frame.reference_accepted.sum()),
        'median_ITAE_ratio':float(frame.itae_ratio.median())}


def specs():
    return {
      'linear': [('fixed',EmpiricalTargetRegressor(make_pipeline(StandardScaler(),LinearRegression()),True))],
      'ridge': [('alpha100',EmpiricalTargetRegressor(make_pipeline(StandardScaler(),Ridge(alpha=100)),True))],
      'polynomial_ridge':[('degree2_alpha1',EmpiricalTargetRegressor(make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=1)),True))],
      'random_forest':[(f'depth{depth}_leaf{leaf}',EmpiricalTargetRegressor(RandomForestRegressor(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,n_jobs=6,random_state=42),True)) for depth,leaf in [(12,3),(None,1)]],
      'xgboost':[(f'trees{trees}',EmpiricalTargetRegressor(MultiOutputRegressor(XGBRegressor(n_estimators=trees,max_depth=6,learning_rate=.08,n_jobs=4,random_state=42)),True)) for trees in [150,400]],
      'standard_mlp':[('standard',None)]}


def reference_rows(indices,pool,frame,weight):
    tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),frame.loc[i,TARGETS].tolist(),weight) for i in indices]
    return {r['source_row']:r for r in pool.map(evaluate_task,tasks,chunksize=8)}


def evaluate(model,indices,pool,frame,references,weight,label):
    started=perf_counter();predictions=model.predict(frame.loc[indices,FEATURES]);elapsed=perf_counter()-started
    if not np.isfinite(predictions).all():raise ValueError('Nonfinite model predictions.')
    tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),p.tolist(),weight) for i,p in zip(indices,predictions)]
    records=[]
    for result,prediction in zip(pool.map(evaluate_task,tasks,chunksize=8),predictions):
        i=result['source_row'];ref=references[i]
        paired=result['accepted'] and ref['accepted']
        result.update(reference_accepted=ref['accepted'],reference_reason=ref['reason'],
            reference_itae=ref['itae'],reference_objective=ref['objective'],
            objective_ratio=result['objective']/ref['objective'] if paired else None,
            itae_ratio=result['itae']/ref['itae'] if paired else None)
        result.update({key:float(frame.loc[i,key]) for key in PLANT_COLUMNS+FEATURES})
        result.update({f'predicted_{g}':float(prediction[j]) for j,g in enumerate(TARGETS)})
        result.update({f'target_{g}':float(frame.loc[i,g]) for g in TARGETS})
        records.append(result)
        if len(records)%500==0:print(f'{label}: {len(records)}/{len(indices)} simulations',flush=True)
    return pd.DataFrame(records),elapsed*1000/len(indices)


def fit_model(name,estimator,indices,validation,frame,label):
    if name=='standard_mlp':
        from MLP import MLPModel
        # Same selected architecture/loss; larger batches reduce overhead at 15k.
        model=MLPModel(beta=0,dropout=.05,weight_decay=.0001,batch_size=256,
                       max_epochs=500,patience=50,seed=42)
        model.fit(frame.loc[indices,FEATURES],frame.loc[indices,TARGETS],
                  validation_data=(frame.loc[validation,FEATURES],frame.loc[validation,TARGETS],None))
        pd.DataFrame(model.history).to_csv(OUT/f'{label}_history.csv',index=False)
        return model
    estimator.fit(frame.loc[indices,FEATURES],frame.loc[indices,TARGETS].to_numpy())
    return estimator


def key(r):return (-r['passes'],r['median_J_ratio'] if np.isfinite(r['median_J_ratio']) else np.inf)


def validate_split(protocol, rows=10000):
    parts=[list(protocol[k]) for k in ['train','validation','test']]
    if rows!=10000 or list(map(len,parts))!=[7000,1500,1500]:
        raise ValueError('Expected exactly 7000/1500/1500 plants.')
    combined=[i for part in parts for i in part]
    if len(set(combined))!=rows or set(combined)!=set(range(rows)):
        raise ValueError('Split rows overlap, repeat, or do not cover the dataset.')
    return parts


def selected_models(names=None):
    """Reconstruct published selected configurations; no checkpoint/result writes."""
    names=MODEL_NAMES if names is None else list(names)
    protocol=json.loads((BASE/'evaluation_protocol.json').read_text())
    path=ROOT/protocol['data_path']
    if not matches_recorded_hash(path,protocol['dataset_sha256']):
        raise ValueError('Final dataset checksum mismatch.')
    frame=load_frame(path)
    train,validation,_=validate_split(protocol,len(frame))
    settings=pd.read_csv(BASE/'selected_before_test.csv').set_index('model')
    result={}
    for name in names:
        if name not in MODEL_NAMES:raise ValueError('Unknown model: '+name)
        print('Reconstructing selected model:',name,flush=True)
        if name=='standard_mlp':
            from MLP import MLPModel
            model=MLPModel(beta=0,dropout=.05,weight_decay=.0001,batch_size=256,
                           max_epochs=500,patience=50,seed=42,verbose=False)
            model.fit(frame.loc[train,FEATURES],frame.loc[train,TARGETS],
                      validation_data=(frame.loc[validation,FEATURES],frame.loc[validation,TARGETS],None))
        else:
            template=next(m for label,m in specs()[name] if label==settings.loc[name,'setting'])
            model=clone(template)
            model.fit(frame.loc[train,FEATURES],frame.loc[train,TARGETS].to_numpy())
        result[name]=model
    return result


def frozen_input_hashes():
    """Guard published tabular evidence and protocols, not undistributed weights."""
    paths=[BASE/n for n in ['selection_lock.json','protocol.json','evaluation_protocol.json','test_metrics.csv']]
    paths+=list(BASE.glob('*_test_predictions.csv'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def model_fingerprints(models):
    """Fingerprint fitted state in memory, without writing checkpoint files."""
    import joblib
    result={}
    for name,model in models.items():
        if name!='standard_mlp':
            result[name]=joblib.hash(model,hash_name='sha1')
            continue
        digest=hashlib.sha256(json.dumps(model.options,sort_keys=True).encode())
        arrays={key:np.asarray(getattr(model,key)) for key in ['mean','std','qscale','logstd']}
        arrays.update({key:value.detach().cpu().numpy() for key,value in model.module.state_dict().items()})
        for key,array in sorted(arrays.items()):
            digest.update(json.dumps([key,array.dtype.str,array.shape]).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        result[name]=digest.hexdigest()
    return result


def runtime_signature(models=None):
    from importlib.metadata import version
    sources=[ROOT/'src'/name for name in ['data_gen.py','benchmark.py','MLP.py','target_normalization.py','study.py','provenance.py']]
    return dict(versions={name:version(name) for name in ['numpy','scipy','scikit-learn','xgboost','torch']},
                source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                model_fingerprints=model_fingerprints(models) if models is not None else {})
