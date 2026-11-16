"""Train all five baselines and standard MLP with a frozen 15000/2500/2500 split."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import FEATURES,TARGETS,PLANT_COLUMNS,load_frame
from control_experiments import EmpiricalTargetRegressor,evaluate_task
from large_experiment import OUT,make_split,summarize


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


if __name__=='__main__':
    from sklearn.base import clone
    OUT.mkdir(parents=True,exist_ok=True)
    data=Path('data/pid_dataset_20000.csv');frame=load_frame(data)
    meta=json.loads(data.with_suffix('.meta.json').read_text());digest=hashlib.sha256(data.read_bytes()).hexdigest()
    assert meta['dataset_sha256']==digest and meta['gain_penalty']==.01
    split=make_split(frame);train=split['train'];val=split['validation'];test=split['test']
    curve_val=train_test_split(val,train_size=500,stratify=frame.loc[val,'tau'].eq(0),random_state=44)[0]
    protocol={**split,'curve_validation':list(map(int,curve_val)),'data_path':str(data),'dataset_sha256':digest,
        'gain_penalty':meta['gain_penalty'],'seed':42,'features':FEATURES,'targets':TARGETS,
        'selection':'validation full-pass count, then median accepted J ratio',
        'test_policy':'score once after selections locked; no post-test retuning',
        'learning_sizes':[1000,5000,10000,15000],
        'learning_policy':'nested random training subsets; fixed 500 stratified validation plants; selected settings fixed',
        'mlp_settings':{'architecture':[5,128,64,32,3],'batch_size':256,'max_epochs':500,'patience':50,
                        'dropout':.05,'weight_decay':.0001,'beta':0,'seed':42},
        'strata_counts':{stage:{'zero':int(frame.loc[ids,'tau'].eq(0).sum()),'positive':int(frame.loc[ids,'tau'].gt(0).sum())} for stage,ids in split.items()}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Frozen exact 15000/2500/2500 split, stratified by zero delay.',flush=True)
    selected=[];candidates=[];models={};templates={};curves=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        val_refs=reference_rows(val,pool,frame,meta['gain_penalty'])
        print('Validation reference controllers checked.',flush=True)
        for family,settings in specs().items():
            choices=[]
            for setting,estimator in settings:
                name=f'{family}_{setting}';started=perf_counter()
                print('Fitting',name,flush=True)
                model=fit_model(family,estimator,train,val,frame,name)
                fit_seconds=perf_counter()-started
                result,latency=evaluate(model,val,pool,frame,val_refs,meta['gain_penalty'],name)
                result.to_csv(OUT/f'{name}_validation.csv',index=False)
                record={'model':family,'setting':setting,'candidate':name,'fit_seconds':fit_seconds,
                        'batch_prediction_ms_per_plant':latency,**summarize(result)}
                choices.append((record,model,estimator));candidates.append(record)
                pd.DataFrame(candidates).to_csv(OUT/'validation_candidates.csv',index=False)
                print(name,'validation pass',record['success_pct'],flush=True)
            winner,model,template=min(choices,key=lambda choice:key(choice[0]));selected.append(winner)
            models[family]=model;templates[family]=template
            if family=='standard_mlp':model.save(OUT/f'{family}.pt',metadata={'protocol':protocol,'selected':winner})
            else:joblib.dump({'pipeline':model,'features':FEATURES,'targets':TARGETS,'protocol':protocol,'selected':winner},OUT/f'{family}.joblib')
            chosen=pd.read_csv(OUT/f'{winner["candidate"]}_validation.csv')
            chosen.to_csv(OUT/f'{family}_validation_predictions.csv',index=False)
            curves.append({'model':family,'training_size':15000,**summarize(chosen[chosen.source_row.isin(curve_val)])})
        # Selection is written before any test scoring and is never changed afterward.
        pd.DataFrame(selected).to_csv(OUT/'selected_before_test.csv',index=False)
        print('All settings frozen. Starting one-time test evaluation.',flush=True)
        test_refs=reference_rows(test,pool,frame,meta['gain_penalty']);test_metrics=[]
        for family,model in models.items():
            result,latency=evaluate(model,test,pool,frame,test_refs,meta['gain_penalty'],f'{family} TEST')
            result.to_csv(OUT/f'{family}_test_predictions.csv',index=False)
            test_metrics.append({'model':family,'batch_prediction_ms_per_plant':latency,**summarize(result)})
            pd.DataFrame(test_metrics).to_csv(OUT/'test_metrics.csv',index=False)
            print(family,'test evaluation saved; no settings changed.',flush=True)
        # Learning curves use validation only; do not use test results for changes.
        nested=np.random.default_rng(45).permutation(train)
        for family in models:
            for size in [1000,5000,10000]:
                rows=nested[:size]
                print('Learning curve',family,size,flush=True)
                model=fit_model(family,clone(templates[family]) if family!='standard_mlp' else None,
                                rows,val,frame,f'{family}_learning{size}')
                result,_=evaluate(model,curve_val,pool,frame,val_refs,meta['gain_penalty'],f'{family} learning {size}')
                result.to_csv(OUT/f'{family}_learning{size}_validation.csv',index=False)
                curves.append({'model':family,'training_size':size,**summarize(result)})
                pd.DataFrame(curves).to_csv(OUT/'learning_curves.csv',index=False)
        pd.DataFrame(curves).to_csv(OUT/'learning_curves.csv',index=False)
    print('20,000-row experiment complete.',flush=True)
