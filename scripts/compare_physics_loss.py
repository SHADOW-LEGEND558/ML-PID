"""Repeat the historical selected standard-vs-physics MLP comparison on 1K data."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from benchmark import FEATURES,TARGETS,PLANT_COLUMNS,load_frame,gain_metrics
from target_normalization import evaluate_task
OUT=ROOT/'temp/physics_comparison'

def score(records):
    f=pd.DataFrame(records);accepted=f[f.accepted]
    predicted=f[['predicted_'+g for g in TARGETS]].to_numpy();truth=f[['target_'+g for g in TARGETS]].to_numpy()
    return {'plants':len(f),'passes':int(f.accepted.sum()),'success_pct':100*float(f.accepted.mean()),
        'stable_pct':100*float(f.stable.mean()),'bounds_violation_pct':100*float((~f.within_bounds).mean()),
        'median_J_ratio':accepted.objective_ratio.median(),
        'p90_J_ratio':accepted.objective_ratio.quantile(.9),'ratio_pairs':len(accepted),
        'median_ITAE_ratio':accepted.itae_ratio.median(),**gain_metrics(truth,predicted)}


def cross_validate(options,name,folds,frame,pool,references,weight):
    from MLP import MLPModel
    records=[];epochs=[]
    for fold in range(3):
        fit=np.array(folds[fold]['fit']);val=np.array(folds[fold]['validation'])
        model=MLPModel(**options)
        model.fit(frame.loc[fit,FEATURES],frame.loc[fit,TARGETS],frame.loc[fit,PLANT_COLUMNS],
            validation_data=(frame.loc[val,FEATURES],frame.loc[val,TARGETS],frame.loc[val,PLANT_COLUMNS]))
        epochs.append(model.best_epoch)
        pd.DataFrame(model.history).to_csv(OUT/f'{name}_fold{fold}_history.csv',index=False)
        predictions=model.predict(frame.loc[val,FEATURES])
        tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),prediction.tolist(),weight) for i,prediction in zip(val,predictions)]
        for result,prediction in zip(pool.map(evaluate_task,tasks),predictions):
            i=result['source_row'];ref=references[i]
            result.update(fold=fold,reference_objective=ref['objective'],reference_itae=ref['itae'],
                objective_ratio=result['objective']/ref['objective'] if result['accepted'] else None,
                itae_ratio=result['itae']/ref['itae'] if result['accepted'] else None)
            result.update({key:float(frame.loc[i,key]) for key in PLANT_COLUMNS})
            result.update({f'predicted_{g}':float(prediction[j]) for j,g in enumerate(TARGETS)})
            result.update({f'target_{g}':float(frame.loc[i,g]) for g in TARGETS})
            records.append(result)
        print(f'{name}: fold {fold+1}/3, best epoch {model.best_epoch}',flush=True)
    pd.DataFrame(records).to_csv(OUT/f'{name}_oof.csv',index=False)
    result={'name':name,**options,'median_best_epoch':int(np.median(epochs)),**score(records)}
    print(f'{name}: pass={result["success_pct"]:.2f}%, stable={result["stable_pct"]:.2f}%, R2={result["r2"]:.3f}',flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve()
    published=ROOT/'results/dataset_final_10000'
    if OUT==published or published in OUT.parents:raise ValueError('Use a separate output directory.')
    OUT.mkdir(parents=True,exist_ok=True)
    reference=published/'development_reference/physics_comparison'
    protocol=json.loads((reference/'protocol.json').read_text())
    path=ROOT/'data/pid_dataset.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['dataset_sha256']
    frame=load_frame(path)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),frame.loc[i,TARGETS].tolist(),.01) for i in protocol['development_rows']]
        refs={r['source_row']:r for r in pool.map(evaluate_task,tasks,chunksize=8)}
        records=[]
        for variant,beta in [('standard',0.),('physics',1.)]:
            for seed in [42,43,44]:
                options=dict(beta=beta,dropout=.05,weight_decay=.0001,seed=seed)
                result=cross_validate(options,f'{variant}_seed{seed}',protocol['folds'],frame,pool,refs,.01)
                records.append({'variant':variant,**result})
        results=pd.DataFrame(records);results.to_csv(OUT/'seed_results.csv',index=False)
        summary=results.groupby('variant').agg(seeds=('seed','size'),mean_success_pct=('success_pct','mean'),std_success_pp=('success_pct','std'))
        summary.to_csv(OUT/'seed_summary.csv');print(summary.to_string())
