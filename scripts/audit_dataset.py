"""Revalidate saved labels and features without modifying the dataset."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name] = '1'
import sys
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_gen import Plant, PIDGains, PlantSimulator, PIDOptimizer, FeatureExtractor


def audit(task):
    index,row,weight=task
    plant=Plant(*(row[k] for k in ['zeta','wn','tau','K']))
    gains=PIDGains(*(row[k] for k in ['Kp','Ki','Kd']))
    sim=PlantSimulator();opt=PIDOptimizer(sim,gain_penalty=weight)
    metrics,reason=opt._validate_candidate(plant,gains,sim.get_dynamic_t_eval(plant))
    result={'row':index,'validation':reason}
    t,y=sim.observe_open_loop(plant)
    features=FeatureExtractor().extract(t,y)
    result['max_feature_difference']=max(abs(value-row[key]) for key,value in features.items())
    result['duration_difference']=abs(t[-1]-row['observation_duration'])
    result['relative_final_value_error']=abs(row['y_ss']/plant.K-1)
    result['relative_peak_error']=abs(row['y_peak']/plant.K-(1+np.exp(-np.pi*plant.zeta/np.sqrt(1-plant.zeta**2)) if plant.zeta<1 else 1))
    if metrics is not None:
        result.update(metrics)
        dense=opt._response_metrics(plant,gains,metrics['duration'],int(2*metrics['n_points']-1))
        result['dense_overshoot_pct']=dense['overshoot_pct']
        result['dense_itae_relative_change']=abs(dense['itae']-metrics['itae'])/max(1e-8,metrics['itae'])
        result.update(opt.objective_terms(plant,np.array([gains.Kp,gains.Ki,gains.Kd]),metrics['itae']))
    return result


if __name__=='__main__':
    path=Path('data/pid_dataset.csv');df=pd.read_csv(path)
    meta=json.loads(path.with_suffix('.meta.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['dataset_sha256']
    assert len(df)==meta['samples']==1000 and np.isfinite(df.to_numpy()).all()
    assert 'AUC' not in df and not df.duplicated(['zeta','wn','tau','K']).any()
    output=Path('results/dataset_audit');output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        for record in pool.map(audit,[(i,r,meta['gain_penalty']) for i,r in enumerate(df.to_dict('records'))]):
            rows.append(record)
            if len(rows)%100==0: print(f'{len(rows)}/{len(df)} checked',flush=True)
    checked=pd.DataFrame(rows);checked.to_csv(output/'controllers.csv',index=False)
    hits=np.isclose(df[['Kp','Ki','Kd']],np.array([100,100,50]),atol=1e-5,rtol=0).any(axis=1)
    summary={'rows':len(df),'zero_delay_rows':int(df.tau.eq(0).sum()),
        'upper_bound_hits':int(hits.sum()),'validation':checked.validation.value_counts().to_dict(),
        'max_dense_overshoot_pct':float(checked.dense_overshoot_pct.max()),
        'max_dense_itae_relative_change':float(checked.dense_itae_relative_change.max()),
        'max_feature_difference':float(checked.max_feature_difference.max()),
        'max_duration_difference':float(checked.duration_difference.max()),
        'max_relative_final_value_error':float(checked.relative_final_value_error.max()),
        'max_relative_peak_error':float(checked.relative_peak_error.max()),
        'max_tail_error':float(checked.tail_error.max()),
        'parameter_quartile_counts':{k:pd.cut(df[k],bins=np.linspace(a,b,5),include_lowest=True).value_counts(sort=False).tolist()
            for k,a,b in [('K',.5,2),('wn',.5,10),('zeta',.1,2),('tau',0,1)]},
        'dataset_sha256':meta['dataset_sha256']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    assert checked.validation.eq('accepted').all()
    assert checked.dense_overshoot_pct.le(10+1e-6).all()
    assert checked.dense_itae_relative_change.le(.001).all()
    assert checked.max_feature_difference.max()<1e-8
