"""Sequential per-plant optimizer versus frozen-model timing; no retraining."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import FEATURES,assess_controller
from data_gen import Plant,PIDOptimizer,PlantSimulator
from MLP import MLPModel
import torch


if __name__=='__main__':
    import time
    while not Path('results/dataset_final_10000/website_extension/verification.json').exists():
        time.sleep(5)
    torch.set_num_threads(2)
    root=Path('results/dataset_final_10000');output=root/'tuning_speed';output.mkdir(exist_ok=True)
    protocol=json.loads((root/'evaluation_protocol.json').read_text());frame=pd.read_csv(protocol['data_path'])
    available=frame.loc[protocol['validation']]
    sample=pd.concat([available[available.tau.eq(0)].sample(2,random_state=48),
                      available[available.tau.gt(0)].sample(18,random_state=48)])
    sample.to_csv(output/'plants.csv',index_label='source_row')
    sim=PlantSimulator();r=sample.iloc[0];warm=Plant(r.zeta,r.wn,r.tau,r.K)
    PIDOptimizer(sim,n_starts=1).optimize(warm,sim.get_dynamic_t_eval(warm))
    optimizations=[]
    for index,r in sample.iterrows():
        plant=Plant(r.zeta,r.wn,r.tau,r.K);times=sim.get_dynamic_t_eval(plant)
        optimizer=PIDOptimizer(sim,n_starts=4,gain_penalty=protocol['gain_penalty'])
        started=perf_counter();gains=optimizer.optimize(plant,times);elapsed=perf_counter()-started
        optimizations.append({'source_row':int(index),'optimizer_seconds':elapsed,'optimizer_accepted':gains is not None,
            'starts':len(optimizer.initial_guesses),'iterations':sum(d['iterations'] for d in optimizer.diagnostics),
            'optimizer_J':optimizer.last_validation['objective'] if gains is not None else None})
        print(f'Optimizer {len(optimizations)}/20: {elapsed:.3f}s, accepted={gains is not None}',flush=True)
    optimization=pd.DataFrame(optimizations).set_index('source_row');optimization.to_csv(output/'optimizer.csv')
    rows=[]
    for name in ['linear','ridge','polynomial_ridge','random_forest','xgboost','standard_mlp']:
        path=root/(name+('.pt' if name=='standard_mlp' else '.joblib'))
        model=MLPModel.load(path) if name=='standard_mlp' else joblib.load(path)['pipeline']
        for _ in range(3):model.predict(sample.iloc[[0]][FEATURES])
        for index,r in sample.iterrows():
            X=sample.loc[[index],FEATURES];durations=[]
            for _ in range(5):
                started=perf_counter();prediction=model.predict(X)[0];durations.append(perf_counter()-started)
            started=perf_counter()
            result=assess_controller(Plant(r.zeta,r.wn,r.tau,r.K),prediction,protocol['gain_penalty'])
            validation_time=perf_counter()-started
            elapsed=float(np.median(durations));opt=optimization.loc[index]
            rows.append({'model':name,'source_row':int(index),'prediction_ms':1000*elapsed,
                'validation_ms':1000*validation_time,'prediction_plus_validation_ms':1000*(elapsed+validation_time),
                'accepted':result['accepted'],'reason':result['reason'],'optimizer_seconds':opt.optimizer_seconds,
                'prediction_speedup':opt.optimizer_seconds/elapsed,
                'verified_speedup':opt.optimizer_seconds/(elapsed+validation_time) if result['accepted'] and opt.optimizer_accepted else None})
        print(name,'prediction and validation measured.',flush=True)
    details=pd.DataFrame(rows);details.to_csv(output/'details.csv',index=False)
    summary=details.groupby('model',sort=False).agg(plants=('accepted','size'),passes=('accepted','sum'),
        median_prediction_ms=('prediction_ms','median'),median_validation_ms=('validation_ms','median'),
        median_prediction_plus_validation_ms=('prediction_plus_validation_ms','median'),
        median_prediction_speedup=('prediction_speedup','median'),
        median_verified_speedup=('verified_speedup','median'),verified_pairs=('verified_speedup','count'))
    summary.to_csv(output/'summary.csv');print('Optimizer median seconds:',optimization.optimizer_seconds.median())
    print(summary.to_string())
