"""Warm prediction timing for saved models; no fitting or control simulation."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import FEATURES
from MLP import MLPModel
import torch


if __name__=='__main__':
    torch.set_num_threads(2)
    root=Path('results/dataset_20000');protocol=json.loads((root/'protocol.json').read_text())
    frame=pd.read_csv(protocol['data_path']);X=frame.loc[protocol['test'],FEATURES]
    fits=pd.read_csv(root/'selected_before_test.csv').set_index('model')
    output=root/'computational_cost';output.mkdir(exist_ok=True)
    rows=[];structure={}
    for name in ['linear','ridge','polynomial_ridge','random_forest','xgboost','standard_mlp']:
        path=root/(name+('.pt' if name=='standard_mlp' else '.joblib'))
        model=MLPModel.load(path) if name=='standard_mlp' else joblib.load(path)['pipeline']
        # Inputs are pre-extracted features. Timings include wrappers and decoding,
        # but exclude model load, feature extraction, and closed-loop validation.
        query=X.iloc[[0]]
        for _ in range(5):model.predict(query)
        for _ in range(3):model.predict(X)
        singles=[];batches=[]
        for index in np.random.default_rng(47).integers(0,len(X),50):
            query=X.iloc[[index]]
            start=perf_counter_ns();model.predict(query);singles.append((perf_counter_ns()-start)/1e6)
        for _ in range(10):
            start=perf_counter_ns();model.predict(X);batches.append((perf_counter_ns()-start)/1e6)
        rows.append({'model':name,'fit_seconds_recorded':fits.loc[name,'fit_seconds'],
            'single_median_ms':float(np.median(singles)),'single_p95_ms':float(np.quantile(singles,.95)),
            'batch2500_median_ms':float(np.median(batches)),
            'batch_amortized_us_per_plant':float(1000*np.median(batches)/len(X)),
            'artifact_bytes':path.stat().st_size,'artifact_MiB':path.stat().st_size/(1024**2)})
        if name=='random_forest':
            forest=model.estimator_
            structure[name]={'trees':len(forest.estimators_),
                'nodes':int(sum(t.tree_.node_count for t in forest.estimators_)),
                'max_depth':max(t.get_depth() for t in forest.estimators_),
                'mean_max_depth':float(np.mean([t.get_depth() for t in forest.estimators_])),
                'leaves':int(sum(t.get_n_leaves() for t in forest.estimators_))}
        elif name=='xgboost':
            ensemble=model.estimator_.estimators_
            structure[name]={'output_models':len(ensemble),'trees_per_output':[len(e.get_booster().get_dump()) for e in ensemble],
                'max_depth':6,'booster_config':json.loads(ensemble[0].get_booster().save_config())}
        elif name=='standard_mlp':
            structure[name]={'parameters':sum(p.numel() for p in model.module.parameters()),
                'weights':sum(p.numel() for key,p in model.module.named_parameters() if key.endswith('weight')),
                'architecture':[5,128,64,32,3],'best_epoch':model.best_epoch,
                'epochs_executed':len(pd.read_csv(root/'standard_mlp_standard_history.csv')),
                'torch_threads':torch.get_num_threads()}
        else:
            estimator=model.estimator_[-1]
            structure[name]={'coefficients':int(estimator.coef_.size),'intercepts':int(estimator.intercept_.size)}
        print(name,'timed',flush=True)
    pd.DataFrame(rows).to_csv(output/'timings.csv',index=False)
    (output/'structure.json').write_text(json.dumps(structure,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
