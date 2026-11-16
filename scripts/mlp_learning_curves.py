"""Matched sample-size curves for the selected MLP variants; no test scoring."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json,copy,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from train_mlp import cross_validate,OUT


if __name__=='__main__':
    selected=pd.read_csv(OUT/'selected.csv')
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert hashlib.sha256(Path('data/pid_dataset.csv').read_bytes()).hexdigest()==protocol['dataset_sha256']
    frame=pd.read_csv('data/pid_dataset.csv')
    saved=pd.read_csv(OUT/f'{selected.iloc[0]["name"]}_oof.csv')
    references={int(r.source_row):{'accepted':True,'objective':r.reference_objective,'itae':r.reference_itae}
                for r in saved.itertuples()}
    results=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        for chosen in selected.to_dict('records'):
            variant=chosen['variant']
            results.append({**chosen,'training_size':533})
            for size in [128,256]:
                folds=copy.deepcopy(protocol['folds'])
                for fold,items in enumerate(folds):
                    items['fit']=np.random.default_rng(100+fold).permutation(items['fit'])[:size].tolist()
                    assert not set(items['fit'])&set(items['validation'])
                    assert not (set(items['fit'])|set(items['validation']))&set(protocol['excluded_test_rows'])
                options={key:chosen[key] for key in ['beta','dropout','weight_decay','seed']}
                options['seed']=int(options['seed'])
                result=cross_validate(options,f'{variant}_learning{size}_seed42',folds,frame,pool,references,protocol['gain_penalty'])
                results.append({'variant':variant,'training_size':size,**result})
                pd.DataFrame(results).to_csv(OUT/'learning_curves.csv',index=False)
    print('Both MLP sample-size curves complete; holdout remains excluded.',flush=True)
