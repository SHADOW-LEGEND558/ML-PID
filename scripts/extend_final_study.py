"""Validation-only learning-curve extension and rejected-plant stress test.

The frozen final models and final test partition are never refit or rescored.
Curve prefixes retain the original 7K training plants, then add independent
20K-development plants. Every point uses the same 500 final-validation plants.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
import train_20000 as shared
from stress_rejected_plants import recover_rejections,validate
from benchmark import FEATURES,PLANT_COLUMNS,load_frame
from data_gen import Plant,PlantSimulator,FeatureExtractor
from large_experiment import summarize

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'results/dataset_final_10000'
OUT=FINAL/'website_extension'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from MLP import MLPModel
    OUT.mkdir(exist_ok=True)
    protocol=json.loads((FINAL/'evaluation_protocol.json').read_text(encoding='utf-8'))
    lock=json.loads((FINAL/'selection_lock.json').read_text(encoding='utf-8'))['files']
    final=load_frame(ROOT/protocol['data_path'])
    old=load_frame(ROOT/'data/pid_dataset_20000.csv')
    train=final.loc[protocol['train']].copy()
    # Match the prefixes already used by the final learning curves exactly.
    first=np.random.default_rng(45).permutation(protocol['train'])
    training=pd.concat([final.loc[first],old.iloc[np.random.default_rng(46).permutation(len(old))]],ignore_index=True)
    assert not training.duplicated(PLANT_COLUMNS).any()
    held=final.loc[protocol['validation']+protocol['test'],PLANT_COLUMNS]
    assert not training[PLANT_COLUMNS].merge(held,on=PLANT_COLUMNS).shape[0]
    combined=pd.concat([training,final.loc[protocol['validation']]],ignore_index=True)
    mapping={original:len(training)+i for i,original in enumerate(protocol['validation'])}
    validation=[mapping[i] for i in protocol['validation']]
    curve_validation=[mapping[i] for i in protocol['curve_validation']]
    refs=pd.read_csv(FINAL/'validation_references.csv')
    refmap={mapping[int(r['source_row'])]:{**r,'source_row':mapping[int(r['source_row'])]} for r in refs.to_dict('records')}
    selected=pd.read_csv(FINAL/'selected_before_test.csv').set_index('model')
    settings=shared.specs()
    shared.OUT=OUT
    curves=pd.read_csv(FINAL/'learning_curves.csv').to_dict('records')
    provenance={'source':'first 7000 frozen final-training rows, extended with independent development plants',
        'training_sizes':[1000,3000,5000,7000,10000,15000,20000],
        'validation_source':'same 500 rows from final validation partition at every point',
        'final_test_used':False,'final_models_refit':False,'settings':'fixed at original 7K validation selections',
        'old_development_rows_are_training_only':True,'seed_first':45,'seed_extension':46,
        'axis':'training plants; the chosen 10000-row final dataset uses 7000 training plants'}
    (OUT/'curve_protocol.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    pd.DataFrame({'pool_position':range(len(training)),
        'cohort':['final_training']*7000+['prior_development']*20000,
        'original_row':list(map(int,first))+np.random.default_rng(46).permutation(len(old)).tolist()}).to_csv(OUT/'training_pool.csv',index=False)
    with ProcessPoolExecutor(max_workers=12) as pool:
        for family in settings:
            template=next(e for s,e in settings[family] if s==selected.loc[family,'setting'])
            for size in [10000,15000,20000]:
                print('Extended learning curve:',family,size,flush=True)
                model=shared.fit_model(family,clone(template) if family!='standard_mlp' else None,
                    np.arange(size),validation,combined,f'{family}_learning{size}')
                result,_=shared.evaluate(model,curve_validation,pool,combined,refmap,.01,f'{family} {size}')
                result.to_csv(OUT/f'{family}_learning{size}_validation.csv',index=False)
                curves.append({'model':family,'training_size':size,**summarize(result)})
                pd.DataFrame(curves).to_csv(OUT/'learning_curves.csv',index=False)
        # New 83 rejected plants: frozen final models, no optimizer or fitting.
        plants=recover_rejections((ROOT/'data/generation_final_10000.log').read_text(encoding='utf-8'))
        meta=json.loads((ROOT/'data/pid_dataset_final_10000.meta.json').read_text(encoding='utf-8'))
        assert len(plants)==meta['attempts']-meta['samples']
        assert not plants.duplicated(PLANT_COLUMNS).any()
        assert plants[PLANT_COLUMNS].merge(final[PLANT_COLUMNS],on=PLANT_COLUMNS).empty
        sim=PlantSimulator();extractor=FeatureExtractor()
        for i,r in plants.iterrows():
            t,y=sim.observe_open_loop(Plant(r.zeta,r.wn,r.tau,r.K))
            for k,v in extractor.extract(t,y).items(): plants.loc[i,k]=v
        plants.to_csv(OUT/'rejected_plants.csv',index=False)
        tasks=[]
        for family in settings:
            path=FINAL/f'{family}.{ "pt" if family=="standard_mlp" else "joblib"}'
            model=MLPModel.load(path) if family=='standard_mlp' else joblib.load(path)['pipeline']
            predictions=model.predict(plants[FEATURES])
            tasks.extend((family,int(r.rejection_id),[r.zeta,r.wn,r.tau,r.K],g.tolist(),.01) for r,g in zip(plants.itertuples(),predictions))
        records=[]
        for result in pool.map(validate,tasks,chunksize=2):
            records.append(result)
            if len(records)%50==0:print('Rejected predictions checked:',len(records),'/',len(tasks),flush=True)
        rejected=pd.DataFrame(records).merge(plants,on='rejection_id',validate='many_to_one')
        rejected.to_csv(OUT/'rejected_predictions.csv',index=False)
        rejected.groupby('model').agg(plants=('accepted','size'),strict_passes=('accepted','sum'),
            stable=('stable','sum'),extended_passes=('extended_accepted','sum')).to_csv(OUT/'rejected_summary.csv')
    assert all(digest(FINAL/name)==value for name,value in lock.items())
    (OUT/'verification.json').write_text(json.dumps({'final_lock_unchanged':True,'final_test_rescored':False,
        'curve_points':len(curves),'rejected_plants':len(plants),'rejection_groups':plants.original_reason.value_counts().to_dict()},indent=2),encoding='utf-8')
    print('Website evidence extension complete.',flush=True)

if __name__=='__main__': main()
