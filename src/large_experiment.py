"""Utilities for the frozen 20,000-row train/validation/test experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from benchmark import FEATURES,TARGETS,gain_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/dataset_20000'


def make_split(frame,seed=42):
    if len(frame)!=20000:raise ValueError('This experiment requires exactly 20,000 rows.')
    if frame.duplicated(['zeta','wn','tau','K']).any():raise ValueError('Duplicate plants must be grouped before splitting.')
    rows=np.arange(len(frame));strata=frame.tau.eq(0).to_numpy()
    train,held=train_test_split(rows,train_size=15000,random_state=seed,stratify=strata)
    val,test=train_test_split(held,test_size=2500,random_state=seed+1,stratify=strata[held])
    return {'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()}


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


def load_models_results(names,stage='test'):
    from plot_baselines import LABELS
    LABELS['standard_mlp']='Standard MLP'
    predictions={name:pd.read_csv(OUT/f'{name}_{stage}_predictions.csv').sort_values('source_row') for name in names}
    metrics=pd.DataFrame([{'model':name,**summarize(f)} for name,f in predictions.items()]).set_index('model')
    metrics['validated_success_pct']=metrics.success_pct
    return metrics,predictions,json.loads((OUT/'protocol.json').read_text())
