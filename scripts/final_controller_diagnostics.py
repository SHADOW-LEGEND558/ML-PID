"""Complete response diagnostics without rejecting a stable loop for overshoot.

This separate analysis never changes the frozen 10% acceptance flags or models.
Finite bounded/unstable checks, grid/horizon convergence and resource limits are
reported independently. Run with --jobs 12 from any working directory.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_gen import Plant,PIDGains,PlantSimulator,PIDOptimizer
from study import frozen_input_hashes,runtime_signature

BASE=ROOT/'results/dataset_final_10000'
OUT=BASE/'final_checks'
NAMES=['linear','ridge','polynomial_ridge','random_forest','xgboost','standard_mlp','numerical_reference']


def diagnostic(task):
    model,source,parameters,vector,strict=task
    plant=Plant(*parameters)
    gains=PIDGains(*vector)
    sim=PlantSimulator();opt=PIDOptimizer(sim,gain_penalty=.01)
    vector=np.asarray(vector)
    result=dict(model=model,source_row=source,original_strict_accepted=bool(strict),
                response_resolved=False,stable=False,within_bounds=False,reason='nonfinite_gains',
                **{f'gain_{k}':float(v) for k,v in zip(['Kp','Ki','Kd'],vector)})
    if not np.isfinite(vector).all():return result
    result['within_bounds']=bool(np.all(vector>=opt.BOUNDS[:,0]) and np.all(vector<=opt.BOUNDS[:,1]))
    try:
        _,den=sim.closed_loop_coefficients(plant,gains)
        poles=np.roots(den)
        if not np.isfinite(poles).all() or max(poles.real)>=-1e-8:
            return {**result,'reason':'unstable'}
        result.update(stable=True,max_pole_real=float(max(poles.real)))
        duration=max(float(sim.get_dynamic_t_eval(plant)[-1]),12/-float(max(poles.real)))
        previous=None
        for _ in range(5):
            if duration>opt.max_duration:return {**result,'reason':'duration_limit'}
            n=max(4001,int(np.ceil(duration*max(abs(poles.imag))*16/np.pi))+1)
            while 2*n-1<=opt.max_validation_points:
                coarse=opt._response_metrics(plant,gains,duration,n)
                fine=opt._response_metrics(plant,gains,duration,2*n-1)
                tolerance=max(1e-8,.001*fine['itae'])
                if abs(fine['itae']-coarse['itae'])<=tolerance and abs(fine['overshoot_pct']-coarse['overshoot_pct'])<=.001:
                    break
                n=2*n-1
            else:return {**result,'reason':'resolution_limit'}
            # Save finite-window values explicitly as diagnostics until resolved.
            result.update(**fine,duration=duration,points=2*n-1)
            if previous is not None and fine['tail_error']<=.001 and abs(fine['itae']-previous)<=tolerance:
                t=np.linspace(0,duration,2*n-1)
                y=sim.close_loop(plant,gains,t)[1]
                outside=np.where(abs(y-1)>.02)[0]
                if len(outside):
                    last=int(outside[-1])
                    if last+1>=len(t):return {**result,'reason':'settling_not_resolved'}
                    target=1.02 if y[last]>1 else .98
                    def crossing(time):
                        return float(sim.close_loop(plant,gains,np.array([0.,time]))[1][-1]-target)
                    settling=brentq(crossing,float(t[last]),float(t[last+1]),xtol=1e-10)
                else:settling=0.
                result.update(response_resolved=True,reason='resolved',settling_time_2pct=settling,
                              **opt.objective_terms(plant,vector,fine['itae']))
                return result
            previous=fine['itae'];duration*=2
        return {**result,'reason':'tail_not_converged'}
    except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):
        return {**result,'reason':'simulation_failure'}


def main(jobs,output=None):
    global OUT
    OUT=Path(output or ROOT/'temp/controller_diagnostics').resolve()
    if OUT==BASE or BASE in OUT.parents:raise ValueError('Use a new output directory outside the published study.')
    OUT.mkdir(parents=True,exist_ok=True)
    lock=frozen_input_hashes()
    for name,value in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
    frame=pd.read_csv(ROOT/'data/pid_dataset_final_10000.csv')
    protocol=json.loads((BASE/'evaluation_protocol.json').read_text())
    tasks=[]
    for name in NAMES:
        if name=='numerical_reference':
            rows=frame.loc[protocol['test']]
            for source,row in rows.iterrows():
                tasks.append((name,int(source),row[['zeta','wn','tau','K']].tolist(),row[['Kp','Ki','Kd']].tolist(),True))
        else:
            rows=pd.read_csv(BASE/f'{name}_test_predictions.csv')
            for row in rows.itertuples():
                tasks.append((name,int(row.source_row),[row.zeta,row.wn,row.tau,row.K],
                              [row.predicted_Kp,row.predicted_Ki,row.predicted_Kd],row.accepted))
    manifest=dict(implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_signature=runtime_signature(),models=NAMES,plants_per_model=1500,original_criterion_unchanged=True,
        thresholds_pct=[10,12,15,20],limits=dict(duration=1000,points=65537,horizon_iterations=5),
        convergence=dict(itae_relative=.001,peak_percentage_points=.001,tail_absolute=.001),
        settling_band_absolute=.02,plant_model='Ideal PID, second-order Pade; not true-delay certification',
        original_artifact_hashes=lock,
        policy='Overshoot never triggers an early diagnostic exit. Raw gains unchanged; gain bounds are separately reported. Threshold rates require bounded gains and converged response metrics.')
    manifest_path=OUT/'diagnostic_protocol.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())==manifest,'A different diagnostic protocol already exists.'
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2))
    path=OUT/'controller_diagnostics.csv'
    if path.exists():
        results=pd.read_csv(path)
        assert len(results)==len(tasks)
    else:
        rows=[];start=perf_counter()
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(diagnostic,tasks,chunksize=2):
                rows.append(result)
                if len(rows)%500==0:print(f'Diagnostics {len(rows)}/{len(tasks)} ({perf_counter()-start:.1f}s)',flush=True)
        results=pd.DataFrame(rows);results.to_csv(path,index=False)
    reference=results[results.model.eq('numerical_reference')].set_index('source_row')
    results['itae_reference_ratio']=[r.itae/reference.loc[r.source_row,'itae']
        if r.response_resolved and reference.loc[r.source_row,'response_resolved'] else np.nan for r in results.itertuples()]
    results['settling_reference_ratio']=[r.settling_time_2pct/reference.loc[r.source_row,'settling_time_2pct']
        if r.response_resolved and reference.loc[r.source_row,'response_resolved'] and reference.loc[r.source_row,'settling_time_2pct']>0 else np.nan for r in results.itertuples()]
    results.to_csv(path,index=False)
    summaries=[];thresholds=[];bands=[]
    for name,part in results.groupby('model',sort=False):
        resolved=part[part.response_resolved]
        above=resolved[resolved.overshoot_pct.gt(10+1e-6)]
        summaries.append(dict(model=name,plants=len(part),stable=int(part.stable.sum()),
            original_strict_passes=int(part.original_strict_accepted.sum()),resolved=int(part.response_resolved.sum()),
            stable_unresolved=int((part.stable&~part.response_resolved).sum()),
            resolved_over_10=len(above),over_10_median_overshoot=above.overshoot_pct.median(),
            over_10_median_itae_ratio=above.itae_reference_ratio.median(),
            over_10_median_settling_ratio=above.settling_reference_ratio.median(),
            reason_counts=json.dumps(part.reason.value_counts().to_dict(),sort_keys=True)))
        for threshold in [10,12,15,20]:
            passed=part.response_resolved & part.within_bounds & part.overshoot_pct.le(threshold+1e-6)
            thresholds.append(dict(model=name,threshold_pct=threshold,plants=len(part),
                resolved_bounded_within_threshold=int(passed.sum()),pct_all_plants=100*passed.mean()))
        for low,high,label in [(0,10,'<=10%'),(10,12,'10-12%'),(12,15,'12-15%'),(15,20,'15-20%'),(20,np.inf,'>20%')]:
            mask=resolved.overshoot_pct.le(high+1e-6)&(resolved.overshoot_pct.gt(low+1e-6) if low>0 else True)
            group=resolved[mask]
            bands.append(dict(model=name,band=label,plants=len(group),
                median_itae_ratio=group.itae_reference_ratio.median(),p90_itae_ratio=group.itae_reference_ratio.quantile(.9),
                median_settling_seconds=group.settling_time_2pct.median(),
                median_settling_ratio=group.settling_reference_ratio.median()))
    pd.DataFrame(summaries).to_csv(OUT/'diagnostic_summary.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'threshold_sensitivity.csv',index=False)
    pd.DataFrame(bands).to_csv(OUT/'overshoot_band_quality.csv',index=False)
    assert not results.duplicated(['model','source_row']).any()
    # Completing diagnostics must not turn an original strict pass into a failure.
    assert results.loc[results.original_strict_accepted,'response_resolved'].all()
    assert results.loc[results.original_strict_accepted,'overshoot_pct'].le(10+1e-6).all()
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in lock.items())
    (OUT/'diagnostic_verification.json').write_text(json.dumps(dict(frozen_hashes_unchanged=True,
        original_strict_flags_preserved=True,controllers=len(results),
        all_original_passes_still_resolved=True),indent=2))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--jobs',type=int,default=12)
    parser.add_argument('--output',type=Path,default=ROOT/'temp/controller_diagnostics')
    args=parser.parse_args();main(args.jobs,args.output)
