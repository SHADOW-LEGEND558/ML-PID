"""Frozen-model robustness experiments; writes only a separate results folder.

Run with python scripts/run_stress_suite.py --jobs 4 --output temp/stress_tests.
Selected models are retrained once in memory; no stress-case tuning or gain clipping.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import bisect
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, step
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark import FEATURES, TARGETS, assess_controller
from study import selected_models, frozen_input_hashes, runtime_signature
from data_gen import Plant, PIDGains, PlantSimulator, FeatureExtractor, PIDOptimizer

FROZEN = ROOT / 'results/dataset_final_10000'
OUT = FROZEN / 'stress_tests'
MODELS = ['random_forest', 'xgboost', 'standard_mlp']
ALL = MODELS + ['numerical_reference']
PARAMS = ['zeta', 'wn', 'tau', 'K']
SEED = 17112026


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_case(task):
    case, parameters, fraction, replicate, seed = task
    plant = Plant(*parameters)
    sim = PlantSimulator()
    clean_t, clean_y = sim.observe_open_loop(plant)
    t = np.arange(64001) * .01
    _, y = sim.open_loop(plant, t)
    # One continuous recording per case: shorter windows are identical prefixes.
    y += np.random.default_rng(seed).normal(0, fraction * plant.K, len(t))
    duration, observed = 10, None
    while duration <= 640:
        size = int(duration * 100) + 1
        tail = y[:size][int(.8 * size):]
        scale, mid = abs(float(tail.mean())), len(tail) // 2
        if (scale > 1e-8 and np.ptp(tail) <= .002 * scale
                and abs(tail[:mid].mean() - tail[mid:].mean()) <= .001 * scale):
            observed = duration
            break
        duration *= 2
    # Diagnostic isolation: use the recording length needed in the noiseless case.
    # This is NOT an end-to-end success claim when noisy acquisition fails.
    features = FeatureExtractor().extract(clean_t, y[:len(clean_t)])
    return dict(case_id=case, noise_fraction=fraction, replicate=replicate,
                acquisition_passed=observed is not None,
                noisy_recording_duration=observed,
                fixed_recording_duration=float(clean_t[-1]), **features)


def boundary_teacher(task):
    case, parameters = task
    plant = Plant(*parameters)
    sim = PlantSimulator()
    opt = PIDOptimizer(sim, gain_penalty=.01)
    gains = opt.optimize(plant, sim.get_dynamic_t_eval(plant))
    return dict(case_id=case, teacher_found=gains is not None,
                teacher_reasons=json.dumps(Counter(r.get('validation', 'unknown') for r in opt.diagnostics)),
                **{k: getattr(gains, k) if gains is not None else np.nan for k in TARGETS})


def assess_task(task):
    info, parameters, gains = task
    info = {**info, **{f'gain_{key}':float(value) for key,value in zip(TARGETS,gains)}}
    if not np.isfinite(gains).all():
        return {**info, 'accepted': False, 'stable': None,
                'reason': 'features_unavailable' if info['test'] == 'noise' else 'teacher_unavailable'}
    return {**info, **assess_controller(Plant(*parameters), np.array(gains), .01)}


class TrueDelayResponse:
    """Method of steps for ideal PID, with the derivative-step impulse included.

    x=(y, y_dot, integral(error)). Delayed controller action drives the plant.
    Kd*delta(t) causes a velocity jump K*wn**2*Kd at exactly t=tau.
    solve_ivp integrates ordinary equations on each delay-length interval;
    previously completed dense solutions supply the history, never Padé.
    """
    def __init__(self, plant, gains, duration, rtol=1e-7):
        self.plant, self.gains, self.duration = plant, gains, duration
        self.solutions, self.starts = [], []
        self.failed = None
        w2, a, b = plant.wn**2, 2 * plant.zeta * plant.wn, plant.K * plant.wn**2
        kp, ki, kd = gains
        if plant.tau == 0:
            def rhs(t, x):
                y, v, z = x
                return [v, -w2*y-a*v+b*(kp*(1-y)+ki*z-kd*v), 1-y]
            result = solve_ivp(rhs, (0, duration), [0, b*kd, 0],
                               method='DOP853', rtol=rtol, atol=rtol*.01,
                               dense_output=True, max_step=min(.1, .5/plant.wn))
            self.starts, self.solutions = [0.], [result.sol]
            if not result.success:
                self.failed = 'integration_failure'
            return
        if np.ceil(duration / plant.tau) > 6000:
            self.failed = 'delay_segment_limit'
            return
        state = np.array([0., b*kd, plant.tau])
        left = plant.tau
        while left < duration - 1e-12:
            right = min(left + plant.tau, duration)
            def rhs(t, x):
                yd, vd, zd = self.history(t - plant.tau)
                y, v, z = x
                return [v, -w2*y-a*v+b*(kp*(1-yd)+ki*zd-kd*vd), 1-y]
            def runaway(t, x):
                return 1e6 - max(abs(x[0]), abs(x[1]) / plant.wn)
            runaway.terminal = True
            runaway.direction = -1
            result = solve_ivp(rhs, (left, right), state, method='DOP853',
                               rtol=rtol, atol=rtol*.01, dense_output=True,
                               max_step=min(plant.tau, .5/plant.wn), events=runaway)
            if not result.success or result.status == 1:
                self.failed = 'observed_divergence' if result.status == 1 else 'integration_failure'
                return
            self.starts.append(left)
            self.solutions.append(result.sol)
            state, left = result.y[:, -1], right

    def history(self, t):
        if t < self.plant.tau or not self.starts:
            return np.array([0., 0., max(0., t)])
        index = max(0, bisect.bisect_right(self.starts, t) - 1)
        return self.solutions[index](t)

    def output(self, times):
        times = np.asarray(times)
        y = np.zeros_like(times, dtype=float)
        indices = np.searchsorted(self.starts, times, side='right') - 1
        for index in np.unique(indices):
            if index >= 0:
                mask = indices == index
                y[mask] = self.solutions[index](times[mask])[0]
        return y


def trajectory_metrics(response, duration, points):
    t = np.linspace(0, duration, points)
    y = response.output(t)
    peak = max(1., float(y.max()))
    local = find_peaks(y)[0]
    for i in sorted(local, key=lambda k: y[k], reverse=True)[:8]:
        result = minimize_scalar(lambda time: -float(response.output([time])[0]),
                                 bounds=(t[i-1], t[i+1]), method='bounded')
        if result.success:
            peak = max(peak, -float(result.fun))
    return dict(overshoot_pct=100*(peak-1), itae=float(simpson(t*abs(1-y), x=t)),
                tail_error=float(abs(1-y[int(.9*len(y)):]).max()))


def exact_task(task):
    info, parameters, gains = task
    plant = Plant(*parameters)
    base = dict(**info, accepted=False, reason='duration_limit',
                pole_stability_certified=False, bounded_in_window=None)
    if not np.isfinite(gains).all() or np.any(gains < PIDOptimizer.BOUNDS[:, 0]) or np.any(gains > PIDOptimizer.BOUNDS[:, 1]):
        return {**base, 'reason': 'invalid_gains'}
    sim = PlantSimulator()
    poles = np.roots(sim.closed_loop_coefficients(plant, PIDGains(*gains))[1])
    duration = float(sim.get_dynamic_t_eval(plant)[-1])
    if max(poles.real) < -1e-8:
        duration = max(duration, 12 / -max(poles.real))
    previous = None
    for _ in range(5):
        if duration > 1000:
            return {**base, 'reason': 'duration_limit'}
        coarse = TrueDelayResponse(plant, gains, duration, 1e-7)
        fine = TrueDelayResponse(plant, gains, duration, 1e-9)
        if coarse.failed or fine.failed:
            reason = fine.failed or coarse.failed
            return {**base, 'reason': reason, 'bounded_in_window': False if reason == 'observed_divergence' else None}
        n = 8001
        # Two integration tolerances AND two output grids must agree.
        while n <= 65537:
            cm = trajectory_metrics(coarse, duration, (n+1)//2)
            fm = trajectory_metrics(fine, duration, n)
            tolerance = max(1e-8, .001 * fm['itae'])
            if abs(cm['itae']-fm['itae']) <= tolerance and abs(cm['overshoot_pct']-fm['overshoot_pct']) <= .001:
                break
            n = 2*n-1
        else:
            return {**base, 'reason': 'resolution_limit'}
        base.update(**fm, bounded_in_window=True, duration=duration, points=n)
        if fm['overshoot_pct'] > 10 + 1e-6:
            return {**base, 'reason': 'overshoot'}
        if previous is not None and fm['tail_error'] <= .001 and abs(previous-fm['itae']) <= tolerance:
            return {**base, 'accepted': True, 'reason': 'time_domain_pass'}
        previous = fm['itae']
        duration *= 2
    return {**base, 'reason': 'tail_not_converged'}


def disturbance_task(task):
    info, parameters, gains = task
    plant = Plant(*parameters)
    base = {**info, 'recovered': False, 'stable': False}
    if not np.isfinite(gains).all():
        return {**base, 'reason': 'invalid_gains'}
    sim = PlantSimulator()
    den = sim.closed_loop_coefficients(plant, PIDGains(*gains))[1]
    poles = np.roots(den)
    if max(poles.real) >= -1e-8:
        return {**base, 'reason': 'unstable'}
    base['stable'] = True
    # Input load is added before G (including delay), after nominal settling.
    # G/(1+CG); amplitude .1/K gives .1 steady output change without feedback.
    delay_num = [plant.tau**2/12, -plant.tau/2, 1.] if plant.tau > 1e-5 else [1.]
    num = np.polymul(.1 * plant.wn**2 * np.array(delay_num), [1., 0.]) / delay_num[0]
    system = sim._state_space(num, den)
    duration = max(10., 12/-max(poles.real), float(sim.get_dynamic_t_eval(plant)[-1]))
    previous = None
    for _ in range(5):
        if duration > 1000:
            return {**base, 'reason': 'duration_limit'}
        n = max(4001, int(np.ceil(duration * max(abs(poles.imag))*16/np.pi))+1)
        while 2*n-1 <= 65537:
            metrics = []
            for count in [n, 2*n-1]:
                t = np.linspace(0, duration, count)
                y = step(system, T=t)[1]
                metrics.append((float(simpson(t*abs(y), x=t)), float(abs(y).max())))
            if abs(metrics[1][0]-metrics[0][0]) <= max(1e-8, .001*metrics[1][0]) and abs(metrics[1][1]-metrics[0][1]) <= 1e-5:
                break
            n = 2*n-1
        else:
            return {**base, 'reason': 'resolution_limit'}
        itae, peak = metrics[1]
        tail = float(abs(y[int(.9*len(y)):]).max())
        outside = np.where(abs(y) > .01)[0]
        recovery = float(t[outside[-1]+1]) if len(outside) and outside[-1]+1 < len(t) else (0. if not len(outside) else None)
        base.update(disturbance_itae=itae, max_deviation=peak,
                    recovery_time=recovery, tail_error=tail, duration=duration)
        if previous is not None and tail <= .001 and abs(itae-previous) <= max(1e-8, .001*itae):
            return {**base, 'recovered': True, 'reason': 'recovered'}
        previous, duration = itae, duration*2
    return {**base, 'reason': 'tail_not_converged'}


def map_saved(pool, function, tasks, filename):
    path = OUT / filename
    if path.exists():
        print('Reusing completed', filename, flush=True)
        return pd.read_csv(path)
    started, rows = perf_counter(), []
    for row in pool.map(function, tasks, chunksize=1):
        rows.append(row)
        if len(rows) % 100 == 0:
            print(filename, len(rows), '/', len(tasks), f'({perf_counter()-started:.1f}s)', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    print('Saved', filename, len(result), f'({perf_counter()-started:.1f}s)', flush=True)
    return result


def main(args):
    global OUT
    OUT=Path(args.output).resolve()
    if OUT==FROZEN or FROZEN in OUT.parents:raise ValueError('Use a new output directory outside the published study.')
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((FROZEN/'evaluation_protocol.json').read_text())
    frame = pd.read_csv(ROOT/'data/pid_dataset_final_10000.csv')
    selected = train_test_split(protocol['test'], train_size=args.plants,
                                random_state=SEED, stratify=frame.loc[protocol['test'], 'tau'].eq(0))[0]
    selected = sorted(map(int, selected))
    hashes = frozen_input_hashes()
    models = selected_models(MODELS)
    import torch
    torch.set_num_threads(1)  # Match the original stress-study inference setting.
    plan = dict(implementation_sha256=digest(Path(__file__)),runtime_signature=runtime_signature(models),seed=SEED, base_rows=selected, plants=args.plants, models=ALL,
        frozen_hashes=hashes, dataset_sha256=digest(ROOT/'data/pid_dataset_final_10000.csv'),
        noise_fractions=[.0001,.0005,.001,.005,.01,.02], noise_replicates=3,
        uncertainty_fractions=[.05,.10,.20], uncertainty_replicates=5,
        noise_protocol='IID Gaussian sigma=fraction*K added to full 0.01s response. Original acquisition rule checked to 640s. Prediction study conditions on noiseless recording length; no smoothing.',
        uncertainty_protocol='Independent uniform multiplicative changes of all four parameters; nominal zero delay stays zero. Same perturbation per controller; no retuning.',
        exact_delay_protocol='Method of steps; ideal derivative impulse at tau; DOP853 rtol 1e-7 vs 1e-9, output grids refined. 1000s / 6000 segments / 65537 points. Finite-window convergence is NOT pole stability certification.',
        disturbance_protocol='Steady-state load step .1/K before plant including delay; G/(1+CG). Recovery band +/-0.01 of unit setpoint, tail <=.001. No setpoint overshoot acceptance criterion applied.',
        interpretation='Additional post-test stress study on a fixed stratified subset, not a new independent main accuracy estimate. Repeated cases cluster by plant. No training, clipping, fallback, or replacement of frozen artifacts.',
        sources=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html',
                 'https://www.mathworks.com/help/slcontrol/ug/analyze-the-design-in-the-pid-tuner.html'])
    plan_path = OUT/'protocol.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'A different stress protocol already exists.'
    else:
        plan_path.write_text(json.dumps(plan, indent=2))
    cohort = frame.loc[selected].copy()
    cohort.index.name='source_row'
    cohort.to_csv(OUT/'cohort.csv')
    nominal = {m:model.predict(cohort[FEATURES]) for m,model in models.items()}
    nominal['numerical_reference']=cohort[TARGETS].to_numpy()
    rng = np.random.default_rng(SEED)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        # 1. Realistic noisy recordings, then unchanged feature extraction.
        tasks, noisy_parameters = [], []
        for source,row in cohort.iterrows():
            for level in plan['noise_fractions']:
                for replicate in range(3):
                    case=f'{source}_noise_{level}_{replicate}'
                    parameters=row[PARAMS].tolist()
                    tasks.append((case,parameters,level,replicate,int(rng.integers(2**32))))
                    noisy_parameters.append(dict(case_id=case,source_row=source,**dict(zip(PARAMS,parameters))))
        noisy = map_saved(pool,noise_case,tasks,'noise_features.csv').merge(pd.DataFrame(noisy_parameters),on='case_id')
        valid = np.isfinite(noisy[FEATURES]).all(axis=1) & noisy.t_rise.gt(0) & noisy.y_ss.gt(0)
        noisy['features_usable'] = valid
        noisy.to_csv(OUT/'noise_acquisition.csv',index=False)
        tasks=[]
        for m in ALL:
            predicted=np.full((len(noisy),3),np.nan)
            if m=='numerical_reference':
                predicted=frame.loc[noisy.source_row,TARGETS].to_numpy()
            elif valid.any():
                predicted[valid]=models[m].predict(noisy.loc[valid,FEATURES])
            for (_,row),gains in zip(noisy.iterrows(),predicted):
                info=dict(test='noise',model=m,case_id=row.case_id,source_row=row.source_row,
                          level=row.noise_fraction,replicate=row.replicate,
                          acquisition_passed=row.acquisition_passed,features_usable=row.features_usable)
                tasks.append((info,row[PARAMS].tolist(),gains.tolist()))
        map_saved(pool,assess_task,tasks,'noise_results.csv')
        # 2. Hold nominal gains fixed while all plant parameters move.
        cases=[]
        for position,(source,row) in enumerate(cohort.iterrows()):
            for level in plan['uncertainty_fractions']:
                for replicate in range(5):
                    factors=rng.uniform(1-level,1+level,4)
                    cases.append(dict(source_row=source,position=position,level=level,replicate=replicate,
                        case_id=f'{source}_uncertainty_{level}_{replicate}',
                        **dict(zip(PARAMS,row[PARAMS].to_numpy()*factors))))
        perturbed=pd.DataFrame(cases)
        perturbed.to_csv(OUT/'uncertainty_plants.csv',index=False)
        tasks=[]
        for m in ALL:
            for r in perturbed.itertuples():
                info=dict(test='uncertainty',model=m,source_row=r.source_row,case_id=r.case_id,
                          level=r.level,replicate=r.replicate)
                tasks.append((info,[getattr(r,p) for p in PARAMS],nominal[m][r.position].tolist()))
        map_saved(pool,assess_task,tasks,'uncertainty_results.csv')
        # 3. Boundary plants include fresh labels and exact corners, no overlap filtering by outcomes.
        boundary=[]
        definitions={'zero_delay':('tau',0.,0.),'tiny_delay':('tau',1e-6,1e-4),
            'low_damping':('zeta',.1,.15),'high_damping':('zeta',1.95,2.),
            'low_gain':('K',.5,.55),'high_gain':('K',1.95,2.),
            'slow_plant':('wn',.5,.75),'fast_plant':('wn',9.75,10.),
            'large_normalized_delay':('tau',.9,1.)}
        for group,(key,low,high) in definitions.items():
            for replicate in range(25):
                pars=dict(zeta=rng.uniform(.1,2),wn=rng.uniform(.5,10),tau=rng.uniform(0,1),K=rng.uniform(.5,2))
                pars[key]=rng.uniform(low,high) if high>low else low
                if group=='large_normalized_delay': pars['wn']=rng.uniform(9,10)
                boundary.append(dict(case_id=f'{group}_{replicate}',group=group,**pars))
        import itertools
        for i,(z,w,t,k) in enumerate(itertools.product([.1,2.],[.5,10.],[0.,1.],[.5,2.])):
            boundary.append(dict(case_id=f'corner_{i}',group='corners',zeta=z,wn=w,tau=t,K=k))
        boundary=pd.DataFrame(boundary)
        for i,row in boundary.iterrows():
            t,y=PlantSimulator().observe_open_loop(Plant(*row[PARAMS]))
            for key,value in FeatureExtractor().extract(t,y).items(): boundary.loc[i,key]=value
        boundary.to_csv(OUT/'boundary_plants.csv',index=False)
        teachers=map_saved(pool,boundary_teacher,[(r.case_id,[getattr(r,p) for p in PARAMS]) for r in boundary.itertuples()], 'boundary_teachers.csv')
        boundary=boundary.merge(teachers,on='case_id')
        tasks=[]
        for m in ALL:
            prediction=boundary[TARGETS].to_numpy() if m=='numerical_reference' else models[m].predict(boundary[FEATURES])
            for (_,r),g in zip(boundary.iterrows(),prediction):
                tasks.append((dict(test='boundary',model=m,case_id=r.case_id,group=r.group),r[PARAMS].tolist(),g.tolist()))
        map_saved(pool,assess_task,tasks,'boundary_results.csv')
        # 4. Independent true-delay response, including the ideal derivative impulse.
        tasks=[]
        for m in ALL:
            for position,(source,row) in enumerate(cohort.iterrows()):
                tasks.append((dict(test='exact_delay',model=m,source_row=source,case_id=str(source)),
                              row[PARAMS].tolist(),nominal[m][position].tolist()))
        map_saved(pool,exact_task,tasks,'exact_delay_results.csv')
        # 5. Incremental disturbance response after nominal steady state.
        tasks=[({**info,'test':'disturbance'},p,g) for info,p,g in tasks]
        map_saved(pool,disturbance_task,tasks,'disturbance_results.csv')
    assert all(digest(ROOT/path)==value for path,value in hashes.items())
    assert digest(ROOT/'data/pid_dataset_final_10000.csv')==plan['dataset_sha256']
    assert runtime_signature(models)==plan['runtime_signature'], 'Implementation or fitted model state changed during the run.'
    (OUT/'verification.json').write_text(json.dumps(dict(frozen_artifacts_unchanged=True,
        dataset_unchanged=True,training_performed=True,model_source="reconstructed selected models from fixed training/validation split",all_five_tests_completed=True,
        base_plants=len(cohort),zero_delay_plants=int(cohort.tau.eq(0).sum())),indent=2))
    print('All stress experiments complete; frozen artifacts unchanged.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=12)
    parser.add_argument('--plants',type=int,default=100)
    parser.add_argument('--output',type=Path,default=ROOT/'temp/stress_tests')
    main(parser.parse_args())
