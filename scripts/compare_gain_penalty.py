"""Compare regularization weights on fixed plants, including zero delay."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import argparse
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_gen import Plant, PlantSimulator, PIDOptimizer


def trial(task):
    index, plant, weight = task
    sim = PlantSimulator()
    opt = PIDOptimizer(sim, gain_penalty=weight)
    gains = opt.optimize(plant, sim.get_dynamic_t_eval(plant))
    row = {'plant_id': index, **plant.__dict__, 'weight': weight, 'accepted': gains is not None}
    row['failure_reasons'] = json.dumps([r['validation'] for r in opt.diagnostics]) if gains is None else ''
    if gains is not None:
        vector = np.array([gains.Kp, gains.Ki, gains.Kd])
        row.update(gains.__dict__)
        row.update(opt.last_validation)
        row['upper_bound_hit'] = bool(np.any(np.isclose(vector, opt.BOUNDS[:,1], atol=1e-5, rtol=0)))
        q = plant.K * np.array([gains.Kp, gains.Ki/plant.wn, plant.wn*gains.Kd])
        row['normalized_gain_norm'] = float(np.linalg.norm(q))
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=None, help='Select matched plants from source; half zero-delay if available')
    parser.add_argument('--source', type=Path, default=Path('data/pid_dataset.csv'))
    parser.add_argument('--weights', type=float, nargs='+', default=[0.,.001,.01,.1])
    parser.add_argument('--output', type=Path, default=Path('results/gain_penalty'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    plants = [Plant(z,w,t,k) for z,w,k in [(.1,.5,.5),(.7,2,1),(2,.5,2),(.1,10,2),
                                           (1,5,.5),(2,10,1),(.3,5,1.5),(1.5,1,.7)]
              for t in [0., .3]]
    source_hash = None
    if args.samples is not None:
        source = pd.read_csv(args.source)
        if not 2 <= args.samples <= len(source):
            parser.error('--samples must be between 2 and the source row count')
        zeros = source[source.tau.eq(0)]
        positives = source[source.tau.gt(0)]
        zero_count = min(len(zeros), args.samples//2)
        # Enrich with the smallest positive delays; those are the other difficult boundary.
        positive_count = args.samples-zero_count
        near = positives.nsmallest(min(20, positive_count), 'tau')
        remaining = positives.drop(near.index).sample(positive_count-len(near),random_state=args.seed)
        selected = pd.concat([zeros.sample(zero_count,random_state=args.seed),near,remaining])
        plants = [Plant(r.zeta,r.wn,r.tau,r.K) for r in selected.itertuples()]
        source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    tasks = [(i,p,a) for i,p in enumerate(plants) for a in args.weights]
    output = args.output;output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([p.__dict__ for p in plants]).to_csv(output/'plants.csv',index=False)
    (output/'experiment.json').write_text(json.dumps({'seed':args.seed,'weights':args.weights,
        'plants':len(plants),'source_sha256':source_hash,'starts':4,
        'purpose':'weight calibration; not an independent final test set'},indent=2))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows=[]
        futures = [pool.submit(trial,task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if len(rows)%20 == 0:
                pd.DataFrame(rows).sort_values(['plant_id','weight']).to_csv(output/'trials_partial.csv',index=False)
            print(f"{len(rows)}/{len(tasks)} weight={row['weight']} accepted={row['accepted']}",flush=True)
    df=pd.DataFrame(rows).sort_values(['plant_id','weight']);df.to_csv(output/'trials.csv',index=False)
    summary=df.assign(zero_delay=df.tau.eq(0)).groupby(['weight','zero_delay']).agg(
        attempted=('accepted','size'),accepted=('accepted','sum'),
        upper_bound_hits=('upper_bound_hit','sum'),median_itae=('itae','median'),
        median_Kp=('Kp','median'),median_Ki=('Ki','median'),median_Kd=('Kd','median'))
    summary.to_csv(output/'summary.csv');print(summary.to_string())
