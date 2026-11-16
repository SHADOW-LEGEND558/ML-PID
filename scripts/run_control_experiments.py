"""Three-fold training-only control selection, target scaling and learning curves."""
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import FEATURES,TARGETS,PLANT_COLUMNS,load_frame,gain_metrics
from Regression import model_specs as regression_specs
from XGBoost import model_specs as tree_specs
from control_experiments import EmpiricalTargetRegressor,evaluate_task

OUTPUT=Path('results/control_experiments')


def summarize(records):
    frame=pd.DataFrame(records)
    passed=frame.accepted
    ratios=frame.loc[passed,'objective_ratio'].dropna()
    error=frame[[f'predicted_{g}' for g in TARGETS]].to_numpy()-frame[[f'target_{g}' for g in TARGETS]].to_numpy()
    return {'plants':len(frame),'passes':int(passed.sum()),'success_pct':float(100*passed.mean()),
            'stable_pct':float(100*frame.stable.mean()),'median_J_ratio':float(ratios.median()) if len(ratios) else np.nan,
            'gain_mse':float(np.mean(error**2)),
            'r2':gain_metrics(frame[[f'target_{g}' for g in TARGETS]].to_numpy(),
                              frame[[f'predicted_{g}' for g in TARGETS]].to_numpy())['r2']}


def ranking(result):
    return (-result['passes'],result['median_J_ratio'] if np.isfinite(result['median_J_ratio']) else np.inf,result['gain_mse'])


def evaluate(model,fit_rows,validation_rows,fold,pool,frame,references,weight):
    model.fit(frame.loc[fit_rows,FEATURES],frame.loc[fit_rows,TARGETS].to_numpy())
    predictions=model.predict(frame.loc[validation_rows,FEATURES])
    tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),prediction.tolist(),weight)
           for i,prediction in zip(validation_rows,predictions)]
    records=[]
    for result,prediction in zip(pool.map(evaluate_task,tasks),predictions):
        i=result['source_row'];ref=references[i]
        result.update(fold=fold,reference_reason=ref['reason'],reference_objective=ref['objective'])
        result['objective_ratio']=(result['objective']/ref['objective']
            if result['accepted'] and ref['accepted'] and ref['objective']>0 else None)
        result.update({key:float(frame.loc[i,key]) for key in PLANT_COLUMNS})
        result.update({f'predicted_{g}':float(prediction[j]) for j,g in enumerate(TARGETS)})
        result.update({f'target_{g}':float(frame.loc[i,g]) for g in TARGETS})
        records.append(result)
    return records


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path=Path('data/pid_dataset.csv');frame=load_frame(path)
    split=json.loads(Path('results/regularized_1000/regression/split.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==split['dataset_sha256']
    training=np.array(split['train_rows']);test=set(split['test_rows']);weight=split['gain_penalty']
    assert not set(training)&test
    groups=frame.loc[training].groupby(PLANT_COLUMNS,sort=True).ngroup().to_numpy()
    folds=[(training[a],training[b]) for a,b in GroupKFold(3).split(training,groups=groups)]
    manifest={'dataset_sha256':split['dataset_sha256'],'gain_penalty':weight,
        'development_rows':training.tolist(),'excluded_test_rows':sorted(test),
        'folds':[{'fit':a.tolist(),'validation':b.tolist()} for a,b in folds],
        'selection':'highest out-of-fold pass count, then lowest median accepted J ratio, then gain MSE',
        'normalization':'q=[y_ss*Kp,y_ss*t_rise*Ki,y_ss*Kd/t_rise]',
        'learning_curve_sizes':[128,256,'full fold (~533)']}
    (OUTPUT/'protocol.json').write_text(json.dumps(manifest,indent=2))
    specs={**regression_specs(42),**tree_specs(42)};candidates=[];configuration={}
    with ProcessPoolExecutor(max_workers=12) as pool:
        reference_tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),frame.loc[i,TARGETS].tolist(),weight) for i in training]
        references={r['source_row']:r for r in pool.map(evaluate_task,reference_tasks)}
        assert all(r['accepted'] for r in references.values())
        print('800 training-label controllers revalidated; test set excluded.',flush=True)
        for family,(estimator,grid,polynomial,scale) in specs.items():
            steps=[]
            if polynomial:steps.append(('polynomial',PolynomialFeatures(2,include_bias=False)))
            if scale:steps.append(('scaler',StandardScaler()))
            steps.append(('model',estimator))
            for normalized in [False,True]:
                for index,parameters in enumerate(ParameterGrid(grid or {})):
                    candidate=f'{family}_{"normalized" if normalized else "raw"}_{index}'
                    pipeline=Pipeline(steps).set_params(**parameters)
                    # Limit forest fit threads; simulations use their own process pool.
                    if family=='random_forest':pipeline.set_params(model__n_jobs=4)
                    model=EmpiricalTargetRegressor(pipeline,normalized)
                    records=[]
                    for fold,(fit_rows,val_rows) in enumerate(folds):
                        records+=evaluate(model,fit_rows,val_rows,fold,pool,frame,references,weight)
                    pd.DataFrame(records).to_csv(OUTPUT/f'{candidate}_oof.csv',index=False)
                    summary={'candidate':candidate,'family':family,'normalized':normalized,
                             'parameters':json.dumps(parameters),**summarize(records)}
                    candidates.append(summary);configuration[candidate]=model
                    pd.DataFrame(candidates).to_csv(OUTPUT/'candidates.csv',index=False)
                    print(f'{candidate}: success={summary["success_pct"]:.2f}%, median J ratio={summary["median_J_ratio"]:.3f}',flush=True)
        selected=[];comparisons=[];curves=[]
        for family in specs:
            eligible=[r for r in candidates if r['family']==family]
            winner=min(eligible,key=ranking);mse_winner=min(eligible,key=lambda r:r['gain_mse'])
            selected.append(winner)
            for normalized in [False,True]:
                subset=[r for r in eligible if r['normalized']==normalized]
                for criterion in ['control','mse']:
                    best=min(subset,key=ranking if criterion=='control' else lambda r:r['gain_mse'])
                    comparisons.append({'criterion':criterion,**best})
            print(f'Selected {winner["candidate"]}; MSE selection would choose {mse_winner["candidate"]}',flush=True)
            curves.append({'family':family,'training_size':min(len(a) for a,b in folds),**winner})
            for size in [128,256]:
                records=[]
                for fold,(fit_rows,val_rows) in enumerate(folds):
                    nested=np.random.default_rng(100+fold).permutation(fit_rows)[:size]
                    records+=evaluate(configuration[winner['candidate']],nested,val_rows,fold,pool,frame,references,weight)
                pd.DataFrame(records).to_csv(OUTPUT/f'{family}_learning_{size}_oof.csv',index=False)
                curves.append({'family':family,'training_size':size,**summarize(records)})
                pd.DataFrame(curves).to_csv(OUTPUT/'learning_curves.csv',index=False)
                print(f'{family}: learning curve {size} complete',flush=True)
            # Refit on all 800 development plants, without evaluating any test rows.
            model=configuration[winner['candidate']]
            model.fit(frame.loc[training,FEATURES],frame.loc[training,TARGETS].to_numpy())
            joblib.dump({'pipeline':model,'features':FEATURES,'targets':TARGETS,
                         'selected_candidate':winner,'protocol':manifest},OUTPUT/f'{family}.joblib')
        pd.DataFrame(selected).to_csv(OUTPUT/'selected.csv',index=False)
        pd.DataFrame(comparisons).to_csv(OUTPUT/'selection_comparison.csv',index=False)
        pd.DataFrame(curves).to_csv(OUTPUT/'learning_curves.csv',index=False)
    print('Completed training-only experiments; no test predictions generated.',flush=True)
