"""Train standard and physics-informed MLPs without scoring the test holdout."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import FEATURES,TARGETS,PLANT_COLUMNS,load_frame,gain_metrics
from control_experiments import evaluate_task

OUT=Path('results/mlp_experiments')


def score(records):
    f=pd.DataFrame(records);accepted=f[f.accepted]
    predicted=f[['predicted_'+g for g in TARGETS]].to_numpy();truth=f[['target_'+g for g in TARGETS]].to_numpy()
    return {'plants':len(f),'passes':int(f.accepted.sum()),'success_pct':100*float(f.accepted.mean()),
        'stable_pct':100*float(f.stable.mean()),'bounds_violation_pct':100*float((~f.within_bounds).mean()),
        'median_J_ratio':accepted.objective_ratio.median(),
        'p90_J_ratio':accepted.objective_ratio.quantile(.9),'ratio_pairs':len(accepted),
        'median_ITAE_ratio':accepted.itae_ratio.median(),**gain_metrics(truth,predicted)}


def rank(r):
    return (-r['passes'],r['median_J_ratio'] if np.isfinite(r['median_J_ratio']) else np.inf)


def cross_validate(options,name,folds,frame,pool,references,weight):
    from MLP import MLPModel
    records=[];epochs=[]
    for fold in range(3):
        fit=np.array(folds[fold]['fit']);val=np.array(folds[fold]['validation'])
        model=MLPModel(**options)
        model.fit(frame.loc[fit,FEATURES],frame.loc[fit,TARGETS],frame.loc[fit,PLANT_COLUMNS],
            validation_data=(frame.loc[val,FEATURES],frame.loc[val,TARGETS],frame.loc[val,PLANT_COLUMNS]))
        epochs.append(model.best_epoch)
        pd.DataFrame(model.history).to_csv(OUT/f'{name}_fold{fold}_history.csv',index=False)
        predictions=model.predict(frame.loc[val,FEATURES])
        tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),prediction.tolist(),weight) for i,prediction in zip(val,predictions)]
        for result,prediction in zip(pool.map(evaluate_task,tasks),predictions):
            i=result['source_row'];ref=references[i]
            result.update(fold=fold,reference_objective=ref['objective'],reference_itae=ref['itae'],
                objective_ratio=result['objective']/ref['objective'] if result['accepted'] else None,
                itae_ratio=result['itae']/ref['itae'] if result['accepted'] else None)
            result.update({key:float(frame.loc[i,key]) for key in PLANT_COLUMNS})
            result.update({f'predicted_{g}':float(prediction[j]) for j,g in enumerate(TARGETS)})
            result.update({f'target_{g}':float(frame.loc[i,g]) for g in TARGETS})
            records.append(result)
        print(f'{name}: fold {fold+1}/3, best epoch {model.best_epoch}',flush=True)
    pd.DataFrame(records).to_csv(OUT/f'{name}_oof.csv',index=False)
    result={'name':name,**options,'median_best_epoch':int(np.median(epochs)),**score(records)}
    print(f'{name}: pass={result["success_pct"]:.2f}%, stable={result["stable_pct"]:.2f}%, R2={result["r2"]:.3f}',flush=True)
    return result


if __name__=='__main__':
    from MLP import MLPModel,stability_margins
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    frame=load_frame('data/pid_dataset.csv')
    protocol=json.loads(Path('results/control_experiments/protocol.json').read_text())
    assert hashlib.sha256(Path('data/pid_dataset.csv').read_bytes()).hexdigest()==protocol['dataset_sha256']
    training=protocol['development_rows'];excluded=set(protocol['excluded_test_rows'])
    for fold in protocol['folds']:
        assert not set(fold['fit'])&set(fold['validation'])
        assert not (set(fold['fit'])|set(fold['validation']))&excluded
    configuration={**protocol,'torch_version':str(torch.__version__),
        'device':'cpu','architecture':[5,128,64,32,3],'physics_weights':[0.,.1,1.],
        'regularization':[{'dropout':.05,'weight_decay':.0001},{'dropout':.1,'weight_decay':.001}],
        'selection_seed':42,'repeat_seeds':[43,44],
        'early_stopping':'validation total loss; patience 50; maximum 500 epochs',
        'loss':'standardized log error of empirical normalized targets + beta * scaled Hurwitz violations',
        'physics_model':'second-order Pade, separate exact-zero cubic, shared tiny-delay cutoff',
        'note':'physics variant uses privileged physical metadata in training loss; inference remains empirical only'}
    (OUT/'protocol.json').write_text(json.dumps(configuration,indent=2))
    # Check how the numerical soft margin treats the known feasible training labels.
    params=torch.tensor(frame.loc[training,PLANT_COLUMNS].to_numpy(),dtype=torch.float64)
    gains=torch.tensor(frame.loc[training,TARGETS].to_numpy(),dtype=torch.float64)
    label_margin_rows=[]
    for mask,margins in stability_margins(params,gains):
        for row_id,margin in zip(np.array(training)[mask.numpy()],margins.min(1).values.detach().numpy()):
            label_margin_rows.append({'source_row':row_id,'minimum_scaled_margin':margin,'soft_margin_active':bool(margin<.001)})
    pd.DataFrame(label_margin_rows).to_csv(OUT/'label_stability_margins.csv',index=False)
    print('CPU MLP training; CUDA available:',torch.cuda.is_available(),flush=True)
    candidates=[];repeats=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        tasks=[(int(i),frame.loc[i,PLANT_COLUMNS].tolist(),frame.loc[i,TARGETS].tolist(),protocol['gain_penalty']) for i in training]
        references={r['source_row']:r for r in pool.map(evaluate_task,tasks)}
        assert all(r['accepted'] for r in references.values())
        for beta in [0.,.1,1.]:
            for index,(dropout,decay) in enumerate([(.05,1e-4),(.1,1e-3)]):
                name=f'{"standard" if beta==0 else "physics"}_beta{beta:g}_reg{index}_seed42'
                result=cross_validate({'beta':beta,'dropout':dropout,'weight_decay':decay,'seed':42},name,
                    protocol['folds'],frame,pool,references,protocol['gain_penalty'])
                candidates.append(result);pd.DataFrame(candidates).to_csv(OUT/'candidates.csv',index=False)
        selected=[]
        for variant in ['standard','physics']:
            eligible=[r for r in candidates if (r['beta']==0)==(variant=='standard')]
            winner=min(eligible,key=rank);selected.append({'variant':variant,**winner});repeats.append({'variant':variant,**winner})
            for seed in [43,44]:
                options={k:winner[k] for k in ['beta','dropout','weight_decay']};options['seed']=seed
                name=f'{variant}_selected_seed{seed}'
                result=cross_validate(options,name,protocol['folds'],frame,pool,references,protocol['gain_penalty'])
                repeats.append({'variant':variant,**result});pd.DataFrame(repeats).to_csv(OUT/'seed_results.csv',index=False)
            # Fixed-epoch final fit uses the median selected stopping epoch, no test peeking.
            options={k:winner[k] for k in ['beta','dropout','weight_decay','seed']}
            options['max_epochs']=winner['median_best_epoch']
            model=MLPModel(**options).fit(frame.loc[training,FEATURES],frame.loc[training,TARGETS],frame.loc[training,PLANT_COLUMNS])
            model.save(OUT/f'{variant}_mlp.pt',metadata={'selected':winner,'protocol':configuration})
        pd.DataFrame(selected).to_csv(OUT/'selected.csv',index=False)
        pd.DataFrame(repeats).to_csv(OUT/'seed_results.csv',index=False)
    print('MLP experiments complete; the 200 test plants remain excluded.',flush=True)
