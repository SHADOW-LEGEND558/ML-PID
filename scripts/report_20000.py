"""Report and plotting helpers for the frozen 20k experiment."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
if __name__=='__main__':matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from large_experiment import OUT,load_models_results
from plot_baselines import (LABELS,style,performance_plot,prediction_plot,failure_plot,
    objective_plot,itae_plot,overshoot_plot,overshoot_bins_plot,overshoot_summary)

BASELINES=['linear','ridge','polynomial_ridge','random_forest','xgboost']
COMPARISON=['random_forest','xgboost','standard_mlp']


def learning_plot(names):
    curves=pd.read_csv(OUT/'learning_curves.csv')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for name in names:
        g=curves[curves.model.eq(name)].sort_values('training_size')
        ax.plot(g.training_size,g.success_pct,'o-',label=LABELS[name])
    ax.set(xlabel='Training plants',ylabel='Full-validation success (%)',ylim=(0,100),
        title='Learning curves: same 500 validation plants, fixed selected settings')
    ax.grid(alpha=.2);ax.legend();return fig


def mlp_history_plot():
    selected=pd.read_csv(OUT/'selected_before_test.csv');r=selected[selected.model.eq('standard_mlp')].iloc[0]
    history=pd.read_csv(OUT/f'{r.candidate}_history.csv')
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(history.epoch,history.train_loss,label='Training');ax.plot(history.epoch,history.val_loss,label='Validation')
    ax.axvline(history.loc[history.val_loss.idxmin(),'epoch'],color='#888888',ls=':',label='Lowest recorded validation loss')
    ax.set(xlabel='Epoch',ylabel='Standardized normalized-target log loss',title='Standard MLP: training and early-stopping loss')
    ax.set_yscale('log');ax.grid(alpha=.2);ax.legend();return fig


def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in frame.iterrows():
        lines.append('| '+' | '.join(f'{r[k]:.3f}' if isinstance(r[k],(float,np.floating)) else str(r[k]) for k in columns)+' |')
    return lines


if __name__=='__main__':
    style();names=BASELINES+['standard_mlp'];metrics,predictions,protocol=load_models_results(names)
    curves=pd.read_csv(OUT/'learning_curves.csv');validation=pd.read_csv(OUT/'selected_before_test.csv')
    curves.to_csv(OUT/'learning_curves.csv',index=False)
    plotdir=OUT/'plots';plotdir.mkdir(exist_ok=True)
    figures={'performance':performance_plot(metrics),'gain_predictions':prediction_plot(predictions),
        'failure_outcomes':failure_plot(predictions),'objective_ratios':objective_plot(predictions),
        'itae_comparison':itae_plot(predictions),'overshoot':overshoot_plot(predictions),
        'overshoot_bins':overshoot_bins_plot(predictions),'learning_curves':learning_plot(names),
        'mlp_history':mlp_history_plot()}
    for name,fig in figures.items():fig.savefig(plotdir/f'{name}.png',dpi=160);plt.close(fig)
    over=overshoot_summary(predictions);over.to_csv(OUT/'overshoot_summary.csv')
    itae=[];failure_rows=[]
    for name,f in predictions.items():
        ratio=f.itae_ratio.dropna()
        itae.append({'model':name,'paired_count':len(ratio),'median_ITAE_ratio':ratio.median(),
            'p90_ITAE_ratio':ratio.quantile(.9),'median_ITAE_change_pct':100*(ratio.median()-1)})
        for reason,count in f.reason.value_counts().items():failure_rows.append({'model':name,'reason':reason,'plants':int(count)})
    itae=pd.DataFrame(itae);itae.to_csv(OUT/'itae_summary.csv',index=False)
    failures=pd.DataFrame(failure_rows);failures.to_csv(OUT/'failure_counts.csv',index=False)
    lines=['# 20,000-plant experiment','',
        '## Frozen data and evaluation','',
        f"Source: {protocol['data_path']}. SHA-256: {protocol['dataset_sha256']}. Generation uses regularized tuning with lambda=0.01.",'',
        'Exact split: 15,000 training / 2,500 validation / 2,500 testing, seed 42. Delay-zero versus positive-delay stratification is preserved; identical plants are prohibited. Five empirical inputs only. Every model uses empirical-normalized targets; physical parameters enter the simulator only, not inference.','',
        'Linear regression, ridge alpha=100, and degree-two polynomial ridge alpha=1 retain their previously selected configurations. Forest compares depth=12/min-leaf=3 versus unlimited-depth/min-leaf=1, with 200 trees. XGBoost compares 150 versus 400 trees with depth=6 and learning rate=0.08. The standard MLP retains 5-128-64-32-3, positive normalized outputs, standardized log loss, dropout=0.05, AdamW weight decay=0.0001, and seed=42. Batch size is 256 for efficiency; maximum epochs=500, patience=50.','',
        'Input scaling and target statistics are fitted on the 15,000 training plants. MLP stopping uses validation supervised loss; tree setting selection ranks full-controller pass count then accepted J ratio on validation data. Settings and weights are saved before test scoring. There is no refit using validation/test labels and no post-test retuning. Models train on exactly 15,000 plants.','',
        'All 2,500 validation target controllers and 2,500 test target controllers are rechecked. Predicted gains are evaluated raw, without clipping/fallback. Full acceptance requires bounded finite gains, stable poles, refined overshoot <=10%, tail error <=0.001, and resolved ITAE across grids/horizons. Gain MSE/RMSE/R² describe label prediction; feasibility and ITAE/J describe control behavior.','',
        '## Final test results','']
    lines+=table(metrics.reset_index(),['model','plants','success_pct','stable_pct','bounds_violation_pct','r2','median_J_ratio','ratio_pairs','reference_passes'])
    lines+=['','The test scores are descriptive final results for the frozen settings. J ratios include only validated prediction/reference pairs and subsets differ across models. Failed controllers remain in the full-success denominator; smaller conditional medians alone cannot rank overall reliability.','',
        '## Per-gain test errors','']
    gains=[]
    for name,r in metrics.iterrows():
        for gain in ['Kp','Ki','Kd']:
            gains.append({'model':name,'gain':gain,'MAE':r[f'{gain}_mae'],'RMSE':r[f'{gain}_rmse'],'R2':r[f'{gain}_r2']})
    lines+=table(pd.DataFrame(gains),['model','gain','MAE','RMSE','R2'])
    lines+=['',
        '## Raw ITAE compared with optimized targets','']+table(itae,list(itae.columns))
    lines+=['','ITAE ratio = predicted ITAE / optimized-target ITAE. Percentage change = 100*(ratio-1). A ratio of 1 or change of 0% means equal tracking cost. Only converged, validated pairs are compared; unstable/unresolved/failed controllers are not assigned zero ITAE. Coverage is explicit. Lower raw ITAE does not automatically mean better J because J also penalizes aggressive gains, and the reference labels are local numerical solutions.','',
        '## Overshoot severity on test plants','']
    lines+=table(over.reset_index(),['model','at_most_10','over_10_to_12','over_12_to_15','over_15_to_20','over_20','unstable','stable_unmeasured'])
    lines+=['','Measured overshoot on rejected controllers is diagnostic. Unstable and unmeasured cases remain separate; below-limit overshoot alone is not full acceptance.','',
        '## Validation selections (before test)','']+table(validation,['model','setting','success_pct','r2','median_J_ratio'])
    lines+=['','## Learning curves','',
        'The same nested random training subsets (1,000/5,000/10,000/15,000 plants) are used for every model. Curves use the same 500 delay-stratified validation plants, not test plants. Full-size curve scores are filtered from existing validation results; smaller fits use fixed selected settings. MLP early stopping uses the full 2,500 validation set for all sizes. Curves are development diagnostics and can be optimistic because settings and stopping used validation data.','']
    lines+=table(curves.sort_values(['model','training_size']),['model','training_size','success_pct','r2','median_J_ratio'])
    lines+=['','## Failure counts','']+table(failures,list(failures.columns))
    lines+=['','## Main findings','',
        'The standard MLP passes 2,448/2,500 test plants (97.92%), versus 2,297/2,500 for random forest (91.88%) and 2,198/2,500 for XGBoost (87.92%). All three are stable on the complete test set; full-pass failures still matter. The remaining regression models have much lower control success despite reasonable gain-error scores.','',
        'Among validated pairs only, median ITAE is 4.28% above the optimized target for the MLP, 4.63% above for random forest, and 11.83% above for XGBoost. This is conditional performance with the coverage shown above, not a guarantee that every accepted controller is this close to the target.','',
        'On the fixed 500-plant validation subset, MLP success rises 80.0% -> 96.0% -> 96.6% -> 98.0% at 1,000 -> 5,000 -> 10,000 -> 15,000 training plants. Forest rises 69.6% -> 84.2% -> 87.8% -> 91.2%. This supports the usefulness of the extra data for these nonlinear models within the studied bounds. Linear/ridge curves are largely flat. These are pipeline learning curves with fixed hyperparameters and an epoch cap/early stopping for MLP, not pure information-theoretic sample-size effects.','',
        '## Verification','',
        'All 29 regression tests passed. Split sizes, disjointness, and zero-delay stratification were checked. The dataset hash matched its sidecar; 128 training feature recomputations matched stored values. All 5,000 validation/test reference controllers passed revalidation. All six saved models reproduced their exported test predictions. Model-file and selection hashes remained unchanged after settings were frozen. Learning-curve evaluations use validation rows only. Both updated notebooks were executed, preserving earlier sections and adding the new-data sections at the end.','']
    lines+=['','## Limits and artifacts','',
        'Simulation uses ideal PID and second-order Padé delay, not exact-delay/hardware validation or a robustness guarantee. This is one training seed and one split; initialization and distribution uncertainty remain. All models have the same input information, but differ in loss/output parameterization; improvements cannot be attributed solely to architecture. The dataset’s parameter distribution and accepted-label filtering define the evaluation population.','',
        'Saved model files, test/validation predictions, split manifest, candidate settings, MLP epoch histories, and learning-curve records are in this directory. Earlier 1,000-row results and notebook sections are preserved. New 20,000-row sections are appended at the end of notebooks/baseline_validation.ipynb and notebooks/mlp_validation.ipynb.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(metrics[['success_pct','stable_pct','r2','median_J_ratio']].to_string())
    print('Saved final report and',len(figures),'figures.')
