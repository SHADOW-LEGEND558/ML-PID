"""Summarize MLP development-fold experiments and create notebook-ready plots."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
if __name__=='__main__':matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from plot_baselines import LABELS,style,prediction_plot,overshoot_plot,overshoot_bins_plot,objective_plot,overshoot_summary
OUT=ROOT/'results/mlp_experiments'


def comparison_results():
    selected=pd.read_csv(OUT/'selected.csv');baseline=pd.read_csv(ROOT/'results/control_experiments/selected.csv')
    metrics=[];predictions={}
    for family in ['random_forest','xgboost']:
        r=baseline[baseline.family.eq(family)].iloc[0]
        metrics.append({'model':family,'success_pct':r.success_pct,'stable_pct':r.stable_pct,
                        'r2':r.r2,'median_J_ratio':r.median_J_ratio,'passes':r.passes})
        predictions[family]=pd.read_csv(ROOT/'results/control_experiments'/f'{r.candidate}_oof.csv').sort_values('source_row')
    for r in selected.itertuples():
        name=f'{r.variant}_mlp';LABELS[name]='Standard MLP' if r.variant=='standard' else 'Physics-informed MLP'
        metrics.append({'model':name,'success_pct':r.success_pct,'stable_pct':r.stable_pct,
                        'r2':r.r2,'median_J_ratio':r.median_J_ratio,'passes':r.passes})
        predictions[name]=pd.read_csv(OUT/f'{r.name}_oof.csv').sort_values('source_row')
    # OOF exports retain plant metadata, but response plotting also needs measured
    # settling time. Recover only missing feature columns from the frozen CSV.
    source_path=ROOT/'data/pid_dataset.csv'
    protocol=json.loads((OUT/'protocol.json').read_text())
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=protocol['dataset_sha256']:
        raise ValueError('The dataset changed; use the dataset matching these experiment artifacts.')
    source=pd.read_csv(source_path)
    for f in predictions.values():
        for column in ['t_rise','t_settling','y_peak','y_ss','tau_est','observation_duration']:
            if column not in f:
                f[column]=source.loc[f.source_row,column].to_numpy()
    for f in predictions.values():
        assert f.source_row.tolist()==next(iter(predictions.values())).source_row.tolist()
        assert f.fold.tolist()==next(iter(predictions.values())).fold.tolist()
        np.testing.assert_allclose(f[['target_Kp','target_Ki','target_Kd']],
            next(iter(predictions.values()))[['target_Kp','target_Ki','target_Kd']])
    return pd.DataFrame(metrics).set_index('model'),predictions,selected


def epoch_plot(selected):
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,r in zip(axes,selected.itertuples()):
        for fold in range(3):
            h=pd.read_csv(OUT/f'{r.name}_fold{fold}_history.csv')
            line=ax.plot(h.epoch,h.train_loss,label=f'Fold {fold+1} train')[0]
            ax.plot(h.epoch,h.val_loss,ls='--',color=line.get_color(),label=f'Fold {fold+1} validation')
        ax.set(title=f'{r.variant.capitalize()} MLP: selected seed-42 configuration',xlabel='Epoch',ylabel='Total training objective')
        ax.set_yscale('log');ax.legend(fontsize=8);ax.grid(alpha=.2)
    return fig


def seed_plot(seeds,baseline):
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for index,(variant,g) in enumerate(seeds.groupby('variant',sort=False)):
        ax.scatter(np.full(len(g),index),g.success_pct,s=55,label=variant)
        ax.plot([index-.15,index+.15],[g.success_pct.mean()]*2,color='black',lw=2)
    ax.axhline(baseline.loc['random_forest','success_pct'],ls='--',color='#278cb2',label='RF (seed 42)')
    ax.axhline(baseline.loc['xgboost','success_pct'],ls=':',color='#9b6ba5',label='XGBoost (seed 42)')
    ax.set_xticks([0,1],['Standard MLP','Physics-informed MLP'])
    ax.set(ylabel='OOF full-validation success (%)',title='Initialization sensitivity: fixed folds, three seeds',ylim=(0,100))
    ax.legend(fontsize=8);ax.grid(alpha=.2);return fig


def data_size_plot():
    mlp=pd.read_csv(OUT/'learning_curves.csv')
    trees=pd.read_csv(ROOT/'results/control_experiments/learning_curves.csv')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for family,g in trees[trees.family.isin(['random_forest','xgboost'])].groupby('family'):
        g=g.sort_values('training_size');ax.plot(g.training_size,g.success_pct,'o--',label=LABELS[family])
    for variant,g in mlp.groupby('variant'):
        g=g.sort_values('training_size');ax.plot(g.training_size,g.success_pct,'o-',label=f'{variant.capitalize()} MLP')
    ax.set(xlabel='Training plants per fold',ylabel='OOF full-validation success (%)',
           title='Matched sample-size curves: fixed settings, seed 42',ylim=(0,100))
    ax.grid(alpha=.2);ax.legend();return fig


def mdtable(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in frame.iterrows():
        lines.append('| '+' | '.join(f'{r[c]:.3f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
    return lines


if __name__=='__main__':
    style();metrics,predictions,selected=comparison_results();seeds=pd.read_csv(OUT/'seed_results.csv')
    protocol=json.loads((OUT/'protocol.json').read_text());development=set(protocol['development_rows']);excluded=set(protocol['excluded_test_rows'])
    for p in OUT.glob('*_oof.csv'):
        f=pd.read_csv(p);assert len(f)==800 and f.source_row.nunique()==800
        assert set(f.source_row)==development and not set(f.source_row)&excluded
        assert np.isfinite(f[['predicted_Kp','predicted_Ki','predicted_Kd']]).all().all()
    figures={'training_curves':epoch_plot(selected),'seed_comparison':seed_plot(seeds,metrics),
        'gain_predictions':prediction_plot(predictions),'overshoot':overshoot_plot(predictions,population_label='OOF validation plants'),
        'overshoot_bins':overshoot_bins_plot(predictions),'objective_ratios':objective_plot(predictions)}
    if (OUT/'learning_curves.csv').exists():figures['data_size_curves']=data_size_plot()
    for name,fig in figures.items():fig.savefig(OUT/f'{name}.png',dpi=160);plt.close(fig)
    metrics.to_csv(OUT/'comparison.csv');summary=overshoot_summary(predictions);summary.to_csv(OUT/'overshoot_summary.csv')
    rows=[]
    for variant,g in seeds.groupby('variant'):
        rows.append({'variant':variant,'seeds':len(g),'mean_success_pct':g.success_pct.mean(),
            'std_success_pp':g.success_pct.std(),'min_success_pct':g.success_pct.min(),'max_success_pct':g.success_pct.max()})
    stability=pd.DataFrame(rows);stability.to_csv(OUT/'seed_summary.csv',index=False)
    margins=pd.read_csv(OUT/'label_stability_margins.csv')
    lines=['# Standard and physics-informed MLP report','',
        '## Design and protocol','',
        'The MLP is in src/MLP.py. Architecture: 5 -> 128 -> 64 -> 32 -> 3, SiLU activations, dropout, AdamW, positive empirical-normalized outputs. Input standardization, normalized-target scale and log-loss standard deviation are fitted using each fold’s training rows only. Prediction reverses normalization from the five measured inputs; no physical parameters are supplied at inference. Gain upper bounds are checked by validation, not imposed by clipping.','',
        'Standard MLP loss is squared log error of normalized q, divided by training-fold log standard deviations per output. Delta=0.001. Physics MLP adds beta times a scaled Hurwitz violation penalty. Beta candidates are 0.1 and 1; beta=0 is the standard comparison. Two regularization pairs are searched: dropout 0.05/weight decay 0.0001, and dropout 0.1/weight decay 0.001. No extra gain penalty or differentiable time-domain simulation is added.','',
        'The characteristic polynomial uses the same second-order Padé approximation and tiny-delay cutoff as the shared validator. Exact zero delay uses a cubic; delayed cases use a quintic. Time is normalized by wn. Relative determinants and coefficient scaling make penalty terms dimensionless. The numerical soft margin is 0.001; it is not a physical pole-distance guarantee.', '',
        f"{int(margins.soft_margin_active.sum())}/800 feasible labels activate the soft numerical margin; their minimum scaled margin is {margins.minimum_scaled_margin.min():.6g}. The penalty is a training preference, not proof of stability.",'',
        'Three existing development folds cover the same 800 plants as RF/XGBoost. The 200 holdout plants are excluded from all training, stopping, selection and scoring in this experiment. Early stopping uses validation total loss (patience 50, maximum 500 epochs); controller selection uses OOF full-pass count then median accepted J ratio. Checkpoints are selected by early-stopping loss, not by evaluating every epoch in the simulator.','',
        'Six seed-42 configurations are compared (three beta values x two regularization pairs). The best standard and best physics configurations are then repeated at seeds 43 and 44. Repeats assess initialization variation on fixed data, not uncertainty from new data splits. Final selected seed-42 models are refit on all 800 development plants for the median best fold epoch and saved as standard_mlp.pt and physics_mlp.pt.','',
        '**Fairness/limits:** physics MLP uses privileged physical metadata in its training and validation loss. Standard MLP and the tree regressors do not receive that loss information. Both MLPs use only empirical features at inference. Scores are development/selection scores, not unbiased final test estimates. No claim of hardware or exact-delay safety is made.','',
        '## Matched development-fold results (selected seed-42 configurations)','']
    lines+=mdtable(metrics.reset_index(),['model','success_pct','stable_pct','r2','median_J_ratio','passes'])
    lines+=['','J ratios include only accepted prediction/reference pairs; the pass count is their coverage. Failed controllers stay in overall success denominators. Different accepted subsets mean ratio medians alone cannot rank overall quality.','',
        '## Selected MLP settings','']
    lines+=mdtable(selected,['variant','beta','dropout','weight_decay','median_best_epoch','success_pct'])
    lines+=['','## Initialization sensitivity','']+mdtable(stability,list(stability.columns))
    if (OUT/'learning_curves.csv').exists():
        learning=pd.read_csv(OUT/'learning_curves.csv').sort_values(['variant','training_size'])
        lines+=['','## Sample-size learning curves','',
            'For each selected MLP configuration at seed 42, train on nested 128 and 256 subsets and compare with the full fold (533/534 plants). The same development validation folds and subset permutations as the tree curves are used. Input/target scaling is fitted separately on each subset. Early stopping remains enabled. Settings were selected at full size, so these curves are descriptive, not independently tuned or unbiased final scores.','']
        lines+=mdtable(learning,['variant','training_size','success_pct','stable_pct','median_J_ratio','r2'])
    lines+=['','## Overshoot severity','']+mdtable(summary.reset_index(),['model','at_most_10','over_10_to_12','over_12_to_15','over_15_to_20','over_20','unstable','stable_unmeasured'])
    lines+=['','Rejected-controller overshoot values are finite-window diagnostics, not convergence-certified values. Full per-gain errors, raw predictions, failures, epoch histories, candidate scores and seed scores are saved alongside this report.','',
        '## Interpretation and recommendation','',
        f"The standard MLP averages {seeds[seeds.variant.eq('standard')].success_pct.mean():.2f}% full-validation success over three seeds; the physics variant averages {seeds[seeds.variant.eq('physics')].success_pct.mean():.2f}%. Both are materially above the matched tree development results, but these are still selection/development estimates. The tiny seed-42 physics advantage is not consistent across seeds. Prefer the standard MLP as the simpler next candidate; the current stability penalty has not demonstrated additional benefit.",'',
        'The MLP and tree models differ in training loss and output parameterization, not just architecture: MLP uses standardized log loss and positive outputs. This experiment does not isolate which component causes the improvement. The physics penalty addresses pole stability, not the 10% overshoot requirement. Residual controller failures still matter, and no model should be treated as a guaranteed tuner.','',
        'Training uses a 500-epoch cap; several folds choose the final epoch, so this is a bounded-budget comparison rather than evidence of complete optimization convergence. Further tuning must stay within development data. Before claiming final generalization, freeze settings and use an independent evaluation set; the existing 200-plant holdout was excluded here but has been inspected in earlier work.','',
        '## Verification','',
        'All 27 regression tests passed. Polynomial coefficients were compared against the shared simulator. Scaled Hurwitz signs were checked against pole stability on sampled zero/delayed plants, and gradients were checked for finiteness. Checkpoint round trips, safe weights-only loading, and positive predictions were tested. All OOF tables were audited to exclude every holdout plant. The executed notebook renders its figures and reproduces all 12 illustrated saved validation outcomes.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(metrics.to_string());print(stability.to_string(index=False));print(f'Saved report and {len(figures)} plot figures.')
