"""Summarize training-only control experiments and overshoot associations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/control_experiments')


def table(frame,columns):
    rows=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in frame.iterrows():
        rows.append('| '+' | '.join(f'{r[c]:.3f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns)+' |')
    return rows


if __name__=='__main__':
    selected=pd.read_csv(OUT/'selected.csv');comparison=pd.read_csv(OUT/'selection_comparison.csv')
    curves=pd.read_csv(OUT/'learning_curves.csv');protocol=json.loads((OUT/'protocol.json').read_text())
    excluded=set(protocol['excluded_test_rows']);development=set(protocol['development_rows'])
    clusters=[];failures=[]
    for chosen in selected.itertuples():
        f=pd.read_csv(OUT/f'{chosen.candidate}_oof.csv')
        assert len(f)==800 and f.source_row.nunique()==800
        assert set(f.source_row)==development and not set(f.source_row)&excluded
        f['measured']=f.stable & np.isfinite(f.overshoot_pct)
        f['overshoot_miss']=f.measured & (f.overshoot_pct>10+1e-6)
        f['small_miss']=f.overshoot_miss & (f.overshoot_pct<=12)
        f['severe_miss']=f.measured & (f.overshoot_pct>20)
        failures.append({'family':chosen.family,'passes':int(f.accepted.sum()),
            'overshoot_misses':int(f.overshoot_miss.sum()),'10_to_12':int(f.small_miss.sum()),
            'over_20':int(f.severe_miss.sum()),'unstable':int((~f.stable).sum()),
            'unmeasured_stable':int((f.stable&~f.measured).sum())})
        f['normalized_delay']=f.tau*f.wn
        for feature,bins in [('K',[.5,.875,1.25,1.625,2]),('wn',[.5,2.875,5.25,7.625,10]),
                             ('zeta',[.1,.575,1.05,1.525,2]),('normalized_delay',[0,.1,.5,1,2,5,10])]:
            categories=pd.cut(f[feature],bins,include_lowest=True)
            for label,g in f.groupby(categories,observed=True):
                clusters.append({'family':chosen.family,'feature':feature,'bin':str(label),'plants':len(g),
                    'pass_pct':100*g.accepted.mean(),'overshoot_miss_pct':100*g.overshoot_miss.mean(),
                    'severe_miss_pct':100*g.severe_miss.mean(),'unstable_pct':100*(~g.stable).mean(),
                    'median_measured_overshoot':g.loc[g.measured,'overshoot_pct'].median()})
        for zero,g in f.groupby(f.tau.eq(0)):
            clusters.append({'family':chosen.family,'feature':'delay_group','bin':'zero' if zero else 'positive',
                'plants':len(g),'pass_pct':100*g.accepted.mean(),'overshoot_miss_pct':100*g.overshoot_miss.mean(),
                'severe_miss_pct':100*g.severe_miss.mean(),'unstable_pct':100*(~g.stable).mean(),
                'median_measured_overshoot':g.loc[g.measured,'overshoot_pct'].median()})
    clusters=pd.DataFrame(clusters);clusters.to_csv(OUT/'failure_clusters.csv',index=False)
    failures=pd.DataFrame(failures);failures.to_csv(OUT/'overshoot_analysis.csv',index=False)
    comparison_rows=[]
    for family,g in comparison.groupby('family',sort=False):
        row={'family':family}
        for normalized,label in [(False,'raw'),(True,'normalized')]:
            for criterion in ['mse','control']:
                r=g[g.normalized.eq(normalized)&g.criterion.eq(criterion)].iloc[0]
                row[f'{label}_{criterion}_pass_pct']=r.success_pct
        comparison_rows.append(row)
    ablation=pd.DataFrame(comparison_rows);ablation.to_csv(OUT/'ablation_summary.csv',index=False)
    lines=['# Training-only baseline improvement experiments','',
        'Only the five existing models were used. No new dataset or model family was added. The 200 saved test plants were excluded from every fit, validation score, overshoot analysis, and learning curve. Protocol and row identities are in protocol.json.','',
        '## Method','',
        'The 800 development plants were evaluated with three grouped folds. Each candidate fits roughly 533 plants and predicts the other roughly 267; pooling the folds gives 800 out-of-fold predictions. Every candidate uses the same folds.','',
        '38 candidates combine the existing hyperparameter grids with raw or empirical-normalized targets. For normalization, q=[y_ss*Kp, y_ss*t_rise*Ki, y_ss*Kd/t_rise]; inverse scaling at prediction uses only the five input features. Polynomial expansion/scaling is fitted inside each fold. No gain clipping or fallback is applied.','',
        'Control selection ranks candidates by full-validation pass count, then median J ratio on accepted pairs, then raw gain MSE. MSE selection chooses the smallest raw gain MSE on these same predictions. All reference controllers were revalidated. The objective uses lambda=0.01 and the unchanged stability, gain-bound, overshoot and convergence criteria.','',
        '**These are development scores used to select settings, not unbiased final test estimates.** Selecting candidates and reporting their scores on the same folds introduces selection optimism. Paired J summaries exclude failed predictions, so success rate remains primary and coverage must be considered.','',
        '## 1–2. Control-based selection and normalized targets','']
    lines+=table(ablation,list(ablation.columns))
    lines+=['','The table holds the folds fixed. Comparing MSE versus control within a target representation isolates the selection criterion. Comparing raw versus normalized under the same criterion examines target scaling (with independently selected hyperparameters). Gains from one change need not add to gains from the other.','',
        '### Selected configuration per existing model','']
    lines+=table(selected,['family','normalized','success_pct','stable_pct','median_J_ratio','r2','parameters'])
    lines+=['','## 3. Overshoot failure analysis','']
    lines+=table(failures,list(failures.columns))
    lines+=['','Misses are measured stable responses above 10%; the near-miss band is >10–12%, and severe misses exceed 20%. Measurements from rejected controllers are finite-window diagnostics. Stable unmeasured and unstable cases remain separate.','',
        '### Highest-overshoot groups for the tree models (at least 30 plants)','']
    worst=clusters[clusters.family.isin(['random_forest','xgboost'])&clusters.plants.ge(30)].sort_values('overshoot_miss_pct',ascending=False).groupby('family',sort=False).head(4)
    lines+=table(worst,['family','feature','bin','plants','pass_pct','overshoot_miss_pct','severe_miss_pct'])
    lines+=['','These are associations, not causal effects: bins overlap across different features and other parameters are not controlled. Do not sum counts across features. Full bin counts, coverage, and instability rates are in failure_clusters.csv.','',
        '## 4. Learning curves','',
        'Keep the selected hyperparameters and target representation fixed. In each fold, the 128-plant subset is nested within the 256-plant subset, both inside that fold’s full training set. The validation plants do not change. Each size has 800 pooled predictions. Full-fold sizes are 533 or 534 and are labeled 533. Curves are descriptive because settings were selected on the full-size folds.','']
    lines+=table(curves.sort_values(['family','training_size']),['family','training_size','success_pct','median_J_ratio','r2'])
    lines+=['','## Saved artifacts and limits','',
        'Each candidate’s out-of-fold predictions, parameter settings, selected configurations, and learning-curve records are retained. Selected models were refitted on all 800 development plants and saved with feature order and protocol. They have NOT been scored on the 200 test plants. No previous baseline models or results were overwritten.','',
        'Checks concern nominal ideal PID and second-order Padé delay, not exact-delay/hardware robustness. The dataset and penalty weight are fixed throughout; these experiments do not establish global optimality of the labels.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for name,g in curves.groupby('family'):
        g=g.sort_values('training_size');ax.plot(g.training_size,g.success_pct,'o-',label=name)
    ax.set(xlabel='Training plants per fold',ylabel='Out-of-fold full-validation success (%)',
        title='Learning curves: fixed settings and validation folds (development scores)',ylim=(0,100))
    ax.grid(alpha=.2);ax.legend();fig.savefig(OUT/'learning_curves.png',dpi=160)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    ablation.set_index('family').plot.bar(ax=ax);ax.set(ylabel='Full-validation success (%)',
        title='Selection and target scaling: same development folds');ax.tick_params(axis='x',rotation=0)
    ax.legend(fontsize=8);fig.savefig(OUT/'ablation.png',dpi=160)
    print(ablation.to_string(index=False));print(selected[['family','normalized','success_pct']].to_string(index=False))
    print('Report saved to',OUT/'report.md')
