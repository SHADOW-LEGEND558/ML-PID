"""Summarize frozen-model predictions on the recovered rejected plants."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
if __name__=='__main__':matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_gen import Plant,PIDGains,PlantSimulator
from plot_baselines import LABELS

OUT=ROOT/'results/rejected_plants'
ORDER=['linear','ridge','polynomial_ridge','random_forest','xgboost','standard_mlp']
LABELS['standard_mlp']='Standard MLP'


def load():
    predictions=pd.read_csv(OUT/'predictions.csv')
    plants=pd.read_csv(OUT/'recovered_plants.csv')
    summary=pd.read_csv(OUT/'summary.csv').set_index('model').loc[ORDER]
    return predictions,plants,summary


def success_plot(summary):
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    x=np.arange(len(summary));width=.36
    ax.bar(x-width/2,100*summary.strict_passes/summary.plants,width,label='Original budget (65,537 points)',color='#278cb2')
    ax.bar(x+width/2,100*summary.passes_with_extended_budget/summary.plants,width,label='Retry resolution limits (262,145 points)',color='#e5a044')
    ax.set_xticks(x,[LABELS[k] for k in summary.index],rotation=15,ha='right')
    ax.set(ylabel='Accepted controllers (%)',ylim=(0,100),title='Rejected-generation stress set: resource budgets reported separately')
    ax.grid(axis='y',alpha=.2);ax.legend();return fig


def outcome_plot(predictions,extended=False):
    column='extended_reason' if extended else 'reason'
    counts=pd.crosstab(predictions.model,predictions[column]).reindex(ORDER)
    counts.index=[LABELS[k] for k in counts.index]
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    counts.plot.barh(stacked=True,ax=ax);ax.invert_yaxis()
    ax.set(xlabel='Rejected plants',title='Prediction outcomes: '+('extended retry policy' if extended else 'original validation budget'))
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left');return fig


def example_ids(predictions):
    mlp=predictions[predictions.model.eq('standard_mlp')]
    return {'Passes original budget':int(mlp[mlp.accepted].iloc[0].rejection_id),
        'Passes only larger budget':int(mlp[(~mlp.accepted)&mlp.extended_accepted].iloc[0].rejection_id),
        'Overshoot failure':int(mlp[mlp.extended_reason.eq('overshoot')].iloc[0].rejection_id),
        'Unresolved sampling limit':int(mlp[mlp.extended_reason.eq('resolution_limit')].iloc[0].rejection_id)}


def response_plot(predictions,plant_id):
    chosen=predictions[predictions.rejection_id.eq(plant_id)].set_index('model').loc[ORDER]
    r=chosen.iloc[0];plant=Plant(r.zeta,r.wn,r.tau,r.K);sim=PlantSimulator()
    duration=max(3.,min(100.,2*r.t_settling));t=np.linspace(0,duration,4001)
    _,open_y=sim.open_loop(plant,t)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,(name,row) in zip(axes.flat,chosen.iterrows()):
        gains=PIDGains(row.predicted_Kp,row.predicted_Ki,row.predicted_Kd)
        with np.errstate(over='ignore',invalid='ignore'):_,y=sim.close_loop(plant,gains,t)
        ax.plot(t,y,color='#278cb2',label='Predicted PID');ax.plot(t,open_y,color='#9b7db6',ls=':',label='Open-loop')
        ax.axhline(1,color='#485567',ls='--',label='Setpoint');ax.axhline(1.1,color='#b35b60',ls=':',label='10% ceiling')
        ax.set(xlim=(0,duration),ylim=(-.5,max(2.2,1.1*open_y.max())),xlabel='Time (s)',ylabel='Output')
        ax.set_title(f'{LABELS[name]}\nstrict: {row.reason}; retry: {row.extended_reason}',fontsize=10)
        if not np.isfinite(y).all() or np.nanmax(y)>ax.get_ylim()[1] or np.nanmin(y)<-.5:
            ax.text(.02,.04,'Predicted response exceeds displayed range',transform=ax.transAxes,fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f'Rejected plant {plant_id}: K={plant.K:.2f}, wn={plant.wn:.2f}, zeta={plant.zeta:.2f}, tau={plant.tau:.5f}s\nDisplay is illustrative, NOT a validation certificate; no optimized reference gains exist.')
    return fig


def mdtable(frame):
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for _,r in frame.iterrows():lines.append('| '+' | '.join(str(r[k]) for k in frame.columns)+' |')
    return lines


if __name__=='__main__':
    results,plants,summary=load()
    breakdown=results.groupby(['model','original_reason'],sort=False).agg(plants=('accepted','size'),
        strict_passes=('accepted','sum'),passes_with_extended_budget=('extended_accepted','sum')).reset_index()
    breakdown.to_csv(OUT/'by_original_rejection.csv',index=False)
    for flag,column in [('strict','reason'),('extended','extended_reason')]:
        pd.crosstab(results.model,results[column]).to_csv(OUT/f'{flag}_outcomes.csv')
    figures={'success_rates':success_plot(summary),'strict_outcomes':outcome_plot(results),
             'extended_outcomes':outcome_plot(results,True)}
    examples=example_ids(results)
    for title,index in examples.items():figures[f'response_{index}']=response_plot(results,index)
    for name,fig in figures.items():fig.savefig(OUT/f'{name}.png',dpi=160);plt.close(fig)
    grouped=plants.original_reason.value_counts().to_dict()
    lines=['# Frozen-model stress test on rejected generation plants','',
        '## Records recovered','',
        'All 142 unique rejected plants were recovered from data/generation_20000.log, matching 20,142 attempts minus 20,000 accepted rows. Parameters are retained at the precision printed by the generator. The five empirical features were recomputed using the unchanged observation rule. No rejected plant overlaps the accepted dataset (parameter comparison tolerance 1e-12).','',
        f'Original plant-level rejection groups: {grouped}. These are plants, not summed per-start failure counts. Earlier shorthand 103/39 counts obtained by dividing start-level totals by four were inaccurate because four plants had mixed reasons.','',
        '## Protocol','',
        'All six final frozen models from results/dataset_20000 were loaded. No training, numerical optimization, parameter adjustment, clipping or controller fallback was performed. Model-file hashes were checked before and after the experiment. All models use only the five measured features at inference; physical parameters are used by simulation/validation only.','',
        'First, each raw prediction was assessed using the original 65,537-point maximum and all existing checks. Only predictions failing with resolution_limit were retried with up to 262,145 points. This secondary resource policy leaves the 10% overshoot threshold, gain bounds, pole criterion, tail tolerance, convergence tolerances, and 1,000 s horizon cap unchanged. Original successes are retained; other failure categories are not retried. Strict and secondary outcomes are reported separately.','',
        '## Results','']
    lines+=mdtable(summary.reset_index())
    mlp=results[results.model.eq('standard_mlp')]
    lines+=['',
        f"Standard MLP: {mlp.accepted.sum()}/142 pass at the original budget; {mlp.extended_accepted.sum()}/142 pass with the secondary budget policy. All {mlp.stable.sum()}/142 predictions are pole-stable. Secondary outcomes: {mlp.extended_reason.value_counts().to_dict()}.",'',
        'The low strict pass rate is largely a resource-limited validation outcome, not evidence of instability. Conversely, a stable pole test or a plausible plot does not certify overshoot and numerical convergence for the unresolved plants. The 22 MLP cases still at resolution_limit remain unresolved; they are not counted as passes.','',
        '## Original rejection group breakdown','']+mdtable(breakdown)
    lines+=['','## What this establishes','',
        'Models can propose validated controllers for some plants the multi-start labeling pipeline rejected. This demonstrates that generator rejection does not establish PID control is impossible. The MLP also passes many cases whose original optimized candidates violated overshoot.','',
        'The recovered set is deliberately rejection-biased and is not an independent random test sample. These pass rates must not replace or be mixed with the 97.92% MLP result on the accepted-data test distribution. They reveal a label-selection/coverage limitation near difficult plants, especially tiny delays and stiff Padé dynamics.','',
        'No accepted optimized gains or objective values were saved for these plants. Therefore gain MAE/RMSE/R², target ITAE ratios, and claims of optimality cannot be computed. Absolute ITAE and J are saved for validated predictions, with coverage; comparing conditional averages across different passing subsets can be misleading.','',
        'Plots use illustrative finite display windows. Unresolved or divergent curves can look plausible initially, and outputs can extend beyond the display bounds. The checks concern ideal PID and second-order Padé delay, not exact-delay/hardware robustness.','',
        'Artifacts: recovered_plants.csv, predictions.csv, summary.csv, by_original_rejection.csv, separate strict/extended outcome tables, protocol.json, and the accompanying response figures. A standalone notebook is notebooks/rejected_plants_stress_test.ipynb. The accepted dataset, main models, and main benchmark scores are unchanged.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary.to_string());print('Example IDs:',examples)
