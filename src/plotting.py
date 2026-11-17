"""Shared scientific figures for the six-model study and analysis notebook."""
import numpy as np
import pandas as pd
from data_gen import Plant,PIDGains,PlantSimulator

LABELS={'linear':'Linear','ridge':'Ridge','polynomial_ridge':'Polynomial ridge',
        'random_forest':'Random forest','xgboost':'XGBoost','standard_mlp':'MLP'}
GAINS=['Kp','Ki','Kd']

def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'font.size':10})


def performance_plot(metrics):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    labels=[LABELS[k] for k in metrics.index]
    for ax,key,title,limit in zip(axes,['r2','stable_pct','validated_success_pct'],
            ['Gain prediction: mean test R²','Stable controllers (%)','Fully validated controllers (%)'],[(-.1,1),(0,100),(0,100)]):
        values=metrics[key].to_numpy();bars=ax.barh(labels,values,color='#278cb2')
        ax.set_title(title);ax.set_xlim(*limit);ax.invert_yaxis()
        for bar,value in zip(bars,values):ax.text(max(0,value)+.012*(limit[1]-limit[0]),bar.get_y()+bar.get_height()/2,f'{value:.3f}' if key=='r2' else f'{value:.1f}',va='center')
    return fig


def prediction_plot(predictions):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score,mean_squared_error
    fig,axes=plt.subplots(len(predictions),3,figsize=(12,3.2*len(predictions)),layout='constrained')
    for i,(name,frame) in enumerate(predictions.items()):
        for j,gain in enumerate(GAINS):
            ax=axes[i,j];true=frame['target_'+gain];pred=frame['predicted_'+gain]
            for flag,color,label in [(True,'#278cb2','Validated'),(False,'#df7952','Failed validation')]:
                mask=frame.accepted.eq(flag);ax.scatter(true[mask],pred[mask],s=14,alpha=.65,color=color,label=label)
            low=min(0,true.min(),pred.min());high=max(true.max(),pred.max());ax.plot([low,high],[low,high],color='#485567',ls='--',lw=1)
            ax.set_title(f'{LABELS[name]} · {gain}\nR²={r2_score(true,pred):.3f}, RMSE={np.sqrt(mean_squared_error(true,pred)):.3f}')
            ax.set_xlabel('Optimized target gain');ax.set_ylabel('Predicted gain')
    axes[0,0].legend(fontsize=8)
    return fig


def failure_plot(predictions):
    import matplotlib.pyplot as plt
    reasons=['accepted','overshoot','unstable','invalid_gains','duration_limit','resolution_limit','simulation_failure','nonfinite_gains']
    counts=pd.DataFrame({LABELS[k]:v.reason.value_counts() for k,v in predictions.items()}).fillna(0).T
    counts=counts.reindex(columns=[k for k in reasons if k in counts],fill_value=0)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    counts.plot.barh(stacked=True,ax=ax,color=['#278cb2','#e5a044','#d66b65','#a178b5','#7b8794','#b0bac4','#888888','#333333'][:len(counts.columns)])
    ax.invert_yaxis();ax.set_xlabel('Held-out plants');ax.set_title('Primary outcome of raw predictions (no clipping or fallback)')
    ax.legend(title='Outcome',bbox_to_anchor=(1.02,1),loc='upper left')
    return fig


def objective_plot(predictions):
    import matplotlib.pyplot as plt
    values=[f.loc[f.accepted,'objective_ratio'].dropna().to_numpy() for f in predictions.values()]
    labels=[f'{LABELS[k]}\n{len(v)}/{len(predictions[k])} pairs' for k,v in zip(predictions,values)]
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    ax.boxplot(values,showfliers=True)
    ax.set_xticks(range(1,len(labels)+1),labels)
    ax.axhline(1,color='#278cb2',ls='--',label='Equal to target cost');ax.set_yscale('log')
    ax.set_ylabel('Predicted J / target J (log scale)');ax.set_title('Regularized cost: accepted pairs only; failed controllers excluded')
    ax.legend();return fig


def itae_plot(predictions):
    """Raw tracking-error comparisons, only where both controllers validated."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    ratios=[];labels=[]
    for name,f in predictions.items():
        paired=f.accepted & np.isfinite(f.itae_ratio)
        selected=f.loc[paired]
        ratios.append(selected.itae_ratio.to_numpy())
        labels.append(f'{LABELS[name]}\n{len(selected)}/{len(f)} pairs')
        axes[0].scatter(selected.reference_itae,selected.itae,s=10,alpha=.35,label=LABELS[name])
    all_values=np.concatenate([f.loc[f.accepted & f.itae_ratio.notna(),['reference_itae','itae']].to_numpy().ravel() for f in predictions.values()])
    positive=all_values[np.isfinite(all_values)&(all_values>0)]
    if len(positive):
        low,high=positive.min(),positive.max();axes[0].plot([low,high],[low,high],ls='--',color='black',lw=1)
    axes[0].set_xscale('log');axes[0].set_yscale('log')
    axes[0].set(xlabel='Optimized target ITAE (log)',ylabel='Predicted-controller ITAE (log)',title='Raw ITAE: diagonal means equal')
    axes[0].legend(fontsize=7)
    axes[1].boxplot(ratios);axes[1].set_xticks(range(1,len(labels)+1),labels,rotation=25,ha='right')
    axes[1].set_yscale('log');axes[1].axhline(1,color='#278cb2',ls='--')
    axes[1].set(ylabel='ITAE predicted / ITAE target (log)',title='ITAE ratio: 1 means equal')
    change=[100*(r-1) for r in ratios]
    axes[2].boxplot(change);axes[2].set_xticks(range(1,len(labels)+1),labels,rotation=25,ha='right')
    axes[2].set_yscale('symlog',linthresh=10);axes[2].axhline(0,color='#278cb2',ls='--')
    # ITAE is nonnegative, so a relative reduction cannot be below -100%.
    axes[2].set_ylim(bottom=-100)
    axes[2].set(ylabel='ITAE change (%) — symmetric log scale',title='Relative change: 0% means equal')
    fig.suptitle('Validated pairs only; failed/unresolved controllers excluded, coverage shown')
    return fig


def overshoot_summary(predictions):
    """Overshoot diagnostic bins partition every test row; missing is not zero."""
    rows=[]
    for name,frame in predictions.items():
        measured=frame.stable & np.isfinite(frame.overshoot_pct)
        values=frame.loc[measured,'overshoot_pct']
        excess=values[values>10+1e-6]-10
        rows.append({'model':name,'test_plants':len(frame),'measured_stable':int(measured.sum()),
            'at_most_10':int((values<=10+1e-6).sum()),
            'over_10_to_12':int(((values>10+1e-6)&(values<=12)).sum()),
            'over_12_to_15':int(((values>12)&(values<=15)).sum()),
            'over_15_to_20':int(((values>15)&(values<=20)).sum()),
            'over_20':int((values>20).sum()),
            'stable_unmeasured':int((frame.stable&~measured).sum()),
            'unstable':int((~frame.stable).sum()),
            'median_overshoot_pct':values.median(),'p90_overshoot_pct':values.quantile(.9),
            'max_overshoot_pct':values.max(),
            'median_excess_pp':excess.median(),'p90_excess_pp':excess.quantile(.9),
            'full_validation_passes':int(frame.accepted.sum())})
    return pd.DataFrame(rows).set_index('model')


def overshoot_plot(predictions,population_label='test plants'):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    values=[];labels=[]
    thresholds=np.linspace(0,30,601)
    for name,frame in predictions.items():
        v=frame.loc[frame.stable&np.isfinite(frame.overshoot_pct),'overshoot_pct'].to_numpy()
        values.append(v);labels.append(f'{LABELS[name]} ({len(v)}/{len(frame)})')
        axes[0].plot(thresholds,[100*np.sum(v<=limit+1e-6)/len(frame) for limit in thresholds],label=LABELS[name])
    axes[0].axvline(10,color='#b35b60',ls='--',label='Current 10% limit')
    axes[0].set(xlim=(0,30),ylim=(0,100),xlabel='Overshoot threshold (%)',
        ylabel=f'Fraction of ALL {population_label} (%)',title='How many meet each overshoot threshold?')
    axes[0].legend(fontsize=8)
    axes[1].boxplot(values,vert=False,showfliers=True)
    axes[1].set_yticks(range(1,len(labels)+1),labels);axes[1].invert_yaxis()
    axes[1].axvline(10,color='#b35b60',ls='--')
    axes[1].set_xscale('symlog',linthresh=20)
    axes[1].set_xticks([0,10,20,50,100],['0','10','20','50','100'])
    axes[1].set(xlabel='Measured overshoot (%): linear to 20%, logarithmic beyond',
        title='Stable measured predictions: full distribution')
    fig.suptitle('Overshoot diagnostics: threshold compliance alone is not full validation')
    return fig


def overshoot_bins_plot(predictions):
    import matplotlib.pyplot as plt
    summary=overshoot_summary(predictions)
    columns=['at_most_10','over_10_to_12','over_12_to_15','over_15_to_20','over_20','stable_unmeasured','unstable']
    counts=summary[columns].copy()
    counts.index=[LABELS[k] for k in counts.index]
    counts.columns=['<=10%','>10 to 12%','>12 to 15%','>15 to 20%','>20%','Stable, unmeasured','Unstable']
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    counts.plot.barh(stacked=True,ax=ax,color=['#278cb2','#f4d884','#edaf59','#e27d52','#b94d57','#a4a9af','#585b64'])
    ax.invert_yaxis();ax.set_xlabel('Number of held-out plants')
    ax.set_title('How far predictions miss the overshoot limit (every test plant counted)')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left')
    return fig


def example_rows(predictions):
    frame=predictions['random_forest']
    accepted=frame[frame.accepted].sort_values('objective_ratio')
    good=int(accepted.iloc[len(accepted)//2].source_row)
    failed=frame[(~frame.accepted)&frame.stable&frame.overshoot_pct.notna()].sort_values('overshoot_pct')
    bad=int(failed.iloc[len(failed)//2].source_row)
    zero=int(frame[frame.tau.eq(0)].iloc[0].source_row)
    return {'Representative RF success':good,'Representative RF overshoot failure':bad,'Zero-delay example':zero}


def response_plot(predictions,source_row):
    import matplotlib.pyplot as plt
    frames={k:f[f.source_row.eq(source_row)].iloc[0] for k,f in predictions.items()}
    row=next(iter(frames.values()));plant=Plant(row.zeta,row.wn,row.tau,row.K);sim=PlantSimulator()
    duration=max(3.,min(100.,2*row.t_settling));t=np.linspace(0,duration,4001)
    target=PIDGains(*(row['target_'+g] for g in GAINS));_,yt=sim.close_loop(plant,target,t);_,yo=sim.open_loop(plant,t)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,(name,r) in zip(axes.flat,frames.items()):
        with np.errstate(over='ignore',invalid='ignore'):
            _,yp=sim.close_loop(plant,PIDGains(*(r['predicted_'+g] for g in GAINS)),t)
        ax.plot(t,yt,color='#278cb2',label='Optimized target PID');ax.plot(t,yp,color='#df7952',label='Predicted PID')
        ax.plot(t,yo,color='#8e7dac',ls=':',label='Open-loop plant');ax.axhline(1,color='#485567',ls='--',lw=1,label='Setpoint')
        ax.axhline(1.1,color='#b35b60',ls=':',lw=1,label='10% ceiling')
        ax.set_ylim(-.5,max(2.2,1.1*float(np.max(yo))));ax.set_xlim(0,duration)
        ax.set_title(f'{LABELS[name]} · {r.reason}');ax.set_xlabel('Time (s)');ax.set_ylabel('Output')
        if not np.isfinite(yp).all() or np.nanmax(yp)>ax.get_ylim()[1] or np.nanmin(yp)<-.5:
            ax.text(.03,.05,'Predicted response exceeds displayed output range',transform=ax.transAxes,fontsize=8)
    unused_axes=list(axes.flat)[len(frames):]
    for unused in unused_axes:unused.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    if unused_axes:
        unused_axes[-1].legend(handles,labels,loc='center')
    else:
        fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle(f'Test row {source_row}: K={plant.K:.2f}, ωn={plant.wn:.2f}, ζ={plant.zeta:.2f}, τ={plant.tau:.3f}s\nDisplay window only; saved metrics use adaptive full validation.')
    return fig
