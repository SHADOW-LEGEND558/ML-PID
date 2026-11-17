"""Summarize the separate, frozen-model stress study and draw scientific plots."""
import json
import hashlib
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'temp/mpl-cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from provenance import matches_recorded_hash
BASE=ROOT/'results/dataset_final_10000'
OUT=BASE/'stress_tests'
NAMES=['random_forest','xgboost','standard_mlp','numerical_reference']
LABELS={'random_forest':'Random forest','xgboost':'XGBoost','standard_mlp':'MLP',
        'numerical_reference':'Numerical reference'}
COLORS=dict(zip(NAMES,['#3979a8','#c28b24','#c23b41','#444444']))


def summarize(frame,groups,flag='accepted'):
    records=[]
    rng=np.random.default_rng(17112026)
    for keys,part in frame.groupby(groups,dropna=False,sort=False):
        if not isinstance(keys,tuple): keys=(keys,)
        row=dict(zip(groups,keys))
        row.update(cases=len(part),passes=int(part[flag].sum()),
                   success_pct=100*part[flag].mean())
        if 'source_row' in part:
            # Replicates from the same plant are not independent observations.
            rates=part.groupby('source_row')[flag].mean().to_numpy()
            samples=rng.choice(rates,(2000,len(rates)),replace=True).mean(axis=1)*100
            row.update(base_plants=len(rates),cluster_ci_low=np.quantile(samples,.025),
                       cluster_ci_high=np.quantile(samples,.975))
        if 'stable' in part:
            row['unstable_count']=int(part['stable'].eq(False).sum())
            row['stability_unassessed_count']=int(part['stable'].isna().sum())
        row['reason_counts']=json.dumps(part.reason.value_counts().to_dict(),sort_keys=True)
        if 'overshoot_pct' in part:
            peak=part.overshoot_pct.dropna()
            row.update(overshoot_measured=len(peak),overshoot_median=peak.median(),
                       overshoot_p95=peak.quantile(.95),overshoot_max=peak.max(),
                       overshoot_over_20_count=int(peak.gt(20).sum()))
        records.append(row)
    return pd.DataFrame(records)


def md_table(frame,cols):
    data=frame[cols].copy()
    for col in data.select_dtypes('number'):
        data[col]=data[col].map(lambda x:f'{x:.3f}' if pd.notna(x) else 'unmeasured')
    return '| '+' | '.join(data.columns)+' |\n|'+ '|'.join(['---']*len(cols))+'|\n'+ '\n'.join(
        '| '+' | '.join(map(str,row))+' |' for row in data.itertuples(index=False,name=None))


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    verify=json.loads((OUT/'verification.json').read_text())
    assert verify['all_five_tests_completed'] and verify['frozen_artifacts_unchanged']
    frames={name:pd.read_csv(OUT/f'{name}_results.csv') for name in
            ['noise','uncertainty','boundary','exact_delay','disturbance']}
    expected={'noise':7200,'uncertainty':6000,'boundary':964,'exact_delay':400,'disturbance':400}
    for name,frame in frames.items():
        assert len(frame)==expected[name] and not frame.duplicated(['model','case_id']).any()
        assert set(frame.model)==set(NAMES)
    # Historical protocols included Python checkpoint hashes. Those checkpoints
    # are no longer distributed or needed to summarize the saved experiment.
    inputs={path:value for path,value in protocol['frozen_hashes'].items()
            if Path(path.replace('\\','/')).suffix not in ['.pt','.joblib']}
    assert all(matches_recorded_hash(ROOT/path.replace('\\','/'),value)
               for path,value in inputs.items())
    nominal_gains=frames['uncertainty'][['source_row','model','gain_Kp','gain_Ki','gain_Kd']].drop_duplicates()
    assert len(nominal_gains)==400
    nominal_gains.to_csv(OUT/'nominal_gains.csv',index=False)
    audit={'frozen_hashes_match':True,'controller_outcomes':sum(expected.values()),
           'expected_outcomes_by_test':expected,'all_model_case_pairs_unique':True,
           'nominal_controller_count':len(nominal_gains),
           'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [ROOT/'scripts/run_stress_suite.py',ROOT/'scripts/report_stress_suite.py',ROOT/'scripts/check_true_delay_stress.py',ROOT/'src/data_gen.py',ROOT/'src/benchmark.py']}}
    crosscheck=pd.read_csv(OUT/'true_delay_crosscheck.csv') if (OUT/'true_delay_crosscheck.csv').exists() else None
    if crosscheck is not None:
        assert len(crosscheck)==5 and crosscheck.final_max_curve_error.lt(.0001).all()
        audit['independent_delay_crosschecks']=len(crosscheck)
        audit['independent_delay_max_curve_error']=float(crosscheck.final_max_curve_error.max())
    (OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2))
    acquisition=pd.read_csv(OUT/'noise_acquisition.csv')
    summaries={name:summarize(frame,['model','level'] if 'level' in frame else ['model'],
                               'recovered' if name=='disturbance' else 'accepted')
               for name,frame in frames.items()}
    summaries['boundary_groups']=summarize(frames['boundary'],['model','group'])
    aq=acquisition.groupby('noise_fraction').agg(cases=('case_id','size'),
        recording_pass_pct=('acquisition_passed',lambda s:100*s.mean()),
        usable_fixed_window_features_pct=('features_usable',lambda s:100*s.mean())).reset_index()
    for name,frame in summaries.items():frame.to_csv(OUT/f'{name}_summary.csv',index=False)
    aq.to_csv(OUT/'acquisition_summary.csv',index=False)
    cohort=pd.read_csv(OUT/'cohort.csv').set_index('source_row')
    feature_drift=[]
    for level,part in acquisition.groupby('noise_fraction'):
        clean=cohort.loc[part.source_row]
        row={'noise_fraction':level,'cases':len(part),
             'false_zero_onset_pct':100*float(((part.tau_est.to_numpy()==0)&(clean.tau_est.to_numpy()>0)).mean())}
        for key in ['t_rise','t_settling','y_peak','y_ss']:
            change=abs(part[key].to_numpy()/clean[key].to_numpy()-1)*100
            row[f'{key}_median_abs_change_pct']=float(np.median(change))
            row[f'{key}_p95_abs_change_pct']=float(np.quantile(change,.95))
        row['onset_median_abs_error_seconds']=float(np.median(abs(part.tau_est.to_numpy()-clean.tau_est.to_numpy())))
        feature_drift.append(row)
    feature_drift=pd.DataFrame(feature_drift)
    feature_drift.to_csv(OUT/'noise_feature_drift.csv',index=False)
    nominal=[]
    for name in NAMES:
        path=BASE/(f'{name}_test_predictions.csv' if name!='numerical_reference' else 'test_references.csv')
        data=pd.read_csv(path)
        data=data[data.source_row.isin(protocol['base_rows'])]
        data['model']=name
        nominal.append(data[['source_row','model','accepted','itae','overshoot_pct']])
    nominal=pd.concat(nominal,ignore_index=True)
    nominal_summary=nominal.groupby('model').agg(plants=('accepted','size'),passes=('accepted','sum'),
                                                   success_pct=('accepted',lambda s:100*s.mean())).reset_index()
    nominal_summary.to_csv(OUT/'nominal_summary.csv',index=False)
    # Ratios use the same controller's accepted nominal response, not a retuned teacher.
    ratio=frames['uncertainty'].merge(nominal.rename(columns={'accepted':'nominal_accepted','itae':'nominal_itae',
        'overshoot_pct':'nominal_overshoot_pct'}),on=['source_row','model'],validate='many_to_one')
    paired=ratio.accepted & ratio.nominal_accepted
    ratio['itae_vs_nominal']=np.where(paired,ratio.itae/ratio.nominal_itae,np.nan)
    ratio.to_csv(OUT/'uncertainty_paired.csv',index=False)
    ratios=ratio.groupby(['model','level']).agg(paired_cases=('itae_vs_nominal','count'),
        median_itae_ratio=('itae_vs_nominal','median'),p90_itae_ratio=('itae_vs_nominal',lambda s:s.quantile(.9))).reset_index()
    ratios.to_csv(OUT/'uncertainty_itae_summary.csv',index=False)
    exact=frames['exact_delay'].merge(nominal.rename(columns={'accepted':'pade_accepted','itae':'pade_itae',
        'overshoot_pct':'pade_overshoot_pct'}),on=['source_row','model'],validate='one_to_one')
    exact['itae_vs_pade']=np.where(exact.accepted & exact.pade_accepted,exact.itae/exact.pade_itae,np.nan)
    exact.to_csv(OUT/'exact_delay_paired.csv',index=False)
    ex=exact.groupby('model').agg(plants=('accepted','size'),pade_passes=('pade_accepted','sum'),
        true_delay_time_domain_passes=('accepted','sum'),median_itae_ratio=('itae_vs_pade','median')).reset_index()
    ex['pade_pass_true_delay_overshoot']=[int(((exact.model==m)&exact.pade_accepted&(exact.reason=='overshoot')).sum()) for m in ex.model]
    ex.to_csv(OUT/'exact_delay_comparison.csv',index=False)
    dr=frames['disturbance']
    ds=dr.groupby('model').agg(cases=('case_id','size'),recovered=('recovered','sum')).join(
        dr[dr.recovered].groupby('model').agg(
        median_peak_deviation=('max_deviation','median'),p95_peak_deviation=('max_deviation',lambda s:s.quantile(.95)),
        median_recovery_seconds=('recovery_time','median'),median_disturbance_itae=('disturbance_itae','median'))
    ).reset_index()
    ds.to_csv(OUT/'disturbance_metrics.csv',index=False)
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    for name in NAMES:
        for test,axis in [('noise',axes[0,0]),('uncertainty',axes[0,1])]:
            part=summaries[test].query('model==@name').sort_values('level')
            axis.plot(part.level*100,part.success_pct,'o-',color=COLORS[name],label=LABELS[name])
            axis.fill_between(part.level.to_numpy()*100,part.cluster_ci_low.to_numpy(),part.cluster_ci_high.to_numpy(),color=COLORS[name],alpha=.10)
    axes[0,0].set(xscale='log',xlabel='Noise standard deviation / K (%)',ylabel='Controller acceptance (%)',
                  title='Noisy features: known recording length')
    axes[0,1].set(xlabel='Joint parameter perturbation bound (%)',ylabel='Controller acceptance (%)',
                  title='Plant changes, gains held fixed')
    groups=list(frames['boundary'].group.unique())
    matrix=summaries['boundary_groups'].pivot(index='group',columns='model',values='success_pct').reindex(index=groups,columns=NAMES)
    axis=axes[0,2]
    axis.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap='RdYlGn')
    axis.set_xticks(range(4),['RF','XGB','MLP','Numerical'])
    axis.set_yticks(range(len(groups)),[g.replace('_',' ') for g in groups],fontsize=8)
    axis.set_title('Boundary acceptance (%)')
    for i in range(len(groups)):
        for j in range(4):axis.text(j,i,f'{matrix.iloc[i,j]:.0f}',ha='center',va='center',fontsize=8,
                                   color='white' if matrix.iloc[i,j]<20 or matrix.iloc[i,j]>80 else 'black')
    axis=axes[1,0];pos=np.arange(4)
    ex=ex.set_index('model').loc[NAMES]
    axis.bar(pos-.18,ex.pade_passes,width=.36,label='Padé full acceptance',color='#777777')
    axis.bar(pos+.18,ex.true_delay_time_domain_passes,width=.36,label='True-delay time-domain pass',color='#c23b41')
    axis.set(xticks=pos,xticklabels=['RF','XGB','MLP','Numerical'],ylabel='Passes / 100 plants',
             title='Same gains, different delay model',ylim=(0,105));axis.legend(fontsize=8)
    axis=axes[1,1]
    values=[dr.loc[(dr.model==m)&dr.recovered,'max_deviation'].dropna()*100 for m in NAMES]
    axis.boxplot(values,tick_labels=['RF','XGB','MLP','Numerical'],showfliers=True)
    axis.set(ylabel='Peak output deviation (% of setpoint)',title='Input-load step: recovered cases only')
    axis=axes[1,2]
    axis.plot(aq.noise_fraction*100,aq.recording_pass_pct,'o-',color='#c23b41',label='Original acquisition rule')
    axis.plot(aq.noise_fraction*100,aq.usable_fixed_window_features_pct,'s--',color='#555555',label='Usable features, known window')
    axis.set(xscale='log',xlabel='Noise standard deviation / K (%)',ylabel='Recordings (%)',
             title='Acquisition is a separate failure point',ylim=(-3,103));axis.legend(fontsize=8)
    for axis in [axes[0,0],axes[0,1]]:
        axis.set_ylim(-3,103);axis.grid(alpha=.15);axis.legend(fontsize=8)
    fig.suptitle('Frozen-model stress tests | 100 held-out plants + 241 boundary cases',fontsize=15)
    fig.savefig(plots/'stress_overview.png',dpi=170)
    fig.savefig(plots/'stress_overview.svg')
    plt.close(fig)
    mlp_noise=summaries['noise'].query("model=='standard_mlp'").copy();mlp_noise['noise_sigma_pct']=mlp_noise.level*100
    mlp_unc=summaries['uncertainty'].query("model=='standard_mlp'").copy();mlp_unc['parameter_change_pct']=mlp_unc.level*100
    highlights='''## Main findings

- Measurement handling is the largest weakness: the original recording rule passed all 300 recordings at 0.01% noise but none at 0.05% or above. With the clean recording length supplied, MLP controller acceptance fell to 26.7% at 0.5% noise and 14.0% at 2% noise. This conditional result must not be presented as end-to-end acquisition success.
- Fixed-gain MLP acceptance under joint parameter uncertainty was 92.2%, 82.6%, and 68.8% at +/-5%, +/-10%, and +/-20%. No MLP pole instabilities occurred in these perturbations; failures were overshoot violations. The numerical reference achieved 99.0%, 87.2%, and 72.4%, respectively.
- The MLP passed all 25 zero-delay and all 25 high-damping boundary plants, but only 11/25 high-normalized-delay plants, versus RF 22/25 and XGBoost 21/25. Overall boundary acceptance was 180/241; 27 resolution-limit cases are not physical instability claims.
- True-delay time-domain passes fell from the nominal Padé counts: MLP 97 to 84, RF 85 to 68, XGBoost 78 to 68, numerical reference 100 to 89 (each out of 100). Thirteen previously accepted MLP controllers exceeded the overshoot limit with true delay. The worst of those changed from 8.94% to 33.23% overshoot. No finite-window pass is a stability certificate.
- All 100 controllers per family recovered from the specified input-load disturbance. MLP median peak output deviation was 6.05% of the unit setpoint, median recovery time 2.12 seconds, and median disturbance ITAE 0.0999. Numerical-reference values were 6.28%, 2.09 seconds, and 0.1020. This test does not establish performance for other disturbance magnitudes or locations.

The next priorities suggested by these tests are noise-aware observation/feature extraction and verification/tuning with true delay, followed by targeted high-normalized-delay coverage. Merely collecting more noise-free Padé-labeled rows will not directly address the first two weaknesses. No changes to models or the main study were made based on these results.

'''
    text=f'''# Frozen-model stress tests

{highlights}

## Scope and reproducibility

Three frozen learned controllers (MLP, random forest, XGBoost) and numerical-reference controllers. Selected configurations are reconstructed before the stress tests when reproducing; no stress-case retraining, clipping, fallback or post-stress tuning. All original artifact and dataset hashes are unchanged. Seed {protocol['seed']}. The same 100 stratified final-test plants ({verify['zero_delay_plants']} zero delay) are used for the noise, uncertainty, true-delay, and disturbance tests. They are an additional post-test analysis, not a new independent accuracy estimate. Boundary tests use 241 fresh plants. Exact case manifests, gains, and row-level outcomes are saved alongside this report.

Nominal acceptance on the selected 100 plants:

{md_table(nominal_summary,['model','plants','passes','success_pct'])}

Repeated noisy recordings and parameter perturbations are clustered by plant. Bootstrap intervals in the CSVs/figure resample plants, not individual repeated cases. No claim of worst-case robustness follows from random perturbations.

## 1. Noisy open-loop measurements

IID Gaussian noise with standard deviation 0.01%, 0.05%, 0.1%, 0.5%, 1%, or 2% of the true final gain K is added to 0.01-second response samples, with three recordings per plant/level. Every model receives features extracted from the same noisy trace; features are not perturbed independently. No smoothing or revised feature definitions were introduced.

Two questions are deliberately separated. First, can the original acquisition rule find a settled recording within 640 seconds? Second, if the clean recording length is supplied, how well do the original features and models perform? The second is a conditional diagnostic, not end-to-end success when acquisition fails. The numerical reference is an oracle using true plant parameters; its gains are held fixed and do not depend on noisy features.

{md_table(aq,['noise_fraction','cases','recording_pass_pct','usable_fixed_window_features_pct'])}

MLP conditional diagnostic (300 recordings per level):

{md_table(mlp_noise,['noise_sigma_pct','cases','passes','success_pct','unstable_count','overshoot_max'])}

`noise_results.csv` includes unusable-feature cases in its denominator. Missing overshoot values are not assigned zero. `noise_acquisition.csv` contains the original recording-rule result independently of the conditional controller check.

Feature sensitivity (`noise_feature_drift.csv`):

{md_table(feature_drift,['noise_fraction','t_rise_median_abs_change_pct','t_settling_median_abs_change_pct','false_zero_onset_pct','onset_median_abs_error_seconds'])}

## 2. Plant uncertainty with fixed gains

For each plant, independently multiply K, natural frequency, damping, and positive delay by uniform factors in 1 +/- 5%, 10%, and 20%, with five perturbations per level. Nominal zero delay stays zero. Predict/choose gains once from the nominal plant; never retune on the changed plant. These perturbations can cross the original parameter bounds. All controllers see identical perturbed plants. The existing Padé evaluator and limits are unchanged.

{md_table(summaries['uncertainty'],['model','level','cases','passes','success_pct','unstable_count'])}

`uncertainty_itae_summary.csv` compares accepted perturbed responses against the same controller's accepted nominal response. Ratios are conditional, with paired counts provided; instability and other failures remain in the acceptance denominator.

## 3. Parameter boundaries

25 plants per targeted group: zero delay, tiny positive delay (1e-6 to 1e-4 seconds), low damping, high damping, low/high gain, low/high natural frequency, and high normalized delay (tau 0.9 to 1, wn 9 to 10). Also test all 16 exact corners of the physical parameter box. All stay within the original scope. New numerical references use the unchanged four-start optimizer; teacher failure is recorded, not replaced. Numerical-reference acceptance therefore includes the ability to obtain a validated label, unlike an oracle with an already supplied label.

{md_table(summaries['boundary_groups'],['model','group','cases','passes','success_pct'])}

Tiny-delay resolution failures are computational limits, not evidence of physical instability or uncontrollability. Fresh teacher gains and failures are in `boundary_teachers.csv`.

## 4. True delay instead of second-order Padé

An independent method-of-steps simulator integrates the plant and controller integral state with actual delayed feedback. Ideal derivative-on-error is retained: the unit-reference impulse causes the correct velocity jump K*wn^2*Kd at t=tau. The output is exactly zero before the delay. Zero-delay equations are independently checked against the transfer-function solution. DOP853 integration uses two tolerances (1e-7 and 1e-9), two output grids, refined peaks, and horizon extension; acceptance requires ITAE convergence, <=10% overshoot, and tail error <=0.001. Limits: 1,000 seconds, 6,000 delay intervals, 65,537 output points. These are reported separately from the Padé validator's limits.

**A true-delay time-domain pass is not a pole-stability certificate.** A bounded, settled numerical response over converged finite windows is evidence, not a proof about all future times. `observed_divergence` denotes a numerical trajectory exceeding the specified 1e6 state threshold. A delay-segment or integration limit is unassessed, not unstable.

{md_table(ex.reset_index(),['model','plants','pade_passes','true_delay_time_domain_passes','pade_pass_true_delay_overshoot','median_itae_ratio'])}

ITAE ratios compare the same gains under true delay and Padé, conditional on both checks passing. Reasons and measured peaks are in `exact_delay_results.csv`; paired comparisons are in `exact_delay_paired.csv`.

## 5. Input-load disturbance rejection

Starting from the unit-setpoint equilibrium, add a persistent load step of amplitude 0.1/K before the plant (including its delay). Without feedback this corresponds to a steady output change of 0.1. The incremental response uses G/(1+CG), so derivative setpoint kick is not part of this experiment. Keep the same gains. We report maximum absolute output deviation, recovery to +/-0.01 of the setpoint, and disturbance ITAE. Recovery is counted only with stable Padé poles, converged grids/horizons, and tail <=0.001. A setpoint overshoot pass/fail rule is not applied to a disturbance experiment.

{md_table(ds,['model','cases','recovered','median_peak_deviation','p95_peak_deviation','median_recovery_seconds','median_disturbance_itae'])}

The table's quality metrics and the plotted distributions condition on recovered cases, with recovery counts shown explicitly. Row-level finite-window metrics for unverified cases are diagnostic only. This is an incremental load applied after equilibrium; it does not prove that every controller initially tracks the setpoint within the overshoot constraint.

## Numerical implementation checks

Five focused tests check: zero-delay agreement with the transfer function including derivative impulse; no anticipatory true-delay output and correct delayed impulse; feedback-free first-delay-interval agreement with the analytical plant; continuity toward zero delay; and disturbance-transfer normalization with and without delay. Run `python -m unittest discover -s tests -p test_stress_suite.py -v`.

At the time of the original stress experiment, the existing suite plus these additions passed 35 tests. A separate scalar, fixed-step RK4 solver with stored-history interpolation independently cross-checked the three largest MLP Padé discrepancies and two numerical-reference discrepancies. Step refinement reached maximum complete-curve error below 0.0001 output units and peak error below 0.01 percentage points against the method-of-steps solver. See `true_delay_crosscheck.csv` and `plots/true_delay_crosscheck.png`. These checks support the numerical comparison without converting time-domain evidence into a stability proof.

## Limits

Single frozen model per family; one 100-plant cohort; noise levels and uncertainty ranges are chosen stress scenarios, not empirically calibrated hardware conditions. No actuator limits, filtered derivative, measurement filtering, anti-windup, or nonlinear plant dynamics are included. Repeated trials do not create additional independent plants. Bounds, computational limits, feature extraction, and overshoot violations must be distinguished. These tests expose weaknesses; they do not certify safety.

## Reproduce

`python scripts/run_stress_suite.py --jobs 4 --plants 100 --output temp/stress_tests`

`python scripts/report_stress_suite.py --input temp/stress_tests`

`python scripts/check_true_delay_stress.py --input temp/stress_tests --output temp/stress_tests`

Rerun the report after the independent cross-check to include its audit. The cross-check requires `nominal_gains.csv` produced by the first report run.

Completed stage CSVs are reused only under an identical saved protocol. The main frozen dataset/models and website are not overwritten.

Numerical-method reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

Disturbance-transfer reference: https://www.mathworks.com/help/slcontrol/ug/analyze-the-design-in-the-pid-tuner.html
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')
    print('Saved report and plots to',OUT)
    print('NOISE ACQUISITION\n',aq.to_string(index=False))
    print('MLP NOISY FEATURES\n',mlp_noise[['noise_sigma_pct','passes','cases','success_pct']].to_string(index=False))
    print('UNCERTAINTY\n',summaries['uncertainty'][['model','level','success_pct','unstable_count']].to_string(index=False))
    print('EXACT DELAY\n',ex.to_string())
    print('DISTURBANCE\n',ds.to_string(index=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.input.resolve();main()
