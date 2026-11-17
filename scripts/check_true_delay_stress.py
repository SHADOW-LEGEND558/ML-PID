"""Independent fixed-step check of the largest method-of-steps discrepancies.

Uses scalar RK4 and linear stored-history interpolation, rather than solve_ivp
dense history. This is a cross-check, not the primary stress acceptance rule.
"""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'temp/mpl-cache'))
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_stress_suite import ROOT, OUT, TrueDelayResponse, Plant, PIDGains, PlantSimulator
INPUT=OUT


def fixed_step(plant,gains,duration,delay_steps):
    kp,ki,kd=map(float,gains)
    tau=plant.tau
    assert tau>0 and delay_steps>=2
    h=tau/delay_steps
    count=int(np.ceil(duration/h))+1
    states=np.zeros((count,3))
    times=np.arange(count)*h
    states[:delay_steps+1,2]=times[:delay_steps+1]
    w2=plant.wn**2;a=2*plant.zeta*plant.wn;b=plant.K*w2
    states[delay_steps,1]=b*kd
    for n in range(delay_steps,count-1):
        y,v,z=states[n]
        j=n-delay_steps
        y0,v0,z0=states[j]
        y1,v1,z1=states[j+1]
        # Use the left limit at the delayed velocity jump, rather than smearing
        # its discontinuity into the preceding integration interval.
        if j+1==delay_steps:v1=0.
        c0=b*(kp*(1-y0)+ki*z0-kd*v0)
        c1=b*(kp*(1-y1)+ki*z1-kd*v1)
        cm=(c0+c1)/2
        k1y=v;k1v=-w2*y-a*v+c0;k1z=1-y
        yy=y+h*k1y/2;vv=v+h*k1v/2
        k2y=vv;k2v=-w2*yy-a*vv+cm;k2z=1-yy
        yy=y+h*k2y/2;vv=v+h*k2v/2
        k3y=vv;k3v=-w2*yy-a*vv+cm;k3z=1-yy
        yy=y+h*k3y;vv=v+h*k3v
        k4y=vv;k4v=-w2*yy-a*vv+c1;k4z=1-yy
        states[n+1]=[y+h*(k1y+2*k2y+2*k3y+k4y)/6,
                     v+h*(k1v+2*k2v+2*k3v+k4v)/6,
                     z+h*(k1z+2*k2z+2*k3z+k4z)/6]
    return times,states[:,0]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'plots').mkdir(exist_ok=True)
    result=pd.read_csv(INPUT/'exact_delay_paired.csv')
    result['peak_difference']=result.overshoot_pct-result.pade_overshoot_pct
    mlp=result[(result.model=='standard_mlp')&result.pade_accepted].nlargest(3,'peak_difference')
    reference=result[result.model=='numerical_reference'].nlargest(2,'peak_difference')
    cases=pd.concat([mlp,reference])
    cohort=pd.read_csv(INPUT/'cohort.csv').set_index('source_row')
    gains=pd.read_csv(INPUT/'nominal_gains.csv').set_index(['source_row','model'])
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for i,r in enumerate(cases.itertuples()):
        row=cohort.loc[r.source_row]
        plant=Plant(row.zeta,row.wn,row.tau,row.K)
        vector=gains.loc[(r.source_row,r.model),['gain_Kp','gain_Ki','gain_Kd']].to_numpy()
        duration=float(r.duration)
        desired=np.linspace(0,duration,16001)
        dense=TrueDelayResponse(plant,vector,duration,1e-10)
        expected=dense.output(desired)
        record=dict(source_row=r.source_row,model=r.model,delay=plant.tau,duration=duration,
                    primary_peak_pct=r.overshoot_pct,pade_peak_pct=r.pade_overshoot_pct)
        for steps in [200,400,800,1600]:
            t,y=fixed_step(plant,vector,duration,steps)
            interp=np.interp(desired,t,y)
            peak=max(0.,100*(y.max()-1))
            itae=float(simpson(desired*abs(1-interp),x=desired))
            error=float(abs(interp-expected).max())
            record.update({f'rk4_{steps}_peak_pct':peak,f'rk4_{steps}_itae':itae,
                           f'rk4_{steps}_max_curve_error':error})
            if steps>=400 and error<.0001 and abs(peak-r.overshoot_pct)<.01:
                break
        # Entire response agreement, not merely a matching pass/fail category.
        assert error<.0001 and abs(peak-r.overshoot_pct)<.01,record
        record.update(final_delay_steps=steps,final_max_curve_error=error,
                      final_peak_error_pct=abs(peak-r.overshoot_pct))
        rows.append(record)
        if i<3:
            axis=axes[i]
            show=np.linspace(0,min(duration,max(6.,8/plant.wn+6*plant.tau)),1601)
            pade=PlantSimulator().close_loop(plant,PIDGains(*vector),show)[1]
            axis.plot(show,pade,label='Second-order Padé',color='#555555',ls='--')
            axis.plot(show,dense.output(show),label='True delay: method of steps',color='#c23b41')
            axis.plot(show,np.interp(show,t,y),label='Independent RK4',color='#3979a8',ls=':')
            axis.axhline(1,color='#888888',lw=.7)
            axis.axhline(1.1,color='#888888',lw=.7,ls=':')
            axis.set(xlabel='Time (s)',ylabel='Output',title=f'MLP, plant {r.source_row}\nPadé {r.pade_overshoot_pct:.1f}% vs true {r.overshoot_pct:.1f}% overshoot')
            axis.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT/'plots/true_delay_crosscheck.png',dpi=170)
    fig.savefig(OUT/'plots/true_delay_crosscheck.svg')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'true_delay_crosscheck.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Five worst-case cross-checks passed, including complete-curve agreement.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT)
    parser.add_argument('--output',type=Path,default=ROOT/'temp/true_delay_crosscheck')
    args=parser.parse_args();INPUT=args.input.resolve();OUT=args.output.resolve()
    published=ROOT/'results/dataset_final_10000'
    if OUT==published or published in OUT.parents:raise ValueError('Use a separate output directory.')
    main()
