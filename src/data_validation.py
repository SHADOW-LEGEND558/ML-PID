"""Independent checks of generated labels and measured features."""
import numpy as np
from data_gen import Plant, PIDGains, PlantSimulator, FeatureExtractor, PIDOptimizer

def audit(task):
    index,row,weight=task
    plant=Plant(*(row[k] for k in ['zeta','wn','tau','K']))
    gains=PIDGains(*(row[k] for k in ['Kp','Ki','Kd']))
    sim=PlantSimulator();opt=PIDOptimizer(sim,gain_penalty=weight)
    metrics,reason=opt._validate_candidate(plant,gains,sim.get_dynamic_t_eval(plant))
    result={'row':index,'validation':reason}
    t,y=sim.observe_open_loop(plant)
    features=FeatureExtractor().extract(t,y)
    result['max_feature_difference']=max(abs(value-row[key]) for key,value in features.items())
    result['duration_difference']=abs(t[-1]-row['observation_duration'])
    result['relative_final_value_error']=abs(row['y_ss']/plant.K-1)
    result['relative_peak_error']=abs(row['y_peak']/plant.K-(1+np.exp(-np.pi*plant.zeta/np.sqrt(1-plant.zeta**2)) if plant.zeta<1 else 1))
    if metrics is not None:
        result.update(metrics)
        dense=opt._response_metrics(plant,gains,metrics['duration'],int(2*metrics['n_points']-1))
        result['dense_overshoot_pct']=dense['overshoot_pct']
        result['dense_itae_relative_change']=abs(dense['itae']-metrics['itae'])/max(1e-8,metrics['itae'])
        result.update(opt.objective_terms(plant,np.array([gains.Kp,gains.Ki,gains.Kd]),metrics['itae']))
    return result
