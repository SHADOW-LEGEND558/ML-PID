"""Shared rejection-log parsing and validation for the final 10K study."""
import ast
import json
import re

import numpy as np
import pandas as pd

from benchmark import assess_controller
from data_gen import Plant, PIDGains, PlantSimulator, PIDOptimizer


def recover_rejections(log_text):
    pattern=r'(Discarded|Failed) Plant\(zeta=([^,]+), wn=([^,]+), tau=([^,]+), K=([^\)]+)\): ([^\r\n]*)'
    rows=[]
    for status,z,w,t,k,details in re.findall(pattern,log_text):
        reasons=ast.literal_eval(details) if status=='Discarded' else {'exception':details}
        rows.append({'rejection_id':len(rows),'zeta':float(z),'wn':float(w),'tau':float(t),'K':float(k),
                     'status':status,'original_reason':'+'.join(sorted(reasons)),
                     'original_details':json.dumps(reasons,sort_keys=True)})
    return pd.DataFrame(rows)


def validate(task):
    name,index,parameters,gains,weight=task
    plant=Plant(*parameters);vector=np.array(gains)
    strict=assess_controller(plant,vector,weight)
    result={'model':name,'rejection_id':index,**strict,
            **{f'predicted_{key}':float(value) for key,value in zip(['Kp','Ki','Kd'],vector)},
            'extended_attempted':False,'extended_accepted':strict['accepted'],
            'extended_reason':strict['reason'],'extended_itae':strict['itae'],
            'extended_objective':strict['objective'],'extended_overshoot_pct':strict['overshoot_pct'] if strict['accepted'] else None}
    if strict['reason']=='resolution_limit':
        result['extended_attempted']=True
        sim=PlantSimulator();opt=PIDOptimizer(sim,gain_penalty=weight,max_validation_points=262145)
        metrics,reason=opt._validate_candidate(plant,PIDGains(*vector),sim.get_dynamic_t_eval(plant))
        result.update(extended_accepted=metrics is not None,extended_reason=reason)
        if metrics is not None:
            result.update(extended_itae=metrics['itae'],extended_overshoot_pct=metrics['overshoot_pct'],
                          extended_objective=opt.objective_terms(plant,vector,metrics['itae'])['objective'])
    return result
