"""Empirical-input MLP with optional matching-model stability regularization.

Physical parameters are used ONLY by the training loss and controller evaluator.
Inference requires the same five measured features as the other baselines.
"""
import copy
import itertools
import random
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURES = ['t_rise','t_settling','y_peak','y_ss','tau_est']


def empirical_factors(X):
    """q = [y_ss*Kp, y_ss*t_rise*Ki, y_ss*Kd/t_rise]."""
    rise,final=X[:,0],X[:,3]
    if torch.any(rise<=0) or torch.any(final<=0):
        raise ValueError('Rise time and final output must be positive.')
    return torch.stack((final,final*rise,final/rise),dim=1)


def characteristic_coefficients(parameters,gains):
    """Coefficients in z=s/wn, descending powers, for second-order Padé.

    Parameter order is zeta, wn, tau, K. Exact zero delay is handled separately
    in stability_margins because its polynomial degree is three, not five.
    """
    zeta,wn,tau,K=parameters.double().unbind(1)
    kp,ki,kd=gains.double().unbind(1)
    d=tau*wn
    qp,qi,qd=K*kp,K*ki/wn,K*wn*kd
    return torch.stack((d*d/12,
        d/2+zeta*d*d/6+qd*d*d/12,
        1+zeta*d+d*d/12+qp*d*d/12-qd*d/2,
        2*zeta+d/2+qi*d*d/12-qp*d/2+qd,
        1+qp-qi*d/2,qi),dim=1)


@lru_cache(maxsize=4)
def determinant_expansion(size):
    permutations=list(itertools.permutations(range(size)))
    signs=[(-1 if sum(p[i]>p[j] for i in range(size) for j in range(i+1,size))%2 else 1) for p in permutations]
    return permutations,signs


def relative_determinant(matrix):
    """Dimensionless signed determinant / sum of absolute expansion terms.

    Positive row scaling leaves the sign unchanged. Dividing by the expansion
    magnitude avoids comparing raw determinants with different units/scales.
    Orders here are at most four, so a short exact expansion is inexpensive.
    """
    size=matrix.shape[1]
    permutations,signs=determinant_expansion(size)
    columns=torch.tensor(permutations,device=matrix.device)
    rows=torch.arange(size,device=matrix.device).unsqueeze(0)
    terms=matrix[:,rows,columns].prod(-1)*torch.tensor(signs,device=matrix.device)
    return terms.sum(1)/terms.abs().sum(1).clamp_min(1e-30)


def stability_margins(parameters,gains):
    """Scaled Hurwitz conditions, with exact-delay-zero cubic branching.

    A nonnegative soft loss is not a stability certificate; saved predictions
    still undergo the existing pole and time-domain validation.
    """
    coefficients=characteristic_coefficients(parameters,gains)
    results=[]
    # Match PlantSimulator's tiny-delay cutoff exactly.
    delayed=parameters[:,2]>1e-5
    for mask,degree in [(delayed,5),(~delayed,3)]:
        if not bool(mask.any()):continue
        if degree==5:
            c=coefficients[mask]
        else:
            zeta,wn,_,K=parameters[mask].double().unbind(1)
            kp,ki,kd=gains[mask].double().unbind(1)
            c=torch.stack((torch.ones_like(wn),2*zeta+K*wn*kd,
                           1+K*kp,K*ki/wn),dim=1)
        # Necessary coefficient positivity plus all leading Hurwitz minors.
        margins=[c[:,1:]/c[:,1:].abs().sum(1,keepdim=True).clamp_min(1e-30)]
        rows=[]
        for row in range(degree):
            entries=[]
            for col in range(degree):
                index=2*col-row+1
                entries.append(c[:,index] if 0<=index<=degree else torch.zeros_like(c[:,0]))
            rows.append(torch.stack(entries,dim=1))
        hurwitz=torch.stack(rows,dim=1)
        # Final determinant equals a0 times the previous; a0 positivity is above.
        for order in range(1,degree):
            margins.append(relative_determinant(hurwitz[:,:order,:order]).unsqueeze(1))
        results.append((mask,torch.cat(margins,dim=1)))
    return results


def stability_loss(parameters,gains,margin=0.001):
    total=gains.sum()*0
    for mask,margins in stability_margins(parameters,gains):
        total=total+F.relu(margin-margins).square().mean(1).sum()
    return total/len(gains)


class MLPModule(nn.Module):
    def __init__(self,hidden_dims=(128,64,32),dropout=0.05):
        super().__init__();layers=[];previous=5
        for width in hidden_dims:
            layers.extend([nn.Linear(previous,width),nn.SiLU(),nn.Dropout(dropout)])
            previous=width
        layers.append(nn.Linear(previous,3));self.net=nn.Sequential(*layers)

    def forward(self,x):
        # Positive q avoids invalid logarithms; upper bounds remain checked later.
        return F.softplus(self.net(x))


class MLPModel:
    def __init__(self,hidden_dims=(128,64,32),dropout=.05,weight_decay=1e-4,
                 beta=0.,lr=.001,max_epochs=500,patience=50,batch_size=64,
                 seed=42,device='cpu',delta=.001,margin=.001,verbose=True):
        self.options=dict(hidden_dims=tuple(hidden_dims),dropout=dropout,weight_decay=weight_decay,
            beta=beta,lr=lr,max_epochs=max_epochs,patience=patience,batch_size=batch_size,
            seed=seed,device=device,delta=delta,margin=margin,verbose=verbose)
        self.module=None

    def fit(self,X,y,physical=None,validation_data=None):
        opt=self.options
        random.seed(opt['seed']);np.random.seed(opt['seed']);torch.manual_seed(opt['seed'])
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        if opt['beta']<0:raise ValueError('Stability weight beta must be nonnegative.')
        if opt['beta']>0 and physical is None:raise ValueError('Physics loss needs training plant metadata.')
        array=np.asarray(X,dtype=np.float32);targets=np.asarray(y,dtype=np.float32)
        if array.ndim!=2 or array.shape[1]!=5 or targets.shape!=(len(array),3):
            raise ValueError('Expected five inputs and three target gains.')
        self.mean=array.mean(0);self.std=array.std(0);self.std[self.std<1e-8]=1
        device=torch.device(opt['device'])
        raw=torch.tensor(array,device=device);truth=torch.tensor(targets,device=device)
        q=truth*empirical_factors(raw)
        self.qscale=q.median(0).values.clamp_min(.001).detach().cpu().numpy()
        self.logstd=torch.log(q+opt['delta']).std(0,unbiased=False).clamp_min(.1).detach().cpu().numpy()
        self.module=MLPModule(opt['hidden_dims'],opt['dropout']).to(device)
        scale=torch.tensor(self.qscale,device=device);spread=torch.tensor(self.logstd,device=device)
        x=torch.tensor((array-self.mean)/self.std,device=device)
        parameters=torch.tensor(np.asarray(physical),dtype=torch.float64,device=device) if physical is not None else None
        optimizer=torch.optim.AdamW(self.module.parameters(),lr=opt['lr'],weight_decay=opt['weight_decay'])
        def losses(inputs,unscaled,target,params):
            predicted_q=self.module(inputs)*scale
            target_q=target*empirical_factors(unscaled)
            supervised=((torch.log(predicted_q+opt['delta'])-torch.log(target_q+opt['delta']))/spread).square().mean()
            physics=stability_loss(params,predicted_q/empirical_factors(unscaled),opt['margin']) if opt['beta']>0 else supervised*0
            return supervised+opt['beta']*physics,supervised,physics
        if validation_data is not None:
            vx,vy,vp=validation_data;vraw=torch.tensor(np.asarray(vx,dtype=np.float32),device=device)
            if opt['beta']>0 and vp is None:
                raise ValueError('Physics loss needs validation plant metadata too.')
            vinput=torch.tensor((np.asarray(vx,dtype=np.float32)-self.mean)/self.std,device=device)
            vtruth=torch.tensor(np.asarray(vy,dtype=np.float32),device=device)
            vparams=torch.tensor(np.asarray(vp),dtype=torch.float64,device=device) if vp is not None else None
        self.history=[];best=np.inf;stale=0;state=None;self.best_epoch=0
        for epoch in range(1,opt['max_epochs']+1):
            self.module.train();order=torch.randperm(len(x),device=device)
            for batch in order.split(opt['batch_size']):
                optimizer.zero_grad()
                total,_,_=losses(x[batch],raw[batch],truth[batch],parameters[batch] if parameters is not None else None)
                if not torch.isfinite(total):raise FloatingPointError('Nonfinite MLP training loss.')
                # Clip neural gradients for training stability, never the predicted PID gains.
                total.backward();nn.utils.clip_grad_norm_(self.module.parameters(),10.);optimizer.step()
            self.module.eval()
            with torch.no_grad():
                train_loss,train_sup,train_phys=losses(x,raw,truth,parameters)
                val_loss,val_sup,val_phys=losses(vinput,vraw,vtruth,vparams) if validation_data is not None else (train_loss,train_sup,train_phys)
            score=float(val_loss)
            self.history.append({'epoch':epoch,'train_loss':float(train_loss),'val_loss':score,
                'train_supervised':float(train_sup),'val_supervised':float(val_sup),
                'train_stability':float(train_phys),'val_stability':float(val_phys)})
            if opt['verbose'] and epoch%25==0:
                print(f'MLP epoch {epoch}: train loss={float(train_loss):.5f}, validation loss={score:.5f}',flush=True)
            if score<best-1e-6:
                best=score;state=copy.deepcopy(self.module.state_dict());self.best_epoch=epoch;stale=0
            else:stale+=1
            if validation_data is not None and stale>=opt['patience']:break
        self.module.load_state_dict(state);self.module.eval()
        return self

    def predict(self,X):
        if self.module is None:raise RuntimeError('Fit MLP before prediction.')
        device=next(self.module.parameters()).device
        raw=torch.tensor(np.asarray(X,dtype=np.float32),device=device)
        with torch.no_grad():
            x=(raw-torch.tensor(self.mean,device=device))/torch.tensor(self.std,device=device)
            return (self.module(x)*torch.tensor(self.qscale,device=device)/empirical_factors(raw)).cpu().numpy()

    def save(self,path,metadata=None):
        torch.save({'options':{**self.options,'device':'cpu'},'state':{k:v.cpu() for k,v in self.module.state_dict().items()},
            'mean':self.mean.tolist(),'std':self.std.tolist(),'qscale':self.qscale.tolist(),
            'logstd':self.logstd.tolist(),'best_epoch':self.best_epoch,'features':FEATURES,
            'metadata':json.loads(json.dumps(metadata or {}))},Path(path))

    @classmethod
    def load(cls,path):
        bundle=torch.load(path,map_location='cpu',weights_only=True)
        model=cls(**bundle['options']);model.module=MLPModule(model.options['hidden_dims'],model.options['dropout'])
        model.module.load_state_dict(bundle['state']);model.module.eval()
        for key in ['mean','std','qscale','logstd']:setattr(model,key,np.array(bundle[key],dtype=np.float32))
        model.best_epoch=bundle['best_epoch'];return model
