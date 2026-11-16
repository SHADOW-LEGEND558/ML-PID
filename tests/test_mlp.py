import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from MLP import MLPModel,characteristic_coefficients,stability_margins,stability_loss,empirical_factors
from data_gen import Plant,PIDGains,PlantSimulator


class MLPTests(unittest.TestCase):
    def test_polynomial_matches_shared_second_order_pade(self):
        sim=PlantSimulator()
        for plant in [Plant(.7,2,.3,1.5),Plant(.1,10,.01,.5),Plant(2,.5,1,2)]:
            gains=PIDGains(.8,1.2,.1)
            p=torch.tensor([[plant.zeta,plant.wn,plant.tau,plant.K]],dtype=torch.float64)
            g=torch.tensor([[gains.Kp,gains.Ki,gains.Kd]],dtype=torch.float64)
            coefficients=characteristic_coefficients(p,g).numpy()[0]
            _,den=sim.closed_loop_coefficients(plant,gains)
            expected=den*plant.wn**np.arange(5,-1,-1)
            np.testing.assert_allclose(coefficients/coefficients[0],expected/expected[0],rtol=1e-10)

    def test_hurwitz_sign_agrees_with_poles_and_gradients_are_finite(self):
        rng=np.random.default_rng(123);sim=PlantSimulator()
        for tau in [0.,.001,.2,1.]:
            plants=[Plant(float(rng.uniform(.1,2)),float(rng.uniform(.5,10)),tau,float(rng.uniform(.5,2))) for _ in range(40)]
            gains=rng.uniform([.01,.01,0],[10,20,5],(40,3))
            params=torch.tensor([[p.zeta,p.wn,p.tau,p.K] for p in plants],dtype=torch.float64)
            tensor=torch.tensor(gains,dtype=torch.float64,requires_grad=True)
            masks=stability_margins(params,tensor)
            for mask,margins in masks:
                for j,idx in enumerate(np.where(mask.numpy())[0]):
                    _,den=sim.closed_loop_coefficients(plants[idx],PIDGains(*gains[idx]))
                    self.assertEqual(bool((margins[j]>0).all()),bool((np.roots(den).real<0).all()))
            loss=stability_loss(params,tensor);loss.backward()
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_fit_prediction_and_checkpoint_roundtrip(self):
        rng=np.random.default_rng(42);x=rng.uniform(.2,2,(20,5));y=rng.uniform(.05,1,(20,3))
        p=np.tile([.7,2,.2,1],(20,1))
        for beta in [0.,.1]:
            model=MLPModel(hidden_dims=(8,),max_epochs=3,patience=2,beta=beta)
            model.fit(x,y,p,validation_data=(x,y,p));prediction=model.predict(x)
            self.assertEqual(prediction.shape,(20,3));self.assertTrue((prediction>0).all())
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.pt';model.save(path,metadata={'score':np.float64(.5)})
                np.testing.assert_allclose(MLPModel.load(path).predict(x),prediction,rtol=1e-6)
        with self.assertRaises(RuntimeError):MLPModel().predict(x)
        with self.assertRaises(ValueError):empirical_factors(torch.zeros((2,5)))


if __name__=='__main__':unittest.main()
