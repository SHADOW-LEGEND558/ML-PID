/* Physical verification uses the unchanged generator and evaluator in Pyodide. */
function predictionWorker() {
  let engine;
  self.onmessage = async ({data}) => {
    try {
      if (!engine) {
        self.postMessage({type:'status',message:'Loading the local simulation engine (first run)…'});
        const {loadPyodide} = await import('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs');
        engine = await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'});
        await engine.loadPackage(['numpy','scipy']);
        await engine.runPythonAsync(data.source+'\n\n'+data.assessor);
      }
      engine.globals.set('request_json',JSON.stringify(data));
      if (data.action === 'observe') {
        const result = await engine.runPythonAsync(`
import json
p = json.loads(request_json)['plant']
plant = Plant(zeta=p['zeta'], wn=p['wn'], tau=p['tau'], K=p['K'])
simulator = PlantSimulator()
t, y = simulator.observe_open_loop(plant)
json.dumps({'features':FeatureExtractor().extract(t,y), 'observation_duration':float(t[-1])})
`);
        self.postMessage({type:'features',result:JSON.parse(result)});
        return;
      }
      self.postMessage({type:'status',message:'Checking predicted gains, poles, overshoot and numerical convergence…'});
      const result = await engine.runPythonAsync(`
import json, time
request = json.loads(request_json)
p = request['plant']
plant = Plant(zeta=p['zeta'], wn=p['wn'], tau=p['tau'], K=p['K'])
simulator = PlantSimulator()
gains = PIDGains(*request['gains'])
started = time.perf_counter()
metrics = assess_controller(plant, np.asarray(request['gains']), .01)
validation_ms = 1000*(time.perf_counter()-started)
duration = min(100.,max(3.,1.4*request['features']['t_settling']))
t = np.linspace(0,duration,1601)
_, opened = simulator.open_loop(plant,t)
y = None
if metrics['stable']:
    _, response = simulator.close_loop(plant,gains,t)
    if np.isfinite(response).all() and np.max(np.abs(response)) < 20:
        y = response.tolist()
        metrics['display_itae'] = float(spi.simpson(t*np.abs(1-response),x=t))
reference = None
reference_points = None
optimization_ms = None
if request['compare']:
    optimizer = PIDOptimizer(simulator,n_starts=4,gain_penalty=.01)
    started = time.perf_counter()
    tuned = optimizer.optimize(plant,simulator.get_dynamic_t_eval(plant))
    optimization_ms = 1000*(time.perf_counter()-started)
    if tuned is not None:
        reference = {'accepted':True, 'gains':[tuned.Kp,tuned.Ki,tuned.Kd], **optimizer.last_validation}
        _, ref_y = simulator.close_loop(plant,tuned,t)
        reference_points = ref_y.tolist()
    else:
        reference = {'accepted':False,'reason':'No independently validated numerical result'}
json.dumps({'metrics':metrics,'referenceMetrics':reference,'t':t.tolist(), 'open':opened.tolist(),
            'reference':reference_points,'y':y,'validationMs':validation_ms,'optimizationMs':optimization_ms},allow_nan=False)
`);
      self.postMessage({type:'complete',result:JSON.parse(result)});
    } catch(error) {self.postMessage({type:'error',message:String(error.message || error)});}
  };
}
