/* Background Python workers: tuning and independent prediction verification. */

// Serialized into a module worker so this also works when index.html is opened
// with file://. Python packages are downloaded only when a sample is run.
function generationWorker() {
  let engine = null;
  self.onmessage = async ({data}) => {
    try {
      if (!engine) {
        self.postMessage({type:'engine', message:'Loading Python runtime…'});
        const {loadPyodide} = await import('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs');
        engine = await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'});
        engine.setStdout({batched: line => {
          if (line.startsWith('__PID_EVENT__')) self.postMessage(JSON.parse(line.slice(13)));
        }});
        self.postMessage({type:'engine', message:'Loading NumPy and SciPy…'});
        await engine.loadPackage(['numpy','scipy']);
        await engine.runPythonAsync(data.source);
      }
      engine.globals.set('parameters_json', JSON.stringify(data.parameters));
      const result = await engine.runPythonAsync(`
import json, sys, scipy

def emit(event):
    print('__PID_EVENT__' + json.dumps(event), flush=True)

def series(t, y):
    indices = np.unique(np.r_[np.linspace(0, len(t)-1, min(1500,len(t)), dtype=int), len(t)-1])
    return [[float(t[i]), float(y[i])] for i in indices]

p = json.loads(parameters_json)
plant = Plant(zeta=p['zeta'], wn=p['wn'], tau=p['tau'], K=p['K'])
simulator = PlantSimulator()
times = simulator.get_dynamic_t_eval(plant)
t_open, y_open = simulator.observe_open_loop(plant)
features = FeatureExtractor().extract(t_open, y_open)
emit({'type':'features', 'features':features, 'duration':float(t_open[-1]), 'n_points':len(t_open),
      'points':series(t_open,y_open)})

active_start = -1
original_minimize = spo.minimize

class WalkthroughOptimizer(PIDOptimizer):
    def _get_initial_guesses(self, plant, t_eval=None):
        emit({'type':'screening'})
        selected = super()._get_initial_guesses(plant, t_eval)
        emit({'type':'starts', 'starts':[x.tolist() for x in selected]})
        return selected

    def _validate_candidate(self, plant, gains, t_eval):
        metrics, reason = super()._validate_candidate(plant, gains, t_eval)
        emit({'type':'validation', 'start':active_start, 'reason':reason, 'metrics':metrics})
        return metrics, reason

optimizer = WalkthroughOptimizer(simulator)

def traced_minimize(*args, **kwargs):
    global active_start
    active_start += 1
    emit({'type':'start', 'start':active_start})
    iterations = [0]
    def callback(x):
        iterations[0] += 1
        if iterations[0] % 5 == 0:
            emit({'type':'iteration', 'start':active_start, 'iteration':iterations[0]})
    kwargs['callback'] = callback
    result = original_minimize(*args, **kwargs)
    emit({'type':'solver', 'start':active_start, 'success':bool(result.success),
          'iterations':int(result.nit), 'message':str(result.message),
          'gains':result.x.tolist() if np.all(np.isfinite(result.x)) else None})
    return result

try:
    spo.minimize = traced_minimize
    gains = optimizer.optimize(plant, times)
finally:
    spo.minimize = original_minimize
starts = optimizer.initial_guesses

payload = {'accepted':gains is not None, 'diagnostics':optimizer.diagnostics,
           'python':sys.version.split()[0], 'scipy':scipy.__version__}
if gains is not None:
    metrics = optimizer.last_validation
    t_final = np.linspace(0, metrics['duration'], int(metrics['n_points']))
    _, y_final = simulator.close_loop(plant, gains, t_final)
    _, y_comparison_open = simulator.open_loop(plant, t_final)
    best = min((record for record in optimizer.diagnostics if record['validation'] == 'accepted'),
               key=lambda record: record['objective'])
    initial_start = best['start']
    _, y_initial = simulator.close_loop(plant, PIDGains(*starts[initial_start]), t_final)
    initial = series(t_final,y_initial) if np.all(np.isfinite(y_initial)) and np.max(abs(y_initial))<20 else None
    row = {'zeta':plant.zeta, 'wn':plant.wn, 'tau':plant.tau, 'K':plant.K,
           'observation_duration':float(t_open[-1]),
           **features, 'Kp':gains.Kp, 'Ki':gains.Ki, 'Kd':gains.Kd}
    payload.update(row=row, metrics=metrics, points=series(t_final,y_final),
                   initial=initial, initial_start=initial_start,
                   open_loop=series(t_final,y_comparison_open))
json.dumps(payload)
`);
      self.postMessage({type:'complete', result:JSON.parse(result)});
    } catch (error) {
      self.postMessage({type:'error', message:String(error.message || error)});
    }
  };
}


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
        await engine.runPythonAsync(data.source+'\n\n'+data.assessor+'\nfrom scipy.optimize import brentq\n'+(data.diagnostic||''));
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
if metrics['stable'] and metrics['reason'] == 'overshoot' and 'diagnostic' in globals():
    detail = diagnostic(('live', 0, [plant.zeta,plant.wn,plant.tau,plant.K], request['gains'], False))
    if detail['response_resolved']:
        metrics.update(response_resolved=True, diagnostic_itae=detail['itae'],
            diagnostic_objective=detail['objective'], diagnostic_settling=detail['settling_time_2pct'],
            overshoot_pct=detail['overshoot_pct'])
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
