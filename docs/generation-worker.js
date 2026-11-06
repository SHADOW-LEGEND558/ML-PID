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
t_open, y_open = simulator.open_loop(plant, times)
features = FeatureExtractor().extract(t_open, y_open)
emit({'type':'features', 'features':features, 'duration':float(times[-1]),
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
               key=lambda record: record['itae'])
    initial_start = best['start']
    _, y_initial = simulator.close_loop(plant, PIDGains(*starts[initial_start]), t_final)
    initial = series(t_final,y_initial) if np.all(np.isfinite(y_initial)) and np.max(abs(y_initial))<20 else None
    row = {'zeta':plant.zeta, 'wn':plant.wn, 'tau':plant.tau, 'K':plant.K,
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
