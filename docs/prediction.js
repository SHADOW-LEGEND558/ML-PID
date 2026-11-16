/* User-entered measurements → frozen MLP → optional independent plant checks. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id), form=$('prediction-form');
  if (!form || !window.PID_MODEL) return;
  const features=window.PID_MODEL.features, physical=['K','wn','zeta','tau'];
  let worker=null, busy=false, current=null, live=null;
  const fmt=(v,d=4)=>Number.isFinite(v)?Number(v.toPrecision(d)).toString():'—';
  const field=k=>form.elements.namedItem(k);
  const knowsPlant=()=>$('prediction-mode').value==='plant'||$('prediction-known').checked;
  const status=text=>{$('prediction-status').textContent=text;};
  function configure() {
    const byPlant=$('prediction-mode').value==='plant', known=knowsPlant();
    $('prediction-feature-fields').hidden=byPlant;
    $('prediction-known-label').hidden=byPlant;
    $('prediction-plant-fields').hidden=!known;
    $('prediction-compare-label').hidden=!known;
    features.forEach(k=>{field(k).required=!byPlant;field(k).disabled=busy||byPlant;});
    physical.forEach(k=>{field(k).required=known;field(k).disabled=busy||!known;});
    ['prediction-mode','prediction-example','prediction-known','prediction-run'].forEach(id=>{$(id).disabled=busy;});
    $('prediction-compare').disabled=busy||!known;
    $('prediction-cancel').hidden=!busy;
    $('prediction-run').textContent=busy?'Working…':'Predict PID gains →';
  }
  function invalidate() {
    live=null; current=null;
    $('prediction-response').hidden=true;
    $('prediction-gains').replaceChildren();
    $('prediction-features-note').textContent='';
    status('Inputs changed. Run the model to obtain a new prediction.');
  }
  function preset(index) {
    const item=window.PID_STUDY.cases[index];
    if(!item) return;
    features.forEach(k=>{field(k).value=fmt(item.features[k],9);});
    physical.forEach(k=>{field(k).value=fmt(item.plant[k],9);});
    invalidate();
    status('Example values loaded. Edit them or predict using the frozen final MLP.');
  }
  function failure(error) {
    busy=false;configure();
    status(`The prediction or verification could not finish: ${error}. No controller is certified by this run.`);
    worker?.terminate();worker=null;
  }
  function getWorker() {
    if(worker) return worker;
    const url=URL.createObjectURL(new Blob([`(${predictionWorker.toString()})();`],{type:'text/javascript'}));
    worker=new Worker(url,{type:'module'});URL.revokeObjectURL(url);
    worker.onerror=e=>failure(e.message||'Browser worker unavailable; serve this site over HTTP.');
    worker.onmessage=({data})=>{
      if(!busy || !current) return;
      if(data.type==='status')status(data.message);
      else if(data.type==='error')failure(data.message);
      else if(data.type==='features') {
        current.features=data.result.features;
        features.forEach(k=>{field(k).value=fmt(current.features[k],9);});
        runNative();
        verify();
      } else if(data.type==='complete') finish(data.result);
    };
    return worker;
  }
  function send(action) {
    getWorker().postMessage({action,source:window.PID_GENERATOR.source,assessor:window.PID_MODEL.assessorSource,
      plant:current.plant,features:current.features,gains:current.gains,compare:current.compare});
  }
  function runNative() {
    const start=performance.now();
    current.gains=window.PIDPredict(current.features,window.PID_MODEL);
    current.inferenceMs=performance.now()-start;
    if(current.gains.some(v=>!Number.isFinite(v)))throw new Error('Nonfinite prediction; use finite inputs within the study scope.');
    $('prediction-gains').innerHTML=current.gains.map((v,i)=>`<div><strong>${fmt(v,6)}</strong><span>K<sub>${['p','i','d'][i]}</sub></span></div>`).join('');
    const outside=features.filter(k=>{const [a,b]=window.PID_MODEL.featureRanges[k];return current.features[k]<a||current.features[k]>b;});
    current.warning=outside.length?' Some feature values lie outside the training range: this is extrapolation.':'';
    $('prediction-features-note').textContent=`Local JavaScript inference: ${fmt(current.inferenceMs)} ms. Used rise ${fmt(current.features.t_rise)} s, settling ${fmt(current.features.t_settling)} s, peak ${fmt(current.features.y_peak)}, final ${fmt(current.features.y_ss)}, onset ${fmt(current.features.tau_est)} s.${current.warning}`;
  }
  function verify() {
    status('Gains predicted. Independently checking the known physical plant…');
    send('verify');
  }
  const outcome=m=>m.accepted?'Accepted':({overshoot:'Rejected: overshoot >10%',invalid_gains:'Rejected: gain bounds',unstable:'Rejected: unstable loop',resolution_limit:'Unresolved: grid cap',duration_limit:'Unresolved: horizon cap'}[m.reason]||`Not accepted: ${m.reason||'no validated numerical result'}`);
  function finish(result) {
    busy=false;configure();
    const m=result.metrics,r=result.referenceMetrics;
    $('prediction-response').hidden=false;
    const rows=[['Predicted MLP',m],...(r?[['Numerical search',r]]:[])];
    $('prediction-metrics').innerHTML=`<div class="study-table-wrap"><table class="study-table"><caption>Current plant, unit setpoint, independent checks</caption><thead><tr><th>Controller</th><th>Outcome</th><th>Overshoot</th><th>Validated ITAE</th><th>Validated J</th></tr></thead><tbody>${rows.map(([name,v])=>`<tr><td>${name}</td><td>${outcome(v)}</td><td>${v.overshoot_pct==null?'—':fmt(v.overshoot_pct)+'%'}</td><td>${v.accepted?fmt(v.itae,6):'—'}</td><td>${v.accepted?fmt(v.objective,6):'—'}</td></tr>`).join('')}</tbody></table></div>`;
    if(!m.accepted && m.display_itae!=null) {
      const note=document.createElement('p');note.className='chart-caption';
      note.textContent=`Illustration only: ITAE over the plotted ${fmt(result.t.at(-1))} s is ${fmt(m.display_itae,6)}. This is not a converged accepted metric; displayed overshoot on a rejected controller is diagnostic.`;
      $('prediction-metrics').append(note);
    }
    if(r?.accepted && m.accepted) {
      const note=document.createElement('p');note.className='chart-caption';
      note.textContent=`Accepted-pair comparison: ITAE ${fmt(m.itae/r.itae)}× and J ${fmt(m.objective/r.objective)}× the best-found numerical result. Numerical gains: ${r.gains.map(v=>fmt(v,6)).join(' / ')}. These searches are local, not certified global optima.`;
      $('prediction-metrics').append(note);
    }
    const note=document.createElement('p');note.className='chart-caption';
    note.textContent=`JavaScript inference ${fmt(current.inferenceMs)} ms; independent verification ${fmt(result.validationMs)} ms${result.optimizationMs==null?'':`; numerical search including its validation ${fmt(result.optimizationMs)} ms`}. Runtime loading, observation and display plotting are excluded from these timings.`;
    $('prediction-metrics').append(note);
    live={plant:current.plant,name:'Live prediction',t:result.t,open:result.open,reference:result.reference,
      models:{live:{y:result.y,reason:m.reason}}};
    window.PID_RENDER_RESPONSE('prediction-chart',live);
    status(`${outcome(m)}.${current.compare?` Numerical comparison: ${r?.accepted?'accepted reference found':'no validated reference found'}.`:''}${current.warning} Curves are illustrative; the table uses independent validation.`);
  }
  form.addEventListener('submit',event=>{
    event.preventDefault();if(busy)return;
    try {
      if(!form.reportValidity())return;
      current={plant:knowsPlant()?Object.fromEntries(physical.map(k=>[k,field(k).valueAsNumber])):null,
        features:Object.fromEntries(features.map(k=>[k,field(k).valueAsNumber])),compare:knowsPlant()&&$('prediction-compare').checked};
      busy=true;configure();
      if($('prediction-mode').value==='plant') {status('Measuring the open-loop response to derive five features…');send('observe');}
      else {runNative();if(current.plant)verify();else{busy=false;configure();status(`Gains predicted locally. Supply a known plant to simulate and verify the response.${current.warning}`);}}
    } catch(error) {failure(error.message||String(error));}
  });
  $('prediction-cancel').addEventListener('click',()=>{worker?.terminate();worker=null;busy=false;configure();invalidate();status('Run canceled. No new controller was accepted.');});
  ['prediction-mode','prediction-known'].forEach(id=>$(id).addEventListener('change',()=>{invalidate();configure();}));
  form.querySelectorAll('input[type=number],#prediction-compare').forEach(input=>input.addEventListener('input',invalidate));
  window.PID_STUDY.cases.forEach((c,i)=>$('prediction-example').add(new Option(c.name,String(i))));
  $('prediction-example').addEventListener('change',()=>{if($('prediction-example').value!=='')preset(Number($('prediction-example').value));});
  window.addEventListener('resize',()=>{if(live)window.PID_RENDER_RESPONSE('prediction-chart',live);});
  preset(0);$('prediction-example').value='0';configure();
})();
