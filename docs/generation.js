/* Single-plant walkthrough. Tuning and validation execute exported Python,
   while previews and presentation stay on the main browser thread. */
function generationFocusWindow(curves, duration) {
  let lastTransient = 0;
  for (const {points, target} of curves) {
    const band = 0.02 * Math.max(Math.abs(target), 1e-6);
    let lastOutside = -1;
    points.forEach((point, i) => {
      if (Math.abs(point[1] - target) > band) lastOutside = i;
    });
    // Don't crop a curve that has not settled or one with late excursions.
    if (lastOutside === points.length - 1) return duration;
    if (lastOutside >= 0) lastTransient = Math.max(lastTransient, points[lastOutside + 1][0]);
  }
  return Math.min(duration, Math.max(2, lastTransient * 1.25));
}

function generationWindowPoints(points, end) {
  const visible = points.filter(point => point[0] <= end);
  const next = points.find(point => point[0] > end);
  const last = visible[visible.length - 1];
  if (last && next && last[0] < end) {
    visible.push([end, last[1] + (next[1] - last[1]) * (end - last[0]) / (next[0] - last[0])]);
  }
  return visible;
}

(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const names = ['K','wn','zeta','tau'];
  const sliders = names.map(name => byId(`sample-${name}`));
  let worker = null, workerURL = null, busy = false, acceptedRow = null;
  let comparisonResult = null, showFullWindow = false;
  let selectedStartCount = 4;
  const stages = [...document.querySelectorAll('.generation-stages li')];
  const format = value => Math.abs(value) > 0 && Math.abs(value) < 0.001 ?
    value.toExponential(3) : Number(value).toFixed(4);

  function stage(index, state, text) {
    stages[index].dataset.state = state;
    stages[index].querySelector('.stage-state').textContent = text;
  }
  function status(title, detail, state='') {
    byId('sample-status').textContent = title;
    byId('sample-status-detail').textContent = detail;
    byId('engine-dot').className = `engine-dot ${state}`;
  }
  function parameters() {
    return Object.fromEntries(names.map((name,i) => [name,Number(sliders[i].value)]));
  }
  function chart(id, series, duration, emptyText='') {
    const svg = byId(id);
    svg.replaceChildren();
    if (emptyText) {
      svg.append(element('text',{x:360,y:150,'text-anchor':'middle',class:'tick'},emptyText));
      return;
    }
    const values = series.flatMap(item => item.points.map(p => p[1]));
    const low = Math.min(0,...values), high = Math.max(1,...values);
    const margin = (high-low)*0.12;
    const ymin = low < 0 ? low-margin : 0, ymax = high+margin;
    const x = t => 62+t/duration*642;
    const y = v => 269-(v-ymin)/(ymax-ymin)*254;
    for(let i=0;i<=4;i++) {
      const value=ymin+(ymax-ymin)*i/4;
      svg.append(element('line',{x1:62,x2:704,y1:y(value),y2:y(value),class:'grid-line'}));
      svg.append(element('text',{x:50,y:y(value)+4,'text-anchor':'end',class:'tick'},value.toFixed(2)));
    }
    for(let i=0;i<=5;i++) {
      const time=duration*i/5;
      svg.append(element('line',{x1:x(time),x2:x(time),y1:15,y2:269,class:'grid-line'}));
      svg.append(element('text',{x:x(time),y:289,'text-anchor':'middle',class:'tick'},time.toFixed(1)));
    }
    svg.append(element('path',{d:`M62,${y(1)} L704,${y(1)}`,class:'reference-line'}));
    for(const item of series) svg.append(element('path',{
      d:item.points.map(([t,v],i)=>`${i?'L':'M'}${x(t).toFixed(2)},${y(v).toFixed(2)}`).join(' '),
      class:item.className
    }));
    svg.append(element('text',{x:390,y:312,'text-anchor':'middle',class:'axis-title'},'TIME / SECONDS'));
    svg.append(element('text',{x:15,y:146,'text-anchor':'middle',transform:'rotate(-90 15 146)',class:'axis-title'},'OUTPUT'));
  }
  function resetResults() {
    acceptedRow = null;
    comparisonResult = null; showFullWindow = false;
    byId('sample-window-toggle').hidden = true;
    byId('sample-download').disabled = true;
    document.querySelectorAll('[data-feature]').forEach(node => {node.textContent='—';});
    byId('feature-window').textContent='AWAITING EXTRACTION';
    byId('sample-starts').replaceChildren();
    const tr=document.createElement('tr'),td=document.createElement('td');
    td.colSpan=5;td.className='empty-cell';td.textContent='Run a sample to see the starting gains and search results.';
    tr.append(td);byId('sample-starts').append(tr);
    for(const id of ['sample-stability','sample-overshoot','sample-itae','sample-convergence']) byId(id).textContent='—';
    byId('sample-row').replaceChildren();
    const note=document.createElement('p');note.className='empty-cell';
    note.textContent='Each accepted row contains 4 plant parameters, 6 response features, and 3 PID target gains.';
    byId('sample-row').append(note);
    byId('sample-response-status').textContent='AWAITING RESULT';
    byId('sample-initial-legend').hidden=true;
    byId('sample-closed-chart').setAttribute('aria-label','Awaiting validated controller comparison');
    byId('sample-response-note').textContent='After tuning, compare the plant alone, the selected search’s initial controller, and its final controller.';
    chart('sample-closed-chart',[],1,'Run the pipeline to see the tuned response.');
    stage(0,'complete','Ready');
    for(let i=1;i<4;i++)stage(i,'','Waiting');
  }
  function preview() {
    const p=parameters();
    sliders.forEach(control => {
      byId(`${control.id}-value`).value=Number(control.value).toFixed(2);
      control.style.setProperty('--fill',`${100*(control.value-control.min)/(control.max-control.min)}%`);
    });
    const rate=p.zeta<=1?p.zeta*p.wn:p.wn/(p.zeta+Math.sqrt(p.zeta*p.zeta-1));
    const duration=Math.max(10,12/rate+3*p.tau);
    const damping=p.zeta>=0.999&&p.zeta<=1.001?1:p.zeta;
    const points=Array.from({length:1000},(_,i)=>{
      const t=duration*i/999;return [t,plantResponse(t,p.K,p.wn,damping,p.tau)];
    });
    chart('sample-open-chart',[{points,className:'response-line'}],duration);
    byId('sample-equation').textContent=
      `G(s) = ${format(p.K*p.wn*p.wn)} / (s² + ${format(2*p.zeta*p.wn)}s + ${format(p.wn*p.wn)}) × exp(−${p.tau.toFixed(2)}s)`;
    byId('sample-regime').textContent=p.zeta<1?'UNDERDAMPED':p.zeta>1?'OVERDAMPED':'CRITICALLY DAMPED';
    resetResults();
    status('Plant ready. Run it through the pipeline.',
      'The selected parameters use the same ranges as data generation. Changing a plant clears its previous results.');
  }
  function setBusy(value) {
    busy=value;
    for(const control of sliders){control.disabled=value;byId(`${control.id}-value`).disabled=value;}
    byId('sample-run').disabled=value;
    byId('sample-random').disabled=value;
    byId('sample-cancel').hidden=!value;
    byId('sample-run').textContent=value?'Running…':'Run this sample →';
  }
  function destroyWorker() {
    if(worker){worker.onmessage=null;worker.onerror=null;worker.terminate();}
    if(workerURL)URL.revokeObjectURL(workerURL);
    worker=null;workerURL=null;
  }
  function startTable(starts) {
    byId('sample-starts').replaceChildren();
    if(!starts.length){
      const tr=document.createElement('tr'),td=document.createElement('td');
      td.colSpan=5;td.className='empty-cell';td.textContent='No starting controller passed the initial checks within the search limit.';
      tr.append(td);byId('sample-starts').append(tr);
    }
    starts.forEach((gains,i)=>{
      const tr=document.createElement('tr');tr.id=`sample-start-${i}`;
      for(const text of [`0${i+1}`,gains.map(format).join(' / '),'—','—','Queued']) {
        const td=document.createElement('td');td.textContent=text;tr.append(td);
      }
      byId('sample-starts').append(tr);
    });
  }
  const reasonLabel = reason => ({accepted:'Accepted',overshoot:'Overshoot > 10%',unstable:'Unstable model',
    duration_limit:'Simulation time limit reached',resolution_limit:'Sampling limit reached',tail_not_converged:'Long-term error unresolved',
    optimizer_failure:'Solver did not converge',simulation_failure:'Simulation failed',invalid_gains:'Invalid gains'}[reason] || reason);
  function outcomeRow(row) {
    const grid=document.createElement('div');grid.className='row-columns';
    for(const [key,value] of Object.entries(row)) {
      const cell=document.createElement('div');
      if(['Kp','Ki','Kd'].includes(key))cell.className='target';
      const label=document.createElement('span'),number=document.createElement('strong');
      label.textContent=key;number.textContent=format(value);number.title=String(value);
      cell.append(label,number);grid.append(cell);
    }
    byId('sample-row').replaceChildren(grid);
  }
  function renderComparison(result) {
    const m=result.metrics;
    const curves=[{points:result.open_loop,className:'plant-comparison-line',target:result.row.K}];
    if(result.initial)curves.push({points:result.initial,className:'open-line',target:1});
    curves.push({points:result.points,className:'response-line',target:1});
    const focus = generationFocusWindow(curves,m.duration);
    const end = showFullWindow ? m.duration : focus;
    chart('sample-closed-chart',curves.map(curve=>({
      ...curve,points:generationWindowPoints(curve.points,end)
    })),end);
    const toggle=byId('sample-window-toggle');
    toggle.hidden=focus>=m.duration;
    toggle.textContent=showFullWindow?'Focus on transient':'Show full window';
    toggle.setAttribute('aria-pressed',String(showFullWindow));
    const startNumber=result.initial_start+1;
    byId('sample-initial-label').textContent=`Initial guess (winning start ${startNumber})`;
    byId('sample-initial-legend').hidden=!result.initial;
    byId('sample-closed-chart').setAttribute('aria-label',
      `Open-loop plant and best validated PID${result.initial?`, with initial gains from winning start ${startNumber}`:''}. Showing 0 to ${end.toFixed(2)} seconds of a ${m.duration.toFixed(2)} second validation window.`);
    byId('sample-response-note').textContent=
      `Showing 0–${end.toFixed(2)} s · Metrics use the full ${m.duration.toFixed(2)} s validation window (${m.n_points.toLocaleString()} points). Kp ${format(result.row.Kp)}, Ki ${format(result.row.Ki)}, Kd ${format(result.row.Kd)}. ${result.initial?`Compared with initial gains from winning start ${startNumber}.`:`Initial response from winning start ${startNumber} omitted because it is nonfinite or exceeds the display limit.`} Curves are downsampled; overshoot uses refined peaks.`;
  }
  function handleMessage({data}) {
    const row=data.start===undefined?null:byId(`sample-start-${data.start}`);
    if(data.type==='engine') {
      status(data.message,'The first run loads the Python engine and scientific packages. Keep this tab open; you can cancel at any time.','busy');
    } else if(data.type==='features') {
      stage(1,'complete','Extracted');stage(2,'active','Starting searches');
      for(const node of document.querySelectorAll('[data-feature]')){
        const key=node.dataset.feature;
        node.textContent=format(data.features[key])+(['t_rise','t_settling','tau_est'].includes(key)?' s':'');
      }
      byId('feature-window').textContent=`1,000 POINTS / ${data.duration.toFixed(2)} s`;
      chart('sample-open-chart',[{points:data.points,className:'response-line'}],data.duration);
    } else if(data.type==='screening') {
      stage(2,'active','Screening starting gains');
      status('Finding feasible starting controllers…',
        'Trying formula-based gains, smaller gain values, and controllers without derivative action. Starts must have stable model poles and sampled overshoot at most 9.5%; final controllers are checked separately.','busy');
    } else if(data.type==='starts') {
      selectedStartCount=data.starts.length;
      startTable(data.starts);
    } else if(data.type==='start') {
      if(row)row.cells[4].textContent='Optimizing…';
      status(`Search ${data.start+1} of ${selectedStartCount} is running.`,
        'The optimizer adjusts the gains to reduce time-weighted tracking error while checking overshoot and model stability. A separate validation pass checks the resulting controller.','busy');
      stage(2,'active',`Search ${data.start+1} / ${selectedStartCount}`);
    } else if(data.type==='iteration') {
      if(row)row.cells[2].textContent=String(data.iteration);
    } else if(data.type==='solver') {
      if(row){
        row.cells[2].textContent=String(data.iterations);
        row.cells[4].textContent=data.success?'Validating…':'Solver did not converge';
        row.cells[4].title=data.message;
        if(!data.success)row.cells[4].className='rejected';
      }
    } else if(data.type==='validation') {
      if(row){
        row.cells[4].textContent=reasonLabel(data.reason);
        row.cells[4].className=data.reason==='accepted'?'accepted':'rejected';
        if(data.metrics){
          row.cells[3].textContent=data.metrics.itae.toPrecision(6);
          row.cells[3].title=String(data.metrics.itae);
        }
      }
    } else if(data.type==='complete') {
      setBusy(false);
      const result=data.result;
      stage(2,'complete',`${result.diagnostics.length} searches complete`);
      byId('sample-provenance').textContent=
        `Python ${result.python} · SciPy ${result.scipy} · Source ${window.PID_GENERATOR.sha256.slice(0,12)}. Results can vary slightly across library versions. The browser executes the exported src/data_gen.py classes; batch replacement sampling has a total attempt limit.`;
      if(!result.accepted){
        stage(3,'failed','No accepted controller');
        status('This plant did not produce a validated row.',
          'No result is exported. The batch generator would draw a replacement plant. Try another sample; this does not prove PID control is impossible.','error');
        byId('sample-response-status').textContent='REJECTED / NO ROW';
        chart('sample-closed-chart',[],1,'No candidate passed every validation check.');
        return;
      }
      acceptedRow=result.row;
      const m=result.metrics;
      stage(3,'complete','Accepted');
      status('One validated training example, ready.',
        'The selected controller has the lowest ITAE among those that passed validation. The chart focuses on the changing part of the response; metrics use the full simulation. Its search is highlighted in the table.','done');
      const best=result.diagnostics.filter(d=>d.validation==='accepted').reduce((a,b)=>a.itae<=b.itae?a:b);
      byId(`sample-start-${best.start}`).classList.add('best');
      comparisonResult=result;
      renderComparison(result);
      byId('sample-response-status').textContent='BEST VALIDATED PID';
      byId('sample-stability').textContent='Stable';
      byId('sample-stability').title=`Largest pole real part: ${m.max_pole_real}`;
      byId('sample-overshoot').textContent=`${m.overshoot_pct.toFixed(3)}%`;
      byId('sample-itae').textContent=format(m.itae);
      byId('sample-convergence').textContent='Passed';
      outcomeRow(result.row);
      byId('sample-download').disabled=false;
    } else if(data.type==='error') {
      fail(data.message);
    }
  }
  function fail(message) {
    destroyWorker();setBusy(false);
    stages.forEach((node,i)=>{if(node.dataset.state==='active')stage(i,'failed','Stopped');});
    stage(3,'failed','Not completed');
    status('The sample could not finish.',
      `Check your connection if the Python packages did not load, then retry. No row was accepted. ${message}`,'error');
  }
  for(const control of sliders) {
    const field=byId(`${control.id}-value`);
    control.addEventListener('input',preview);
    const commit=()=>{
      if(busy)return;
      if(field.value.trim()!==''&&Number.isFinite(field.valueAsNumber))
        control.value=snapParameter(field.valueAsNumber,Number(control.min),Number(control.max),0.01);
      preview();
    };
    field.addEventListener('change',commit);
    field.addEventListener('blur',()=>{if(field.value!==Number(control.value).toFixed(2))commit();});
    field.addEventListener('keydown',event=>{
      if(event.key==='Enter'){event.preventDefault();commit();field.blur();}
      if(event.key==='Escape'){field.value=Number(control.value).toFixed(2);field.blur();}
    });
  }
  byId('sample-random').addEventListener('click',()=>{
    const random=new Uint32Array(4);crypto.getRandomValues(random);
    sliders.forEach((control,i)=>{
      control.value=snapParameter(Number(control.min)+(Number(control.max)-Number(control.min))*random[i]/4294967296,
        Number(control.min),Number(control.max),0.01);
    });
    preview();
  });
  byId('sample-run').addEventListener('click',()=>{
    if(busy)return;
    resetResults();setBusy(true);stage(1,'active','Loading engine');
    status('Preparing the Python pipeline…','The interface remains responsive while a separate worker performs tuning.','busy');
    try {
      if(!worker){
        workerURL=URL.createObjectURL(new Blob([`(${generationWorker.toString()})()`],{type:'text/javascript'}));
        worker=new Worker(workerURL,{type:'module'});
        worker.onmessage=handleMessage;
        worker.onerror=event=>{event.preventDefault();fail(event.message || 'Browser worker unavailable.');};
      }
      worker.postMessage({parameters:parameters(),source:window.PID_GENERATOR.source});
    } catch(error){fail(error.message);}
  });
  byId('sample-cancel').addEventListener('click',()=>{
    destroyWorker();setBusy(false);
    stages.forEach((node,i)=>{if(node.dataset.state==='active')stage(i,'failed','Cancelled');});
    stage(3,'','Not collected');
    for(const row of byId('sample-starts').rows){
      if(row.cells.length===5&&['Queued','Optimizing…','Validating…'].includes(row.cells[4].textContent))
        row.cells[4].textContent='Cancelled';
    }
    status('Run cancelled. No row was collected.','You can adjust the plant and start again. Restarting after cancellation reloads the Python engine.');
  });
  byId('sample-download').addEventListener('click',()=>{
    if(!acceptedRow)return;
    const csv=Object.keys(acceptedRow).join(',')+'\n'+Object.values(acceptedRow).join(',')+'\n';
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download='pid_training_example.csv';link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  byId('sample-window-toggle').addEventListener('click',()=>{
    if(!comparisonResult)return;
    showFullWindow=!showFullWindow;
    renderComparison(comparisonResult);
  });
  window.addEventListener('pagehide',destroyWorker);
  preview();
})();
