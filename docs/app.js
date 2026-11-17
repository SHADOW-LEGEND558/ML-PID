/* ===== Shared navigation and system-basics simulator ===== */
'use strict';

// Analytic SOPDT step response. This explorer does not run PID optimization.
function plantResponse(time, gain, frequency, damping, delay) {
  const t = time - delay;
  if (t < 0) return 0;
  if (Math.abs(damping - 1) < 1e-8) {
    return gain * (1 - (1 + frequency * t) * Math.exp(-frequency * t));
  }
  if (damping < 1) {
    const root = Math.sqrt(1 - damping * damping);
    return gain * (1 - Math.exp(-damping * frequency * t) *
      (Math.cos(frequency * root * t) + damping / root * Math.sin(frequency * root * t)));
  }
  const root = Math.sqrt(damping * damping - 1);
  const slow = frequency / (damping + root);
  const fast = frequency * (damping + root);
  return gain * (1 - (fast * Math.exp(-slow * t) - slow * Math.exp(-fast * t)) / (fast - slow));
}

const controls = ['gain', 'frequency', 'damping', 'delay'].map(id => document.getElementById(id));
const defaults = [1, 2, 0.35, 0.2];
const svgNS = 'http://www.w3.org/2000/svg';
function element(tag, attributes, text) {
  const node = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function render() {
  const [gain, frequency, damping, delay] = controls.map(control => Number(control.value));
  controls.forEach(control => {
    document.getElementById(`${control.id}-value`).value = Number(control.value).toFixed(2);
    const percent = 100 * (control.value - control.min) / (control.max - control.min);
    control.style.setProperty('--fill', `${percent}%`);
  });
  const regime = damping < 1 ? 'UNDERDAMPED' : damping > 1 ? 'OVERDAMPED' : 'CRITICALLY DAMPED';
  document.getElementById('regime').textContent = regime;
  // Extend the horizon for the actual slow pole of overdamped plants.
  const rate = damping <= 1 ? damping * frequency : frequency / (damping + Math.sqrt(damping * damping - 1));
  const duration = Math.max(5, delay + 7 / rate);
  const points = Array.from({length: 801}, (_, i) => {
    const t = duration * i / 800;
    return [t, plantResponse(t, gain, frequency, damping, delay)];
  });
  // Adapt to the response while keeping the unit-input reference visible.
  const ymax = Math.max(1, gain, ...points.map(point => point[1])) * 1.18;
  const left = 62, right = 704, top = 15, bottom = 269;
  const x = t => left + t / duration * (right - left);
  const y = value => bottom - value / ymax * (bottom - top);
  const grid = document.getElementById('grid');
  const ticks = document.getElementById('tick-labels');
  grid.replaceChildren();
  ticks.replaceChildren();
  for (let i = 0; i <= 4; i++) {
    const value = ymax * i / 4;
    grid.append(element('line', {x1:left, x2:right, y1:y(value), y2:y(value), class:'grid-line'}));
    ticks.append(element('text', {x:left - 12, y:y(value) + 4, 'text-anchor':'end', class:'tick'}, value.toFixed(2)));
  }
  for (let i = 0; i <= 5; i++) {
    const time = duration * i / 5;
    grid.append(element('line', {x1:x(time), x2:x(time), y1:top, y2:bottom, class:'grid-line'}));
    ticks.append(element('text', {x:x(time), y:bottom + 20, 'text-anchor':'middle', class:'tick'}, time.toFixed(1)));
  }
  document.getElementById('response-line').setAttribute('d', points.map(([t, value], i) =>
    `${i ? 'L' : 'M'}${x(t).toFixed(2)},${y(value).toFixed(2)}`).join(' '));
  document.getElementById('final-line').setAttribute('d', `M${left},${y(gain)} L${right},${y(gain)}`);
  document.getElementById('unit-reference').setAttribute('d', `M${left},${y(1)} L${right},${y(1)}`);
  document.getElementById('open-context').textContent = `Input stays at 1. Output approaches K = ${gain.toFixed(2)}. The unit-input reference is a comparison level, not a feedback target.`;
  renderPID(gain, frequency, damping, delay, duration, points);
  document.getElementById('chart-description').textContent =
    `Unit-step response: gain ${gain}, natural frequency ${frequency} radians per second, damping ${damping}, delay ${delay} seconds. ${regime.toLowerCase()}. Time axis spans ${duration.toFixed(1)} seconds and adjusts to plant speed. Output axis adjusts to the response and includes the unit input reference at 1. Final output approaches ${gain}.`;
}
controls.forEach(control => control.addEventListener('input', render));
document.getElementById('reset').addEventListener('click', () => {
  controls.forEach((control, i) => {control.value = defaults[i];});
  render();
});

const pidControls = ['kp', 'ki', 'kd'].map(id => document.getElementById(id));

// Same ideal PID and second-order Pade plant as src/data_gen.py.
function multiplyPolynomial(a, b) {
  const out = Array(a.length + b.length - 1).fill(0);
  a.forEach((x,i) => b.forEach((y,j) => {out[i+j] += x*y;}));
  return out;
}
function addPolynomial(a, b) {
  const out = Array(Math.max(a.length,b.length)).fill(0);
  a.forEach((v,i) => {out[out.length-a.length+i] += v;});
  b.forEach((v,i) => {out[out.length-b.length+i] += v;});
  return out;
}
function closedCoefficients(gain, frequency, damping, delay, kp, ki, kd) {
  let pn=[gain*frequency*frequency], pd=[1,2*damping*frequency,frequency*frequency];
  if (delay > 1e-5) {
    pn=multiplyPolynomial(pn,[delay*delay/12,-delay/2,1]);
    pd=multiplyPolynomial(pd,[delay*delay/12,delay/2,1]);
  }
  if (kp===0 && ki===0 && kd===0) return {num:[0],den:pd};
  const cn=ki===0?[kd,kp]:[kd,kp,ki], cd=ki===0?[1]:[1,0];
  const num=multiplyPolynomial(pn,cn);
  while(num.length>1 && num[0]===0) num.shift();
  const den=addPolynomial(multiplyPolynomial(pd,cd),num);
  const scale=den[0];
  return {num:num.map(v=>v/scale),den:den.map(v=>v/scale)};
}
function closedResponse(gain, frequency, damping, delay, kp, ki, kd, duration) {
  const {num,den}=closedCoefficients(gain,frequency,damping,delay,kp,ki,kd);
  const order=den.length-1;
  const padded=Array(order+1-num.length).fill(0).concat(num);
  // Controllable canonical state space, zero initial state, unit reference.
  // The plant output is proper even though ideal control effort has an impulse.
  const C=padded.slice(1);
  const step=Math.min(0.0005,delay>1e-5?delay/40:0.0005);
  const count=Math.ceil(duration/step),dt=duration/count;
  let state=Array(order).fill(0), peak=0, output=0, diverged=false;
  let itae=0, previousWeightedError=0;
  const points=[[0,0]],stride=Math.max(1,Math.floor(count/1000));
  const derivative=x=>[1-den.slice(1).reduce((sum,a,j)=>sum+a*x[j],0),...x.slice(0,-1)];
  const advance=(x,k,scale)=>x.map((v,j)=>v+scale*k[j]);
  for(let i=0;i<count;i++) {
    const k1=derivative(state),k2=derivative(advance(state,k1,dt/2));
    const k3=derivative(advance(state,k2,dt/2)),k4=derivative(advance(state,k3,dt));
    state=state.map((v,j)=>v+dt*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6);
    output=C.reduce((sum,c,j)=>sum+c*state[j],0);
    peak=Math.max(peak,output);
    if(!Number.isFinite(output)||Math.abs(output)>20){diverged=true;break;}
    // Integrate at every solver step, not just the decimated plotting points.
    const weightedError=(i+1)*dt*Math.abs(1-output);
    itae+=dt*(previousWeightedError+weightedError)/2;
    previousWeightedError=weightedError;
    if((i+1)%stride===0||i===count-1)points.push([(i+1)*dt,output]);
  }
  return {points,peak,error:1-output,itae,diverged};
}

function renderPID(gain, frequency, damping, delay, duration, openPoints) {
  const gains = pidControls.map(control => {
    document.getElementById(`${control.id}-value`).value = Number(control.value).toFixed(2);
    control.style.setProperty('--fill', `${100 * control.value / control.max}%`);
    return Number(control.value);
  });
  const result = closedResponse(gain, frequency, damping, delay, ...gains, duration);
  const visible = result.diverged ? openPoints.map(p => p[1]) : [...openPoints.map(p => p[1]), ...result.points.map(p => p[1])];
  const low = Math.min(0, ...visible), high = Math.max(1, ...visible);
  const margin = (high - low) * 0.12;
  const ymin = low - (low < 0 ? margin : 0), ymax = high + margin;
  const x = t => 62 + t / duration * 642;
  const y = v => 269 - (v - ymin) / (ymax - ymin) * 254;
  const grid = document.getElementById('pid-grid');
  const ticks = document.getElementById('pid-ticks');
  grid.replaceChildren(); ticks.replaceChildren();
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (ymax - ymin) * i / 4;
    grid.append(element('line', {x1:62,x2:704,y1:y(v),y2:y(v),class:'grid-line'}));
    ticks.append(element('text', {x:50,y:y(v)+4,'text-anchor':'end',class:'tick'}, v.toFixed(2)));
  }
  for (let i = 0; i <= 5; i++) {
    const t = duration * i / 5;
    grid.append(element('line', {x1:x(t),x2:x(t),y1:15,y2:269,class:'grid-line'}));
    ticks.append(element('text', {x:x(t),y:289,'text-anchor':'middle',class:'tick'},t.toFixed(1)));
  }
  const path = points => points.map(([t,v],i) => `${i?'L':'M'}${x(t).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
  document.getElementById('pid-open').setAttribute('d',path(openPoints));
  document.getElementById('pid-line').setAttribute('d',result.diverged ? '' : path(result.points));
  document.getElementById('pid-target').setAttribute('d',`M62,${y(1)} L704,${y(1)}`);
  const status = document.getElementById('pid-status');
  status.textContent = result.diverged ? 'DIVERGING · PID TRACE HIDDEN' : 'FINITE-WINDOW SIMULATION';
  status.classList.toggle('warning',result.diverged);
  document.getElementById('pid-overshoot').textContent = result.diverged ? '—' : `${Math.max(0,(result.peak-1)*100).toFixed(1)}%`;
  document.getElementById('pid-error').textContent = result.diverged ? '—' : result.error.toFixed(3);
  document.getElementById('pid-itae').textContent = result.diverged ? '—' :
    (result.itae < 0.001 ? result.itae.toExponential(2) : result.itae.toFixed(3));
  document.getElementById('pid-window').textContent = `${duration.toFixed(1)} s`;
  document.getElementById('pid-description').textContent = result.diverged ?
    'PID response exceeded the display limit and is hidden. Reduce gains or dead time.' :
    `Unit setpoint tracking with Kp ${gains[0]}, Ki ${gains[1]}, Kd ${gains[2]}. Sampled overshoot ${Math.max(0,(result.peak-1)*100).toFixed(1)} percent. Final error ${result.error.toFixed(3)}. ITAE over the displayed ${duration.toFixed(1)} seconds is ${result.itae.toPrecision(4)}. Uses the plant settings above.`;
}
pidControls.forEach(control => control.addEventListener('input', render));
document.getElementById('reset-pid').addEventListener('click', () => {
  pidControls.forEach((control,i) => {control.value = [1,0.5,0.25][i];});
  render();
});

function selectSection() {
  const previous = document.querySelector('main > section:not([hidden])')?.id;
  let selected = previous || 'overview';
  const requested = document.getElementById(location.hash.slice(1));
  if (!location.hash) selected = 'overview';
  else if (requested?.closest('main > section')) selected = requested.closest('main > section').id;
  document.querySelectorAll('main > section').forEach(section => {section.hidden = section.id !== selected;});
  document.querySelectorAll('.sidebar nav a').forEach(link => {
    const active = link.hash === `#${selected}`;
    link.classList.toggle('active',active);
    if (active) link.setAttribute('aria-current','page'); else link.removeAttribute('aria-current');
  });
  document.title = `ML-PID · ${{overview:'The study',basics:'System basics','data-generation':'Data generation',models:'Learning to tune',results:'Results & evidence',limitations:'Limits & next steps'}[selected] || 'Research'}`;
  document.querySelector('footer a').setAttribute('href', `#${selected}`);
  // The destination was hidden when the browser followed the hash. Start each
  // newly selected tab at the top instead of retaining the previous tab's scroll.
  if (previous !== selected) {
    if (requested && requested.id !== selected && requested.closest('main > section')) {
      requestAnimationFrame(() => requested.scrollIntoView({block:'start',behavior:'instant'}));
    } else window.scrollTo({top: 0, behavior: 'instant'});
  }
}
// Commit on Enter or blur so partial input such as "0." remains editable.
function snapParameter(value, min, max, step) {
  const bounded = Math.max(min, Math.min(max, value));
  return Number(Math.max(min, Math.min(max,
    min + Math.round((bounded - min) / step) * step)).toFixed(8));
}
for (const control of [...controls, ...pidControls]) {
  const field = document.getElementById(`${control.id}-value`);
  const commit = () => {
    if (field.value.trim() !== '' && Number.isFinite(field.valueAsNumber)) {
      control.value = snapParameter(field.valueAsNumber,
        Number(control.min), Number(control.max), Number(control.step));
    }
    render(); // Empty/invalid input restores the last valid slider value.
  };
  field.addEventListener('change', commit);
  field.addEventListener('blur', () => {
    if (field.value !== Number(control.value).toFixed(2)) commit();
  });
  field.addEventListener('keydown', event => {
    if (event.key === 'Enter') {event.preventDefault(); commit(); field.blur();}
    if (event.key === 'Escape') {field.value = Number(control.value).toFixed(2); field.blur();}
  });
}

window.addEventListener('hashchange',selectSection);
selectSection();
render();

// Make intentional table/diagram scrolling clear and keyboard-accessible.
let overflowFrame = 0;
function queueOverflowCues() {
  if (overflowFrame) return;
  overflowFrame = requestAnimationFrame(() => {
    overflowFrame = 0;
    document.querySelectorAll('.study-table-wrap,.table-wrap,.dataset-row,.feedback-diagram').forEach(region => {
      if (!region.clientWidth) return;
      const scrollable = region.scrollWidth > region.clientWidth + 2;
      let hint = region.nextElementSibling;
      if (!hint?.classList.contains('scroll-hint')) {
        if (!scrollable) return;
        hint = document.createElement('p');
        hint.className = 'scroll-hint';
        hint.setAttribute('aria-hidden', 'true');
        hint.textContent = region.classList.contains('feedback-diagram')
          ? 'Scroll horizontally to follow the complete diagram →'
          : 'Scroll horizontally to see every column →';
        region.after(hint);
      }
      hint.hidden = !scrollable;
      if (scrollable) {
        region.tabIndex = 0;
        region.setAttribute('role', 'region');
        const caption = region.querySelector('caption')?.textContent.trim();
        region.setAttribute('aria-label', caption ? `Scrollable table: ${caption}` : region.classList.contains('feedback-diagram') ? 'Scrollable feedback diagram' : 'Scrollable data table');
      } else {
        region.removeAttribute('tabindex');
        region.removeAttribute('role');
        region.removeAttribute('aria-label');
      }
    });
  });
}
window.addEventListener('resize', queueOverflowCues);
window.addEventListener('hashchange', queueOverflowCues);
new MutationObserver(queueOverflowCues).observe(document.getElementById('main'), {childList: true, subtree: true});
document.getElementById('main').addEventListener('keydown', event => {
  const region = event.target;
  if (region.getAttribute('role') !== 'region' || region.tabIndex !== 0) return;
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  event.preventDefault();
  region.scrollBy({left: event.key === 'ArrowRight' ? 120 : -120, behavior: 'instant'});
});
queueOverflowCues();


/* ===== Data-generation interface ===== */
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
    note.textContent='Each accepted row contains 4 plant parameters, 5 response features, 3 PID target gains, and the observation duration as metadata.';
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
    byId('sample-equation').innerHTML=
      `<span>G(s) = <span class="fraction"><span>${format(p.K*p.wn*p.wn)}</span><span>s<sup>2</sup> + ${format(2*p.zeta*p.wn)}s + ${format(p.wn*p.wn)}</span></span> e<sup>−${p.tau.toFixed(2)}s</sup></span>`;
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
      const symbols={K:'K',wn:'ω<sub>n</sub>',zeta:'ζ',tau:'τ',t_rise:'t<sub>rise</sub>',t_settling:'t<sub>settling</sub>',
        y_peak:'y<sub>peak</sub>',y_ss:'y<sub>ss</sub>',tau_est:'τ<sub>est</sub>',Kp:'K<sub>p</sub>',Ki:'K<sub>i</sub>',Kd:'K<sub>d</sub>',observation_duration:'Observation / s'};
      if(symbols[key])label.innerHTML=symbols[key];else label.textContent=key;
      number.textContent=format(value);number.title=String(value);
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
      byId('feature-window').textContent=`${data.n_points.toLocaleString()} POINTS / ${data.duration.toFixed(2)} s`;
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
        'The selected controller has the lowest regularized objective among those that passed validation. The chart focuses on the changing part of the response; metrics use the full simulation. Its search is highlighted in the table.','done');
      const best=result.diagnostics.filter(d=>d.validation==='accepted').reduce((a,b)=>a.objective<=b.objective?a:b);
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
    const random=new Uint32Array(5);crypto.getRandomValues(random);
    sliders.forEach((control,i)=>{
      control.value=snapParameter(Number(control.min)+(Number(control.max)-Number(control.min))*random[i]/4294967296,
        Number(control.min),Number(control.max),0.01);
    });
    if(random[4]/4294967296 < 0.1)byId('sample-tau').value='0';
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


/* ===== Supplemental stress-test charts ===== */
/* Native SVG charts drawn from completed, frozen post-test analyses. */
(() => {
  'use strict';
  const data=window.PID_SUPPLEMENT;if(!data)return;
  const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
  const names={standard_mlp:'MLP',random_forest:'Random Forest',xgboost:'XGBoost',numerical_reference:'Numerical reference'};
  const order=['standard_mlp','random_forest','xgboost','numerical_reference'];
  const colors=['var(--accent)','var(--model-forest)','var(--model-xgboost)','var(--model-numerical)'];
  const fmt=(n,d=1)=>Number.isFinite(n)?n.toFixed(d):'—';
  function add(parent,tag,attrs={},text=''){const el=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);el.textContent=text;parent.append(el);return el;}
  function graph(id,{xmin=0,xmax=20,ymax=100,ymin=0,xlabel='',ylabel='',xticks=[0,5,10,15,20],log=false}={}){
    const svg=$(id);svg.replaceChildren();const narrow=innerWidth<600,W=narrow?600:800,H=350;
    svg.setAttribute('viewBox',`0 0 ${W} ${H}`);const p={l:70,r:26,t:24,b:66};
    const scale=x=>log?Math.log10(x):x;
    const X=x=>p.l+(scale(x)-scale(xmin))/(scale(xmax)-scale(xmin))*(W-p.l-p.r),Y=y=>H-p.b-(y-ymin)/(ymax-ymin)*(H-p.t-p.b);
    for(const x of xticks){add(svg,'line',{x1:X(x),x2:X(x),y1:p.t,y2:H-p.b,class:'chart-grid'});add(svg,'text',{x:X(x),y:H-p.b+23,'text-anchor':'middle',class:'chart-tick'},String(x));}
    for(const y of [0,.25,.5,.75,1].map(t=>ymin+t*(ymax-ymin))){add(svg,'line',{x1:p.l,x2:W-p.r,y1:Y(y),y2:Y(y),class:'chart-grid'});add(svg,'text',{x:p.l-12,y:Y(y)+5,'text-anchor':'end',class:'chart-tick'},fmt(y,ymax-ymin<10?2:0));}
    add(svg,'text',{x:(p.l+W-p.r)/2,y:H-10,'text-anchor':'middle',class:'chart-label'},xlabel);
    add(svg,'text',{transform:`translate(18 ${(p.t+H-p.b)/2}) rotate(-90)`,'text-anchor':'middle',class:'chart-label'},ylabel);
    return {svg,X,Y,p,W,H};
  }
  function line(g,points,color,dash=''){add(g.svg,'path',{d:points.map(([x,y],i)=>`${i?'L':'M'}${g.X(x)},${g.Y(y)}`).join(' '),stroke:color,'stroke-width':2.7,fill:'none','stroke-dasharray':dash});for(const [x,y] of points)if(points.length<12)add(g.svg,'circle',{cx:g.X(x),cy:g.Y(y),r:4,fill:color});}
  function band(g,rows,color,x){const points=[...rows.map(r=>[x(r),r.cluster_ci_low]),...rows.toReversed().map(r=>[x(r),r.cluster_ci_high])];add(g.svg,'polygon',{points:points.map(([a,b])=>`${g.X(a)},${g.Y(b)}`).join(' '),fill:color,opacity:.10});}
  function table(id,caption,headers,rows){$(id).innerHTML=`<div class="study-table-wrap"><table class="study-table"><caption>${caption}</caption><thead><tr>${headers.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((v,i)=>`<td class="${i?'numeric':''}">${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
  function render(){
    document.querySelectorAll('.stress-legend').forEach(legend=>legend.querySelectorAll('span').forEach((el,i)=>{el.querySelector('i').style.borderColor=colors[i];}));
    $('tolerance-chart').parentElement.querySelectorAll('.chart-legend span').forEach((el,i)=>{el.querySelector('i').style.borderColor=colors[i];});
    let g=graph('tolerance-chart',{xmin:10,xmax:20,xticks:[10,12,15,20],xlabel:'EXPLORATORY OVERSHOOT LIMIT / %',ylabel:'ALL TEST PLANTS / %'});
    order.slice(0,3).forEach((name,i)=>line(g,data.thresholds.filter(r=>r.model===name).map(r=>[r.threshold_pct,r.pct_all_plants]),colors[i],i===2?'7 4':''));
    for(const row of data.thresholds.filter(r=>r.model==='standard_mlp'))add(g.svg,'text',{x:g.X(row.threshold_pct),y:g.Y(row.pct_all_plants)-12,'text-anchor':row.threshold_pct===20?'end':row.threshold_pct===10?'start':'middle',class:'chart-label',fill:'var(--accent)'},`${fmt(row.pct_all_plants,1)}%`);
    g=graph('noise-chart',{xmin:.01,xmax:2,xticks:[.01,.05,.1,.5,1,2],log:true,xlabel:'NOISE STANDARD DEVIATION / K (%)',ylabel:'CONDITIONAL ACCEPTANCE / %'});
    order.forEach((name,i)=>{const rows=data.noise.filter(r=>r.model===name).sort((a,b)=>a.level-b.level);band(g,rows,colors[i],r=>r.level*100);line(g,rows.map(r=>[r.level*100,r.success_pct]),colors[i],i>1?'6 4':'');});
    g=graph('uncertainty-chart',{xmin:5,xmax:20,xticks:[5,10,20],xlabel:'JOINT PARAMETER CHANGE BOUND / %',ylabel:'STRICT ACCEPTANCE / %'});
    order.forEach((name,i)=>{const rows=data.uncertainty.filter(r=>r.model===name).sort((a,b)=>a.level-b.level);band(g,rows,colors[i],r=>r.level*100);line(g,rows.map(r=>[r.level*100,r.success_pct]),colors[i],i>1?'6 4':'');});
    const svg=$('true-delay-chart');svg.replaceChildren();const W=innerWidth<600?600:800;svg.setAttribute('viewBox',`0 0 ${W} 340`);
    const X=x=>180+x*(W-260)/100;
    [0,25,50,75,100].forEach(x=>{add(svg,'line',{x1:X(x),x2:X(x),y1:24,y2:285,class:'chart-grid'});add(svg,'text',{x:X(x),y:312,'text-anchor':'middle',class:'chart-tick'},String(x));});
    order.forEach((name,i)=>{const r=data.exact.find(r=>r.model===name),y=40+i*63;add(svg,'text',{x:165,y:y+17,'text-anchor':'end',class:'chart-label'},name==='numerical_reference'?'Numerical':names[name]);for(const [j,val]of [r.pade_passes,r.true_delay_time_domain_passes].entries()){add(svg,'rect',{x:180,y:y+j*22,width:X(val)-180,height:16,rx:2,fill:j?'var(--accent)':'var(--trace-muted)'});add(svg,'text',{x:X(val)+6,y:y+j*22+13,class:'chart-label'},String(val));}});
    add(svg,'text',{x:180,y:332,class:'chart-label'},'PASSES / 100 · GREY: PADÉ · RED: TRUE-DELAY TIME DOMAIN');
    renderDelay();
  }
  function renderDelay(){const r=data.curveCases[Number($('delay-case').value)],g=graph('delay-response-chart',{xmin:0,xmax:r.t.at(-1),ymin:-.3,ymax:1.45,xticks:[0,2,4,6].filter(x=>x<=r.t.at(-1)),xlabel:'TIME / SECONDS',ylabel:'OUTPUT'});line(g,[[0,1],[r.t.at(-1),1]],'var(--reference)','2 5');line(g,r.t.map((t,i)=>[t,r.pade[i]]),'var(--trace-muted)','7 4');line(g,r.t.map((t,i)=>[t,r.true[i]]),'var(--accent)');$('delay-case-note').textContent=`Plant ${r.row}, identical frozen MLP gains: Padé overshoot ${fmt(r.padePeak,2)}%; true-delay overshoot ${fmt(r.truePeak,2)}%. A shorter display window shows the transient; stress metrics use the independent full checks. These deliberately selected discrepancies are examples, not a random extra test set.`;}
  table('diagnostic-quality-table','MLP · converged response quality by overshoot severity',['Overshoot band','Plants','Median ITAE / reference','Median settling / reference'],data.bands.filter(r=>r.model==='standard_mlp').map(r=>[r.band,r.plants,fmt(r.median_itae_ratio,3)+'×',fmt(r.median_settling_ratio,3)+'×']));
  const groups=[...new Set(data.boundary.map(r=>r.group))];
  table('boundary-evidence-table','Boundary plants · strict passes / plants',['Plant region',...order.map(n=>names[n])],groups.map(group=>[group.replaceAll('_',' '),...order.map(name=>{const r=data.boundary.find(r=>r.group===group&&r.model===name);return `<span class="boundary-score ${r.success_pct<60?'low':''}">${r.passes}/${r.cases}</span>`;})]));
  table('disturbance-evidence-table','Load disturbance · metrics among converged recovered responses',['Controller','Recovered','Median peak deviation','Median recovery','Median disturbance ITAE'],order.map(name=>{const r=data.disturbance.find(r=>r.model===name);return [names[name],`${r.recovered}/${r.cases}`,fmt(r.median_peak_deviation*100,2)+'%',fmt(r.median_recovery_seconds,2)+' s',fmt(r.median_disturbance_itae,4)];}));
  $('delay-case').addEventListener('change',renderDelay);window.addEventListener('resize',render);render();
})();


/* ===== Saved research evidence and charts ===== */
/* Saved evidence, native SVG. Controls explore results; they never retrain models. */
(() => {
  'use strict';
  const study = window.PID_STUDY;
  if (!study) return;
  const N = study.testCount || 1500;
  const NS = 'http://www.w3.org/2000/svg';
  const red = 'var(--accent)', ink = 'var(--reference)', grey = 'var(--trace-muted)';
  const colors = [1, 2, 3, 4, 5].map(i => `var(--chart-series-${i})`).concat(red);
  const $ = id => document.getElementById(id);
  const fmt = (n, d = 2) => Number.isFinite(n) ? n.toFixed(d) : '—';
  const shortName = name => ({'Linear regression': 'Linear', 'Polynomial ridge': 'Polynomial', 'Numerical tuning': 'Optimizer'}[name] || name);
  let compact = window.matchMedia('(max-width:600px)').matches;
  function node(tag, attrs = {}, text = '') {
    const n = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) n.setAttribute(key, value);
    if (text !== '') n.textContent = text;
    return n;
  }
  function add(parent, tag, attrs, text) { const n = node(tag, attrs, text); parent.append(n); return n; }
  function clear(id, title) { const svg = $(id); svg.replaceChildren(node('title', {}, title)); return svg; }
  function graph(id, {xmin = 0, xmax = 1, ymin = 0, ymax = 100, xLabel = '', yLabel = '', logX = false, logY = false, title = '', xTicks, yTicks, left = 66, bottom = 54} = {}) {
    const svg = clear(id, title), H = Number(svg.getAttribute('viewBox').split(' ')[3]), W = compact ? 640 : 800;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const pad = {l: left, r: 24, t: 25, b: bottom};
    const tx = x => logX ? Math.log10(x) : x, ty = y => logY ? Math.log10(y) : y;
    const X = x => pad.l + (tx(x) - tx(xmin)) / (tx(xmax) - tx(xmin)) * (W - pad.l - pad.r);
    const Y = y => H - pad.b - (ty(y) - ty(ymin)) / (ty(ymax) - ty(ymin)) * (H - pad.b - pad.t);
    const ticks = (a, b, log) => log ? Array.from({length: Math.floor(Math.log10(b)) - Math.ceil(Math.log10(a)) + 1}, (_, i) => 10 ** (i + Math.ceil(Math.log10(a)))) : Array.from({length: 6}, (_, i) => a + (b - a) * i / 5);
    const tickText = v => v === 0 ? '0' : Math.abs(v) >= 1000 ? `${fmt(v / 1000, 0)}k` : Math.abs(v) < .01 ? v.toExponential(0) : Number(v.toPrecision(3)).toString();
    for (const value of xTicks || ticks(xmin, xmax, logX)) {
      add(svg, 'line', {x1: X(value), x2: X(value), y1: pad.t, y2: H - pad.b, class: 'chart-grid'});
      add(svg, 'text', {x: X(value), y: H - pad.b + 20, 'text-anchor': 'middle', class: 'chart-tick'}, tickText(value));
    }
    for (const value of yTicks || ticks(ymin, ymax, logY)) {
      add(svg, 'line', {x1: pad.l, x2: W - pad.r, y1: Y(value), y2: Y(value), class: 'chart-grid'});
      add(svg, 'text', {x: pad.l - 10, y: Y(value) + 4, 'text-anchor': 'end', class: 'chart-tick'}, tickText(value));
    }
    add(svg, 'text', {x: (pad.l + W - pad.r) / 2, y: H - 8, 'text-anchor': 'middle', class: 'chart-label'}, xLabel);
    add(svg, 'text', {transform: `translate(17 ${(pad.t + H - pad.b) / 2}) rotate(-90)`, 'text-anchor': 'middle', class: 'chart-label'}, yLabel);
    const clipId = `${id}-clip`;
    const defs = add(svg, 'defs', {}), clip = add(defs, 'clipPath', {id: clipId});
    add(clip, 'rect', {x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b});
    const plot = add(svg, 'g', {'clip-path': `url(#${clipId})`});
    return {svg, plot, X, Y, W, H, pad};
  }
  function line(g, points, color, dash = '', width = 2.3) {
    let d = '', active = false;
    for (const [x, y] of points) {
      if (!Number.isFinite(g.X(x)) || !Number.isFinite(g.Y(y))) {active = false; continue;}
      d += `${active ? 'L' : 'M'}${g.X(x).toFixed(2)},${g.Y(y).toFixed(2)}`; active = true;
    }
    return add(g.plot, 'path', {d, fill: 'none', stroke: color, 'stroke-width': width, 'stroke-dasharray': dash, 'stroke-linejoin': 'round'});
  }
  function curve(id, item, model, hero = false) {
    const pred = item.models[model], arrays = hero ? [item.open, pred.y] : [item.open, item.reference, pred.y];
    // Crop only the display's settled tail, never the stored validation metrics.
    let lastChange = 0;
    arrays.forEach((ys, k) => {if (ys) ys.forEach((y, i) => {const final = k === 0 ? item.plant.K : 1; if (Math.abs(y - final) > .02 * Math.abs(final)) lastChange = Math.max(lastChange, item.t[i]);});});
    const end = Math.min(item.t.at(-1), Math.max(2, lastChange * 1.2));
    const all = arrays.filter(Boolean).flatMap(ys => ys.filter((_, i) => item.t[i] <= end)), lo = Math.min(0, ...all), hi = Math.max(1.1, ...all), margin = (hi - lo) * .08;
    const g = graph(id, {xmax: end, ymin: lo - margin, ymax: hi + margin, xLabel: 'TIME / SECONDS', yLabel: 'OUTPUT', title: `${item.name}: plant, reference and learned PID`});
    const zip = ys => item.t.map((t, i) => [t, ys[i]]);
    line(g, [[0, 1], [end, 1]], ink, '2 5', 1.2);
    line(g, zip(item.open), grey, '7 4', 2);
    if (!hero && item.reference) line(g, zip(item.reference), ink, '', 1.8);
    if (pred.y) line(g, zip(pred.y), red, '', 2.6);
    else add(g.svg, 'text', {x: 90, y: 45, fill: red, class: 'chart-label'}, `Model trace omitted: ${pred.reason.replaceAll('_', ' ')}`);
  }
  window.PID_RENDER_RESPONSE = (id,item) => curve(id,item,'live');
  const modelSelect = $('result-model');
  study.models.forEach(m => modelSelect.add(new Option(m.name, m.id)));
  modelSelect.value = 'standard_mlp';
  study.cases.forEach((c, i) => $('case-choice').add(new Option(c.name, i)));
  const selected = () => study.models.find(m => m.id === modelSelect.value);
  const readout = (value, label) => `<div><strong>${value}</strong><span>${label}</span></div>`;
  $('results-table').innerHTML = study.models.map(m => `<tr class="${m.id === 'standard_mlp' ? 'highlight' : ''}"><td>${m.name}</td><td class="numeric">${m.metrics.passes} · ${fmt(m.metrics.success_pct)}%</td><td class="numeric">${fmt(m.metrics.stable_pct, 1)}%</td><td class="numeric">${fmt(m.metrics.r2, 3)}</td><td class="numeric">${fmt(m.metrics.median_ITAE_ratio, 3)}×</td><td class="numeric">${fmt(m.metrics.median_J_ratio, 3)}×</td></tr>`).join('');

  function renderAcceptance() {
    const svg = clear('acceptance-chart', `Full controller acceptance on ${N} held-out plants`);
    const W = compact ? 640 : 800;
    svg.setAttribute('viewBox', `0 0 ${W} 340`);
    const x = v => 190 + v * (W - 260) / 100;
    [0, 25, 50, 75, 100].forEach(t => {
      add(svg, 'line', {x1: x(t), x2: x(t), y1: 20, y2: 295, class: 'chart-grid'});
      add(svg, 'text', {x: x(t), y: 322, 'text-anchor': 'middle', class: 'chart-tick'}, `${t}%`);
    });
    study.models.forEach((m, i) => {
      const y = 30 + i * 44;
      add(svg, 'text', {x: 174, y: y + 18, 'text-anchor': 'end', class: 'chart-label'}, shortName(m.name));
      add(svg, 'rect', {x: 190, y, width: x(m.metrics.success_pct) - 190, height: 26, rx: 2, fill: m.id === 'standard_mlp' ? red : (m.id === 'random_forest' ? ink : 'var(--trace-muted)')});
      add(svg, 'text', {x: x(m.metrics.success_pct) + 9, y: y + 18, fill: ink, class: 'chart-label'}, `${fmt(m.metrics.success_pct)}%`);
    });
  }
  function renderGain(m) {
    const index = ['Kp', 'Ki', 'Kd'].indexOf($('gain-choice').value);
    const points = m.scatter.filter(p => p.actual[index] > 0 && p.predicted[index] > 0);
    const values = points.flatMap(p => [p.actual[index], p.predicted[index]]);
    const lower = Math.max(1e-6, 10 ** Math.floor(Math.log10(Math.min(...values))));
    const upper = 10 ** Math.ceil(Math.log10(Math.max(...values)));
    const g = graph('gain-chart', {xmin: lower, xmax: upper, ymin: lower, ymax: upper, logX: true, logY: true, xLabel: `REFERENCE ${$('gain-choice').value}`, yLabel: `PREDICTED ${$('gain-choice').value}`, title: `${m.name} gain agreement`});
    line(g, [[lower, lower], [upper, upper]], 'var(--line-strong)', '5 4', 1.3);
    points.forEach(p => {
      const circle = add(g.plot, 'circle', {cx: g.X(p.actual[index]), cy: g.Y(p.predicted[index]), r: 3.4, fill: p.pass ? 'var(--muted)' : red, opacity: .82});
      add(circle, 'title', {}, `Plant ${p.row}: reference ${fmt(p.actual[index], 4)}, predicted ${fmt(p.predicted[index], 4)}; ${p.pass ? 'accepted' : 'fails full validation'}`);
    });
    $('gain-note').textContent = `${points.length} of 180 sampled predictions are positive on this gain; ${180 - points.length} nonpositive predictions cannot appear on a log axis.`;
  }
  function renderItae(m) {
    const positive = m.itaeCdf.filter(p => p[0] > 0);
    const lower = Math.min(.5, positive[0][0] * .9), upper = Math.max(2, positive.at(-1)[0] * 1.15);
    const ticks = [];
    for (let p = Math.floor(Math.log10(lower)); p <= Math.ceil(Math.log10(upper)); p++) for (const k of [1, 2, 5]) {const v = k * 10 ** p; if (v >= lower && v <= upper) ticks.push(v);}
    const g = graph('itae-chart', {xmin: lower, xmax: upper, logX: true, xTicks: ticks, xLabel: 'ITAE RATIO / PREDICTED ÷ REFERENCE', yLabel: 'ACCEPTED PAIRS / PERCENTILE', title: `${m.name} accepted-pair ITAE distribution`});
    line(g, [[1, 0], [1, 100]], ink, '3 4', 1.3);
    line(g, positive, red, '', 2.7);
    $('itae-note').textContent = `${m.metrics.ratio_pairs.toLocaleString()} / ${N.toLocaleString()} accepted pairs (${fmt(m.metrics.success_pct)}% coverage). Median ${fmt(m.metrics.median_ITAE_ratio, 3)}×; 90th percentile ${fmt(m.itaeP90, 3)}×; maximum ${fmt(m.itaeMax, 2)}×. ${m.itaeOver2} accepted controllers exceed twice the reference ITAE. Failures are excluded, not assigned zero cost.`;
  }
  function renderOvershoot(m) {
    const threshold = Number($('overshoot-threshold').value), obs = m.measuredOvershoot;
    $('overshoot-value').textContent = `${fmt(threshold, 1)}%`;
    const g = graph('overshoot-chart', {xmax: 30, xLabel: 'MEASURED PEAK OVERSHOOT / %', yLabel: 'ALL TEST PLANTS / %', title: `${m.name} overshoot distribution; study rule stays at 10 percent`});
    const points = [[0, 0]];
    for (let t = 0; t <= 30.001; t += .1) points.push([t, 100 * obs.filter(v => v <= t + 1e-6).length / N]);
    line(g, points, red, '', 2.6);
    line(g, [[10, 0], [10, 100]], ink, '5 5', 1.3);
    line(g, [[threshold, 0], [threshold, 100]], grey, '2 3', 1.8);
    const n = obs.filter(v => v <= threshold + 1e-6).length;
    $('threshold-note').innerHTML = `<strong>${n.toLocaleString()} / ${N.toLocaleString()} (${fmt(100*n/N)}%)</strong> have stable, measured overshoot ≤${fmt(threshold, 1)}%. Full acceptance at the fixed 10% rule remains <strong>${m.metrics.passes} / ${N.toLocaleString()}</strong>. ${N - obs.length} responses are unstable or unmeasured; ${obs.filter(v => v > 30).length} measured responses exceed the chart’s 30% range.`;
    const b = m.overshoot;
    const bins = [['≤10%', b.at_most_10], ['10–12%', b.over_10_to_12], ['12–15%', b.over_12_to_15], ['15–20%', b.over_15_to_20], ['>20%', b.over_20], ['Stable, unmeasured', b.stable_unmeasured], ['Unstable', b.unstable]];
    $('failure-table').innerHTML = `<div class="study-table-wrap"><table class="study-table"><caption class="sr-only">All test plants by overshoot severity or measurement status</caption><thead><tr>${bins.map(([k]) => `<th>${k}</th>`).join('')}</tr></thead><tbody><tr>${bins.map(([, n]) => `<td class="numeric">${n}</td>`).join('')}</tr></tbody></table></div><p class="chart-caption">Full-validation outcomes: ${Object.entries(m.reasons).map(([k, n]) => `${k.replaceAll('_', ' ')} ${n}`).join(' · ')}.</p>`;
  }
  function renderCase() {
    const m = selected(), item = study.cases[Number($('case-choice').value)], p = item.models[m.id];
    curve('case-chart', item, m.id);
    const plant = item.plant;
    $('case-note').textContent = `Saved test row ${item.sourceRow}. K ${fmt(plant.K, 3)} · ωₙ ${fmt(plant.wn, 3)} rad/s · ζ ${fmt(plant.zeta, 3)} · τ ${fmt(plant.tau, 3)} s. Predicted gains: ${p.gains.map(v => fmt(v, 4)).join(' / ')} (proportional / integral / derivative). Strict outcome: ${p.reason.replaceAll('_', ' ')}. ${p.diagnosticResolved?`Converged ITAE ${fmt(p.diagnosticItae,4)}; 2% settling time ${fmt(p.diagnosticSettling,3)} s. An overshoot miss does not erase these response metrics.`:'Response diagnostics are unresolved.'}`;
    $('case-readouts').innerHTML = readout(p.accepted ? 'Meets checks' : p.reason==='overshoot'?'Overshoot miss':'Unresolved / other', 'ORIGINAL STRICT CRITERION') + readout(`${fmt(p.overshoot)}%`, 'MEASURED OVERSHOOT') + readout(p.diagnosticResolved ? `${fmt(p.diagnosticItaeRatio, 3)}×` : '—', 'ITAE / REFERENCE · RESOLVED PAIRS');
  }
  function renderGroups(m) {
    const groups = m.groups[$('group-choice').value];
    $('group-table').innerHTML = `<div class="study-table-wrap"><table class="study-table"><caption>Full numerical acceptance by plant group · ${m.name}</caption><thead><tr><th>Group</th><th>Plants</th><th>Accepted</th><th>Acceptance rate</th></tr></thead><tbody>${groups.map(g => `<tr><td>${g.name}</td><td class="numeric">${g.n}</td><td class="numeric">${g.passes}</td><td class="numeric">${fmt(g.pct)}%</td></tr>`).join('')}</tbody></table></div>`;
  }
  function renderLearning() {
    const maximum = Math.max(...study.models.flatMap(m => m.learning.map(p => p[0])));
    const g = graph('learning-chart', {xmin: 1000, xmax: maximum, ymin: 0, ymax: 100, xLabel: 'TRAINING PLANTS / 10K FINAL DATASET = 7K TRAIN', yLabel: 'VALIDATION ACCEPTANCE / %', xTicks: [1000,5000,10000,15000,20000].filter(v=>v<=maximum), title: 'Learning curves on the same 500 validation plants'});
    line(g,[[7000,0],[7000,100]],ink,'3 6',1.2);
    study.models.forEach((m, i) => {
      line(g, m.learning, colors[i], i < 3 ? '5 4' : '', m.id === 'standard_mlp' ? 3 : 2);
      m.learning.forEach(([x, y]) => {const dot = add(g.plot, 'circle', {cx: g.X(x), cy: g.Y(y), r: 4, fill: colors[i]}); add(dot, 'title', {}, `${m.name}: ${x} training plants, ${fmt(y, 1)}% validation acceptance`);});
    });
    $('learning-legend').innerHTML = study.models.map((m, i) => `<span><i style="border-color:${colors[i]};${i < 3 ? 'border-top-style:dashed' : ''}"></i>${m.name}</span>`).join('');
    const mlp=study.models.find(m=>m.id==='standard_mlp'), first=mlp.learning[0], last=mlp.learning.at(-1), chosen=mlp.learning.find(p=>p[0]===7000);
    $('learning-note').textContent=`The same 500 validation plants at every size; fixed settings and nested training prefixes. MLP acceptance: ${mlp.learning.map(p=>`${p[0].toLocaleString()} train → ${fmt(p[1],1)}%`).join(' · ')}. The dashed line marks the chosen 7,000 training plants.`;
    $('dataset-choice').textContent=`We chose 10,000 total rows (7,000 training / 1,500 validation / 1,500 test). MLP validation acceptance rises from ${fmt(first[1],1)}% at ${first[0].toLocaleString()} training plants to ${fmt(chosen[1],1)}% at 7,000; extending to ${last[0].toLocaleString()} training plants reaches ${fmt(last[1],1)}%. The marginal gain is smaller than the early learning gains, while labeling and training costs keep growing. The final size gives a reproducible, manageable comparative study with strong nonlinear-model performance. Larger datasets still help; 10K is not a proven optimum or a safety guarantee.`;
  }
  function renderSpeed() {
    const g = graph('speed-chart', {xmin: .1, xmax: 2000, ymin: 0, ymax: 7, logX: true, xLabel: 'WARM SINGLE-PLANT TIME / MILLISECONDS · LOG SCALE', yTicks: [], left: 190, title: 'Prediction latency versus sequential optimization and validation'});
    [...study.models.map(m => ({name: m.name, value: m.timing.median_prediction_ms})), {name: 'Numerical tuning', value: study.optimizerMedianMs}].forEach((r, i) => {
      const y = 6.6 - i;
      add(g.svg, 'text', {x: 177, y: g.Y(y) + 5, 'text-anchor': 'end', class: 'chart-label'}, shortName(r.name));
      add(g.plot, 'rect', {x: g.X(.1), y: g.Y(y) - 10, width: Math.max(2, g.X(r.value) - g.X(.1)), height: 20, fill: i === 5 ? red : i === 6 ? ink : 'var(--trace-muted)', rx: 2});
      add(g.svg, 'text', {x: i === 6 ? g.W - g.pad.r : g.X(r.value) + 8, y: g.Y(y) + (i === 6 ? -15 : 4), 'text-anchor': i === 6 ? 'end' : 'start', class: 'chart-label'}, `${fmt(r.value, r.value < 1 ? 3 : 1)} ms`);
    });
    const m=study.models.find(m=>m.id==='standard_mlp').timing;
    $('speed-note').textContent=`${study.timingScope}: median numerical tuning plus validation ${(study.optimizerMedianMs/1000).toFixed(3)} s; warm MLP prediction ${fmt(m.median_prediction_ms,3)} ms. Median matched prediction speedup ${fmt(m.median_prediction_speedup,0)}×. Including verification: ${fmt(m.median_prediction_plus_validation_ms,1)} ms and ${fmt(m.median_verified_speedup,1)}× speedup on jointly accepted pairs.`;
  }
  function renderModel() {
    const m = selected();
    $('model-readouts').innerHTML = readout(`${fmt(m.metrics.success_pct)}%`, `STRICT ACCEPTANCE / ${N.toLocaleString()} TEST PLANTS`) + readout(`${fmt(m.metrics.median_ITAE_ratio, 3)}×`, 'MEDIAN ITAE / ACCEPTED PAIRS') + readout(`${N - m.metrics.passes}`, 'OUTSIDE STRICT CHECKS');
    $('gain-metrics-table').innerHTML = `<div class="study-table-wrap"><table class="study-table"><caption>${m.name} · decoded gain errors</caption><thead><tr><th>Gain</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>${['Kp', 'Ki', 'Kd'].map(g => `<tr><td>${g}</td><td class="numeric">${fmt(m.metrics[`${g}_mae`], 4)}</td><td class="numeric">${fmt(m.metrics[`${g}_rmse`], 4)}</td><td class="numeric">${fmt(m.metrics[`${g}_r2`], 4)}</td></tr>`).join('')}</tbody></table></div>`;
    renderGain(m); renderItae(m); renderOvershoot(m); renderCase(); renderGroups(m);
  }
  modelSelect.addEventListener('change', renderModel);
  $('gain-choice').addEventListener('change', () => renderGain(selected()));
  $('overshoot-threshold').addEventListener('input', () => renderOvershoot(selected()));
  $('case-choice').addEventListener('change', renderCase);
  $('group-choice').addEventListener('change', () => renderGroups(selected()));
  if (study.rejected) {
    const reasons=study.rejectedProvenance.rejection_groups;
    $('generation-rejection-summary').textContent=`Observed plant-level reasons in this run: ${reasons.resolution_limit||0} resolution-only rejections, ${reasons.overshoot||0} overshoot-only rejections, and ${reasons['overshoot+resolution_limit']||0} with mixed overshoot/resolution outcomes across starts. Counts refer to plants, not individual failed searches.`;
    const ordered=study.models.map(m=>study.rejected.find(r=>r.model===m.id));
    $('stress-table').innerHTML=`<div class="study-table-wrap"><table class="study-table"><thead><tr><th>Model</th><th>Rejected plants</th><th>Strict passes</th><th>Passes after resolution-only retry</th></tr></thead><tbody>${ordered.map(r=>`<tr class="${r.model==='standard_mlp'?'highlight':''}"><td>${study.models.find(m=>m.id===r.model).name}</td><td>${r.plants}</td><td>${r.strict_passes}</td><td>${r.extended_passes}</td></tr>`).join('')}</tbody></table></div>`;
    const r=study.rejected.find(r=>r.model==='standard_mlp');
    $('stress-note').textContent=`The final MLP passes ${r.strict_passes}/${r.plants} at the original grid budget and ${r.extended_passes}/${r.plants} (${fmt(100*r.extended_passes/r.plants,1)}%) with resolution-only retries. ${r.extended_passes>r.plants/2?'Most of these rejected plants can receive validated model-predicted gains under the expanded budget.':'Some rejected plants can receive validated model-predicted gains.'} This does not prove the model beats a global optimizer: the teacher uses local search and finite numerical resources. Unresolved outcomes remain unresolved.`;
  }
  curve('hero-response', study.cases[0], 'standard_mlp', true);
  $('hero-caption').textContent = `Final test row ${study.cases[0].sourceRow} · model: MLP · unit setpoint · shorter display window.`;
  renderAcceptance(); renderLearning(); renderSpeed(); renderModel();
  window.addEventListener('resize', () => {
    const next = window.matchMedia('(max-width:600px)').matches;
    if (next === compact) return;
    compact = next;
    curve('hero-response', study.cases[0], 'standard_mlp', true);
    renderAcceptance(); renderLearning(); renderSpeed(); renderModel();
  });
})();


/* ===== Live prediction interface ===== */
/* User-entered measurements → selected frozen model → independent plant checks. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id), form=$('prediction-form');
  if (!form || !window.PID_MODEL) return;
  const features=window.PID_MODEL.features, physical=['K','wn','zeta','tau'];
  const modelNames={standard_mlp:'MLP',random_forest:'Random Forest',xgboost:'XGBoost'};
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
    ['prediction-model','prediction-mode','prediction-example','prediction-known','prediction-run'].forEach(id=>{$(id).disabled=busy;});
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
    status('Example values loaded. Edit them or predict using the selected frozen model.');
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
    worker.onmessage=async ({data})=>{
      if(!busy || !current) return;
      if(data.type==='status')status(data.message);
      else if(data.type==='error')failure(data.message);
      else if(data.type==='features') {
        current.features=data.result.features;
        features.forEach(k=>{field(k).value=fmt(current.features[k],9);});
        try {if(await runNative())verify();} catch(error) {if(busy)failure(error.message||String(error));}
      } else if(data.type==='complete') finish(data.result);
    };
    return worker;
  }
  function send(action) {
    getWorker().postMessage({action,source:window.PID_GENERATOR.source,assessor:window.PID_MODEL.assessorSource,
      diagnostic:window.PID_SUPPLEMENT?.diagnosticSource,
      plant:current.plant,features:current.features,gains:current.gains,compare:current.compare});
  }
  async function runNative() {
    const request=current;
    let predictor=null;
    if(request.model!=='standard_mlp'){
      try {predictor=await window.PIDBaseline.load(request.model,status);}
      catch(error){if(!busy||current!==request)return false;throw error;}
    }
    if(!busy||current!==request)return false;
    const start=performance.now();
    current.gains=predictor?window.PIDBaseline.decode(predictor(features.map(k=>current.features[k])),features.map(k=>current.features[k])):window.PIDPredict(current.features,window.PID_MODEL);
    current.inferenceMs=performance.now()-start;
    if(current.gains.some(v=>!Number.isFinite(v)))throw new Error('Nonfinite prediction; use finite inputs within the study scope.');
    $('prediction-gains').innerHTML=current.gains.map((v,i)=>`<div><strong>${fmt(v,6)}</strong><span>K<sub>${['p','i','d'][i]}</sub></span></div>`).join('');
    const outside=features.filter(k=>{const [a,b]=window.PID_MODEL.featureRanges[k];return current.features[k]<a||current.features[k]>b;});
    current.warning=outside.length?' Some feature values lie outside the training range: this is extrapolation.':'';
    $('prediction-features-note').textContent=`${modelNames[current.model]} · local inference ${fmt(current.inferenceMs)} ms (model download excluded). Used rise ${fmt(current.features.t_rise)} s, settling ${fmt(current.features.t_settling)} s, peak ${fmt(current.features.y_peak)}, final ${fmt(current.features.y_ss)}, onset ${fmt(current.features.tau_est)} s.${current.warning}`;
    return true;
  }
  function verify() {
    status('Gains predicted. Independently checking the known physical plant…');
    send('verify');
  }
  const outcome=m=>m.accepted?'Meets strict checks':({overshoot:'Outside 10% overshoot criterion',invalid_gains:'Outside gain bounds',unstable:'Unstable loop',resolution_limit:'Unresolved: grid cap',duration_limit:'Unresolved: horizon cap'}[m.reason]||`Not accepted: ${m.reason||'no validated numerical result'}`);
  function finish(result) {
    busy=false;configure();
    const m=result.metrics,r=result.referenceMetrics;
    $('prediction-response').hidden=false;
    const rows=[[`Predicted ${modelNames[current.model]}`,m],...(r?[['Numerical search',r]]:[])];
    $('prediction-metrics').innerHTML=`<div class="study-table-wrap"><table class="study-table"><caption>Current plant, unit setpoint, independent checks</caption><thead><tr><th>Controller</th><th>Specification</th><th>Overshoot</th><th>Converged ITAE</th><th>Converged J</th></tr></thead><tbody>${rows.map(([name,v])=>`<tr><td>${name}</td><td>${outcome(v)}</td><td>${v.overshoot_pct==null?'—':fmt(v.overshoot_pct)+'%'}</td><td>${v.accepted?fmt(v.itae,6):v.response_resolved?fmt(v.diagnostic_itae,6):'—'}</td><td>${v.accepted?fmt(v.objective,6):v.response_resolved?fmt(v.diagnostic_objective,6):'—'}</td></tr>`).join('')}</tbody></table></div>`;
    if(!m.accepted && m.response_resolved){
      const note=document.createElement('p');note.className='chart-caption';
      note.textContent=`This controller misses the strict overshoot specification but has a stable, converged response. Its 2% settling time is ${fmt(m.diagnostic_settling)} s. The ITAE and cost above describe its performance; they do not change the 10% acceptance result.`;
      $('prediction-metrics').append(note);
    }
    if(!m.accepted && !m.response_resolved && m.display_itae!=null) {
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
  form.addEventListener('submit',async event=>{
    event.preventDefault();if(busy)return;
    try {
      if(!form.reportValidity())return;
      current={model:$('prediction-model').value,plant:knowsPlant()?Object.fromEntries(physical.map(k=>[k,field(k).valueAsNumber])):null,
        features:Object.fromEntries(features.map(k=>[k,field(k).valueAsNumber])),compare:knowsPlant()&&$('prediction-compare').checked};
      busy=true;configure();
      if($('prediction-mode').value==='plant') {status('Measuring the open-loop response to derive five features…');send('observe');}
      else {if(!await runNative())return;if(current.plant)verify();else{busy=false;configure();status(`${modelNames[current.model]} gains predicted locally. Supply a known plant to simulate and verify the response.${current.warning}`);}}
    } catch(error) {failure(error.message||String(error));}
  });
  $('prediction-cancel').addEventListener('click',()=>{worker?.terminate();worker=null;busy=false;configure();invalidate();status('Run canceled. No new controller was accepted.');});
  ['prediction-mode','prediction-known','prediction-model'].forEach(id=>$(id).addEventListener('change',()=>{invalidate();configure();}));
  form.querySelectorAll('input[type=number],#prediction-compare').forEach(input=>input.addEventListener('input',invalidate));
  window.PID_STUDY.cases.forEach((c,i)=>$('prediction-example').add(new Option(c.name,String(i))));
  $('prediction-example').addEventListener('change',()=>{if($('prediction-example').value!=='')preset(Number($('prediction-example').value));});
  window.addEventListener('resize',()=>{if(live)window.PID_RENDER_RESPONSE('prediction-chart',live);});
  preset(0);$('prediction-example').value='0';configure();
})();
