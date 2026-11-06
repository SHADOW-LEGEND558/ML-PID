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
  let selected = document.querySelector('main > section:not([hidden])')?.id || 'basics';
  if (location.hash === '#data-generation') selected = 'data-generation';
  if (location.hash === '#basics') selected = 'basics';
  document.querySelectorAll('main > section').forEach(section => {section.hidden = section.id !== selected;});
  document.querySelectorAll('.sidebar nav a').forEach(link => {
    const active = link.hash === `#${selected}`;
    link.classList.toggle('active',active);
    if (active) link.setAttribute('aria-current','page'); else link.removeAttribute('aria-current');
  });
  document.title = `ML-PID · ${selected === 'basics' ? 'System basics' : 'Data generation'}`;
  document.querySelector('footer a').setAttribute('href', `#${selected}`);
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
