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
    $('case-note').textContent = `Saved test row ${item.sourceRow}. K ${fmt(plant.K, 3)} · ωₙ ${fmt(plant.wn, 3)} rad/s · ζ ${fmt(plant.zeta, 3)} · τ ${fmt(plant.tau, 3)} s. Predicted Kp / Ki / Kd: ${p.gains.map(v => fmt(v, 4)).join(' / ')}. Outcome: ${p.reason.replaceAll('_', ' ')}. Raw validated ITAE: model ${fmt(p.itae, 4)}; numerical reference ${fmt(p.referenceItae, 4)}.`;
    $('case-readouts').innerHTML = readout(p.accepted ? 'Accepted' : 'Not accepted', 'FULL NUMERICAL VALIDATION') + readout(`${fmt(p.overshoot)}%`, 'VALIDATED OVERSHOOT') + readout(p.itaeRatio === null ? '—' : `${fmt(p.itaeRatio, 3)}×`, 'ITAE / REFERENCE IF ACCEPTED');
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
    $('model-readouts').innerHTML = readout(`${fmt(m.metrics.success_pct)}%`, `FULL ACCEPTANCE / ${N.toLocaleString()} TEST PLANTS`) + readout(`${fmt(m.metrics.median_ITAE_ratio, 3)}×`, 'MEDIAN ITAE / ACCEPTED PAIRS') + readout(`${N - m.metrics.passes}`, 'CONTROLLERS NOT ACCEPTED');
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
