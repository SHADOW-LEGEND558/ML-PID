/* Frozen 5→128→64→32→3 MLP, evaluated locally. No runtime training. */
(function(root) {
  'use strict';
  function predict(features, model) {
    const input = Array.isArray(features) ? features : model.features.map(k => Number(features[k]));
    if (input.length !== 5 || input.some(v => !Number.isFinite(v)) || input[0] <= 0 || input[3] <= 0) throw new Error('Provide five finite features, with positive rise time and final output.');
    let x = input.map((v,i) => (Math.fround(v) - model.mean[i]) / model.std[i]);
    model.layers.forEach((layer,index) => {
      x = layer.weights.map((row,j) => {
        const value = row.reduce((sum,w,k) => sum + w*x[k], layer.bias[j]);
        return index < model.layers.length-1 ? value/(1+Math.exp(-value)) : Math.max(value,0)+Math.log1p(Math.exp(-Math.abs(value)));
      });
    });
    const factors = [input[3], input[3]*input[0], input[3]/input[0]];
    return x.map((v,i) => v*model.qscale[i]/factors[i]);
  }
  root.PIDPredict = predict;
  if (typeof module !== 'undefined') module.exports = {predict};
})(typeof window !== 'undefined' ? window : globalThis);
