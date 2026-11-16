/* Check exported browser inference against the frozen PyTorch validation gains. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const root = path.resolve(__dirname, '..');
const context = {window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'docs/model-data.js'),'utf8'),context);
const {predict} = require(path.join(root,'docs/model-inference.js'));
const directory = path.join(root,'results/dataset_final_10000');
const fixture = JSON.parse(fs.readFileSync(path.join(directory,'browser_parity_fixtures.json'),'utf8'));
let maxAbsolute=0,maxRelative=0;
fixture.features.forEach((features,i)=>predict(features,context.window.PID_MODEL).forEach((value,j)=>{
  const expected=fixture.gains[i][j], difference=Math.abs(value-expected);
  maxAbsolute=Math.max(maxAbsolute,difference);
  maxRelative=Math.max(maxRelative,difference/Math.max(1e-6,Math.abs(expected)));
  if(difference>1e-5+1e-5*Math.abs(expected)) throw Error('Portable MLP inference parity failed.');
}));
const result={examples:fixture.features.length,maxAbsolute,maxRelative,modelSha256:context.window.PID_MODEL.sha256};
fs.writeFileSync(path.join(directory,'browser_export_verification.json'),JSON.stringify(result,null,2));
console.log(result);
