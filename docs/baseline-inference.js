/* Exact frozen-tree traversal. Models download only when selected. */
(function(root){
  'use strict';
  function forest(buffer){
    const view=new DataView(buffer);let offset=8;
    const uint=()=>{const n=view.getUint32(offset,true);offset+=4;return n;};
    const count=uint(),trees=[];
    for(let k=0;k<count;k++){
      const n=uint(),leaves=uint();
      const feature=offset;offset+=n;
      const threshold=offset;offset+=8*n;
      const left=offset;offset+=4*n;
      const right=offset;offset+=4*n;
      const leaf=offset;offset+=4*n;
      const values=offset;offset+=24*leaves;
      trees.push({feature,threshold,left,right,leaf,values});
    }
    if(offset!==buffer.byteLength)throw new Error('Incomplete random-forest model.');
    return input=>{
      const x=input.map(Math.fround),q=[0,0,0];
      for(const t of trees){
        let i=0;
        while(view.getInt32(t.left+4*i,true)>=0){
          const f=view.getInt8(t.feature+i),cut=view.getFloat64(t.threshold+8*i,true);
          i=view.getInt32((x[f]<=cut?t.left:t.right)+4*i,true);
        }
        const leaf=view.getInt32(t.leaf+4*i,true);
        for(let j=0;j<3;j++)q[j]+=view.getFloat64(t.values+24*leaf+8*j,true)/count;
      }
      return q;
    };
  }
  function boosted(model){
    return input=>{
      const x=input.map(Math.fround);
      return model.targets.map(target=>{
        let value=Math.fround(target.base);
        for(const t of target.trees){
          let i=0;
          while(t.left_children[i]>=0)i=x[t.split_indices[i]]<Math.fround(t.split_conditions[i])?t.left_children[i]:t.right_children[i];
          value=Math.fround(value+Math.fround(t.split_conditions[i]));
        }
        return value;
      });
    };
  }
  function decode(q,x){return [q[0]/x[3],q[1]/(x[3]*x[0]),q[2]*x[0]/x[3]];}
  const cache=new Map();
  async function load(name,onStatus=()=>{}){
    if(cache.has(name))return cache.get(name);
    const meta=root.PID_BASELINES[name];
    onStatus(`Loading ${name==='random_forest'?'Random Forest':'XGBoost'} comparison baseline (${(meta.bytes/1048576).toFixed(1)} MB, first use only)…`);
    const response=await fetch(new URL(meta.url,document.baseURI));
    if(!response.ok)throw new Error(`Baseline download failed (${response.status}).`);
    let buffer=await response.arrayBuffer();
    const first=new Uint8Array(buffer);
    if(first[0]===31&&first[1]===139){
      if(typeof DecompressionStream==='undefined')throw new Error('This browser cannot unpack the comparison model. Use MLP or a current browser.');
      buffer=await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    }
    const predictor=name==='random_forest'?forest(buffer):boosted(JSON.parse(new TextDecoder().decode(buffer)));
    cache.set(name,predictor);return predictor;
  }
  async function predict(name,features,onStatus){
    const x=root.PID_MODEL.features.map(k=>Number(features[k]));
    if(x.some(v=>!Number.isFinite(v))||x[0]<=0||x[3]<=0)throw new Error('Provide finite features with positive rise time and final output.');
    const predictor=await load(name,onStatus);
    return decode(predictor(x),x);
  }
  root.PIDBaseline={predict,load,forest,boosted,decode};
  if(typeof module!=='undefined')module.exports={forest,boosted,decode};
})(typeof window!=='undefined'?window:globalThis);
