"""Audit and summarize website-only development evidence without refitting."""
import hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'results/dataset_final_10000'
OUT=FINAL/'website_extension'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(frame):
    return frame.to_csv(index=True)  # Raw values also remain in dedicated CSVs.

protocol=json.loads((FINAL/'evaluation_protocol.json').read_text(encoding='utf-8'))
lock=json.loads((FINAL/'selection_lock.json').read_text(encoding='utf-8'))['files']
assert all(sha(FINAL/name)==value for name,value in lock.items())
assert sha(ROOT/protocol['data_path'])==protocol['dataset_sha256']
curve=pd.read_csv(OUT/'learning_curves.csv')
assert len(curve)==42 and curve.plants.eq(500).all()
assert set(curve.training_size)=={1000,3000,5000,7000,10000,15000,20000}
pool=pd.read_csv(OUT/'training_pool.csv')
assert len(pool)==27000 and set(pool.iloc[:7000].original_row)==set(protocol['train'])
expected={len(pool)+protocol['validation'].index(i) for i in protocol['curve_validation']}
for path in OUT.glob('*_learning*_validation.csv'):
    frame=pd.read_csv(path)
    assert len(frame)==500 and set(frame.source_row)==expected,path.name
stress=pd.read_csv(OUT/'rejected_predictions.csv')
assert len(stress)==6*83 and stress.groupby('model').rejection_id.nunique().eq(83).all()
summary=pd.read_csv(OUT/'rejected_summary.csv').set_index('model')
mlp=stress[stress.model.eq('standard_mlp')]
timing=pd.read_csv(FINAL/'tuning_speed/summary.csv').set_index('model')
opts=pd.read_csv(FINAL/'tuning_speed/optimizer.csv')
assert len(opts)==20 and set(opts.source_row)<=set(protocol['validation'])
audit={'final_model_selection_and_protocol_hashes_unchanged':True,'final_dataset_hash_unchanged':True,
    'same_500_validation_plants_verified_for_all_extended_points':True,'final_test_used_for_extension':False,
    'final_dataset_sha256':protocol['dataset_sha256'],'development_dataset_sha256':sha(ROOT/'data/pid_dataset_20000.csv'),
    'training_points':len(curve),'rejected_plants':83,'mlp_expanded_outcomes':mlp.extended_reason.value_counts().to_dict(),
    'timing_plants_from_validation':20,'browser_parity':json.loads((FINAL/'browser_export_verification.json').read_text(encoding='utf-8'))}
(OUT/'evidence_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
points=curve.pivot(index='model',columns='training_size',values='success_pct')
r=summary.loc['standard_mlp'];t=timing.loc['standard_mlp']
text=f'''# Final website: additional development evidence

## Learning curves

All 42 points use the same 500 final-validation plants. The first 7,000 training
plants are the frozen final-training cohort (nested seed 45). Larger prefixes add
independent prior-development plants (seed 46); 20,000 training plants uses 13,000
additional development plants. Settings and early stopping rules remain fixed.
The final test partition is not used. Extended fits do not replace final models.

The axis counts TRAINING plants. The chosen 10,000-row final dataset uses 7,000
training plants. MLP acceptance rises 84.6% at 1K, 95.2% at 7K, and 99.6% at 20K.
The 4.4 percentage-point gain from 7K to 20K is real in this validation sample;
10K is a resource/scope tradeoff, not a proven plateau. Gains per added plant are
smaller on average than the initial 1K-to-3K gain. These single-seed development
curves can be optimistic because validation also informs settings and stopping.

```csv
{table(points)}
```

## Rejected-plant stress set

83 rejected final-generation plants are separate from the accepted test cohort.
Original reasons: 57 resolution-only, 25 overshoot-only, 1 mixed. Frozen final
models predict gains without numerical optimization or retraining.

MLP: {int(r.strict_passes)}/83 strict passes; {int(r.extended_passes)}/83 passes after
resolution-only retries with a 262,145-point cap (versus 65,537 initially).
All gains, tolerances, bounds and the horizon cap are unchanged. Expanded passes
must not be counted as strict passes. Remaining outcomes: {audit['mlp_expanded_outcomes']}.
There are no accepted reference gains for these plants; gain error and teacher-
relative cost cannot be reported. Rejection is not proof of uncontrollability.

```csv
{table(summary)}
```

## Computational edge

20 fixed validation plants, warm CPU. Numerical optimization plus validation:
median {opts.optimizer_seconds.median():.4f} s. MLP prediction: median
{t.median_prediction_ms:.5f} ms; median matched speedup
{t.median_prediction_speedup:.1f}x. Prediction plus verification:
{t.median_prediction_plus_validation_ms:.3f} ms, median matched speedup
{t.median_verified_speedup:.2f}x on jointly accepted pairs ({int(t.verified_pairs)}/20).
Feature collection, loading and training are excluded. This sample is for timing,
not a new accuracy estimate. ML amortizes expensive offline labeling but does not
guarantee feasibility or remove the need for physical verification.

## Portable live model

The JavaScript explorer uses the frozen 7K-trained final MLP. Export parity is
checked on 64 validation examples. Physical verification and fresh numerical
comparisons run the same exported Python classes/evaluator in a Pyodide worker.
Feature-only mode predicts gains; a supplied plant is required for simulation.
Inputs stay local. All frozen-model/data hashes remain unchanged.
'''
(OUT/'report.md').write_text(text,encoding='utf-8')
print(json.dumps(audit,indent=2))
