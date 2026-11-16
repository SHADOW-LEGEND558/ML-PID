# Computational complexity and measured runtime

Based on the frozen 20,000-row experiment: 15,000 training / 2,500 validation / 2,500 test plants, five empirical features, three PID outputs. All models use normalized targets. CPU: Intel Core i5-13450HX. No models were retrained or changed for this cost analysis.

## Measured costs

Training times below are the original selected-candidate fit times, not new controlled training benchmarks. They exclude closed-loop simulation, other candidates, and learning-curve fits. MLP fit includes per-epoch validation loss and early stopping; regression/tree fits do not include simulation-based selection. Thread settings and warm-up differ by estimator, so small fit-time differences should not be ranked strictly.

Prediction is a fresh warm benchmark: five single-input warm-ups, three batch warm-ups, 50 single-input calls, and 10 calls on the same 2,500-row batch. Times include preprocessing/normalization wrappers and output decoding. Inputs are already extracted features; model loading, response measurement, feature extraction and closed-loop validation are excluded. No parameter or accuracy selection used these measurements. RF uses six fit/predict threads, XGBoost four threads, PyTorch two threads, BLAS/OpenMP environment one thread. OS scheduling and Python overhead influence results.

| Model | Recorded fit (s) | Single median (ms) | Single p95 (ms) | Batch amortized (µs/plant) | Saved artifact (MiB) |
|---|---:|---:|---:|---:|---:|
| linear | 0.0080 | 0.4834 | 0.5825 | 0.212 | 0.0616 |
| ridge | 0.0061 | 0.4845 | 0.5158 | 0.215 | 0.0615 |
| polynomial_ridge | 0.0131 | 0.5700 | 0.6830 | 0.399 | 0.0625 |
| random_forest | 1.1863 | 25.3483 | 47.5301 | 14.155 | 318.4980 |
| xgboost | 1.0959 | 1.8846 | 2.3487 | 12.758 | 4.7791 |
| standard_mlp | 48.1108 | 0.1403 | 0.2663 | 0.273 | 0.1067 |

Batch-amortized time is throughput expressed per plant, not single-request latency. For example, the MLP averages about 0.273 µs/plant in a batch, but one normal single-request prediction takes about 0.140 ms with the current wrapper. Artifact sizes include weights, preprocessing and experiment/split metadata; they are not pure model-weight memory or peak process RAM. Metadata dominates the small linear files.

## Complexity notation

N: training plants (15,000); d: input features (5); m: outputs (3); p: expanded features (20); T: trees per ensemble; H: tree depth; L: leaves per tree; E: epochs; P: neural parameters (11,203). The estimates below are rough dense-data work counts, not wall-clock guarantees. They assume fixed output count; tree costs depend on split strategy, depth, histogram bins and implementation.

| Model | Approximate training work | Prediction for one plant | Learned core |
|---|---|---|---|
| Linear regression | O(N d² + N d m), for N >= d | O(d m) | 15 coefficients + 3 intercepts |
| Ridge | O(N d² + d³ + N d m) | O(d m) | 15 coefficients + 3 intercepts |
| Degree-two polynomial ridge | O(N p² + p³ + N p m), plus expansion | O(p + p m), fixed degree | 60 coefficients + 3 intercepts |
| Random forest | Roughly O(T d N log N), approximately balanced efficient trees | O(T(H+m)) | 200 joint-output trees |
| XGBoost histogram trees | Rough data scanning O(m T N d H), plus bin/split construction and quantization | O(m T H) | 3 independent ensembles x 400 trees, depth <=6 |
| Standard MLP | O(E N P), plus O(E N_val P) validation forwards | O(P) | 11,203 parameters; fixed architecture |

Polynomial expansion gives p = binomial(d+2,2)-1 = 20. This is why the degree-two model remains small here, while higher degree/more inputs can cause a combinatorial increase in expanded features. Standard scaling and target decoding add small O(d+m) per-input work.

The forest estimate assumes roughly balanced trees and efficient split processing. Tree inference is proportional to path depth, not the entire stored node count; a highly unbalanced tree can have depth O(N). [scikit-learn tree complexity](https://scikit-learn.org/stable/modules/tree.html#complexity).

XGBoost hist performs a quantile sketch and uses per-node histograms, so full training cost is not just the scanning term in the table. For fixed tree count, depth and bin count, the dominant sample-processing work scales roughly linearly with N. Boosting rounds depend sequentially on earlier rounds, although split/histogram operations can be parallelized. [XGBoost tree methods](https://xgboost.readthedocs.io/en/stable/treemethod.html).

## Actual structures and memory

- Linear/ridge predict all three gains using 18 fitted coefficients/intercepts, plus preprocessing.
- Polynomial ridge has 63 fitted coefficients/intercepts, plus expansion/scaling.
- Forest: 200 trees, 3,793,626 nodes, 1,896,913 leaves. Mean tree maximum depth 26.16; largest depth 33. Its 318.5 MiB saved size mainly reflects the deep, unconstrained trees.
- XGBoost: three output-specific models, each with 400 trees and maximum depth 6: 1,200 trees total. It is much smaller than the forest because the trees are shallow.
- MLP: 5 -> 128 -> 64 -> 32 -> 3, with 11,203 parameters and 10,976 weights. A forward pass needs roughly 10,976 multiply-accumulates plus biases and activations. Float32 parameter values alone occupy about 43.8 KiB. The complete saved bundle is about 109.3 KiB.
- The MLP executed 463 epochs and restored epoch 413. Training repeatedly performs forward/backward passes; prediction performs one forward pass.

Inference model storage is approximately O(d m) for linear/ridge, O(p m) for polynomial ridge, O(T L m) for the joint-output forest, O(m T L) for the separate boosting ensembles, and O(P) for the MLP. Tree split metadata is additional. MLP training also holds gradients, Adam optimizer state, mini-batch activations and input data; training RAM is larger than inference weights.

## What changes as the dataset grows?

1. With feature count fixed, linear/ridge/polynomial training grows roughly linearly in N. Their final coefficient count and inference cost do not grow with N.
2. Unrestricted forests grow more leaves/nodes as N increases. Model storage can grow roughly linearly with N, and prediction paths tend to get deeper. Fixed tree depth/leaf limits can control this.
3. With XGBoost tree count, depth and bins fixed, training sample work grows roughly linearly with N; prediction cost and model-size upper bounds stay approximately fixed. Increasing the number or depth of trees changes those bounds.
4. With a fixed MLP architecture and fixed epochs, training is roughly linear in N. Final parameter count and single-input forward work remain fixed. Early stopping can change the epoch count, so measured training time need not scale exactly linearly.
5. Doubling every hidden width roughly quadruples hidden-to-hidden matrix work; adding layers adds matrix operations. A larger training dataset does not itself make the saved MLP heavier.

## Interpretation for this project

The linear models are cheapest, but their controller pass rates are only around 28–44%. Forest and XGBoost train quickly and perform well, but the forest is much larger and its current multithreaded wrapper has substantial one-request overhead. The MLP costs more offline training time (about 48 s for this selected fit) yet combines 97.92% test full-validation success with small model storage and quick inference.

The MLP is not theoretically cheaper than linear regression: its observed single-call advantage is implementation/wrapper overhead and efficient dense operations. Batch results show linear/ridge arithmetic remains slightly cheaper. These timings should not be treated as universal hardware-independent rankings. RF single-request latency may improve with a different threading configuration, which was not tuned here.

## Model prediction versus controller validation

All six prediction operations are fast compared with an optimization/simulation pipeline. Closed-loop validation is a separate adaptive cost: pole calculation, state-space response samples, peak refinement, and repeated grids/horizons. For our fixed low-order plant/controller family, matrix order is small; the dominant practical variation comes from time-sample counts, peak refinements and horizon extensions. Hard/near-boundary plants can cost much more than easy plants.

Numerical label generation additionally evaluates many trial controllers across starts and SLSQP iterations. ML inference avoids that optimization, but does not remove the cost of any independent validation one chooses to perform. End-to-end tuning also needs the measured open-loop response; none of the prediction timings include waiting for the physical experiment.

## Reproduce

Run `python scripts/benchmark_model_cost.py` from the project root. It loads frozen artifacts and times prediction only; it does not retrain or change the final model settings. Raw results are in timings.csv and structure.json.
