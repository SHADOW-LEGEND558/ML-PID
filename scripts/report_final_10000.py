"""Summarize the final 10K study and plot saved predictions without fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dataset_final_10000"
sys.path.insert(0, str(ROOT / "src"))
from plot_baselines import (LABELS, style, performance_plot, prediction_plot, failure_plot,
    objective_plot, itae_plot, overshoot_plot, overshoot_bins_plot, overshoot_summary)

NAMES = ["linear", "ridge", "polynomial_ridge", "random_forest", "xgboost", "standard_mlp"]
LABELS["standard_mlp"] = "MLP"


def load_final(names=None):
    names = NAMES if names is None else names
    metrics = pd.read_csv(OUT / "test_metrics.csv").set_index("model").loc[names].copy()
    metrics["validated_success_pct"] = metrics.success_pct
    frames = {name: pd.read_csv(OUT / f"{name}_test_predictions.csv").sort_values("source_row") for name in names}
    protocol = json.loads((OUT / "evaluation_protocol.json").read_text(encoding="utf-8"))
    for frame in frames.values():
        assert len(frame) == 1500 and set(frame.source_row) == set(protocol["test"])
    return metrics, frames, protocol


def learning_plot(names=None):
    names = NAMES if names is None else names
    curves = pd.read_csv(OUT / "learning_curves.csv")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for name in names:
        group = curves[curves.model.eq(name)].sort_values("training_size")
        ax.plot(group.training_size, group.success_pct, "o-", label=LABELS[name])
    ax.set(xlabel="Training plants", ylabel="Validation acceptance (%)", ylim=(0,100),
           title="Final cohort: fixed 500 validation plants, selected settings fixed")
    ax.legend(); return fig


def history_plot():
    selected = pd.read_csv(OUT / "selected_before_test.csv")
    row = selected[selected.model.eq("standard_mlp")].iloc[0]
    history = pd.read_csv(OUT / f"{row.candidate}_history.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(history.epoch, history.train_loss, label="Train")
    ax.plot(history.epoch, history.val_loss, label="Validation")
    ax.axvline(history.loc[history.val_loss.idxmin(), "epoch"], color="grey", ls=":", label="Best validation epoch")
    ax.set(xlabel="Epoch", ylabel="Standardized normalized-target log loss", yscale="log",
           title="Final-cohort MLP: training and early-stopping loss")
    ax.legend(); return fig


def markdown_table(frame, columns):
    result = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        result.append("| " + " | ".join(f"{row[k]:.4f}" if isinstance(row[k], (float,np.floating)) else str(row[k]) for k in columns) + " |")
    return result


def report():
    metrics, predictions, protocol = load_final()
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert verification["all_locked_artifact_hashes_unchanged"]
    style()
    figures = {"performance": performance_plot(metrics), "gain_predictions": prediction_plot(predictions),
        "failure_outcomes": failure_plot(predictions), "objective_ratios": objective_plot(predictions),
        "itae_comparison": itae_plot(predictions), "overshoot": overshoot_plot(predictions),
        "overshoot_bins": overshoot_bins_plot(predictions), "learning_curves": learning_plot(), "mlp_history": history_plot()}
    plotdir = OUT / "plots"; plotdir.mkdir(exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(plotdir / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    over = overshoot_summary(predictions)
    over.to_csv(OUT / "overshoot_summary.csv")
    tracking, failures, groups, intervals = [], [], [], []
    for name, frame in predictions.items():
        ratios = frame.loc[frame.accepted & frame.reference_accepted, "itae_ratio"].dropna()
        tracking.append({"model": name, "pairs": len(ratios), "median_ITAE_ratio": ratios.median(),
            "median_change_pct": 100*(ratios.median()-1), "p90_ITAE_ratio": ratios.quantile(.9),
            "p95_ITAE_ratio": ratios.quantile(.95), "max_ITAE_ratio": ratios.max(), "ITAE_over_2x": int(ratios.gt(2).sum())})
        failures += [{"model": name, "reason": reason, "plants": int(count)} for reason,count in frame.reason.value_counts().items()]
        n = len(frame); p = float(frame.accepted.mean()); z = 1.95996398454
        center = (p + z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        intervals.append({"model":name,"acceptance_pct":100*p,"Wilson95_low":100*(center-half),"Wilson95_high":100*(center+half)})
        bins = [("zero delay",frame.tau.eq(0)), ("0 < delay <= 0.1",frame.tau.gt(0)&frame.tau.le(.1)),
                ("0.1 < delay <= 0.5",frame.tau.gt(.1)&frame.tau.le(.5)), ("delay > 0.5",frame.tau.gt(.5)),
                ("zeta < 0.5",frame.zeta.lt(.5)), ("0.5 <= zeta < 1",frame.zeta.ge(.5)&frame.zeta.lt(1)), ("zeta >= 1",frame.zeta.ge(1))]
        groups += [{"model":name,"group":label,"plants":int(mask.sum()),"passes":int(frame.loc[mask,"accepted"].sum()),
                    "success_pct":100*float(frame.loc[mask,"accepted"].mean())} for label,mask in bins]
    tracking = pd.DataFrame(tracking); failures = pd.DataFrame(failures); groups = pd.DataFrame(groups); intervals = pd.DataFrame(intervals)
    for filename, table in [("itae_summary",tracking),("failure_counts",failures),("group_results",groups),("acceptance_intervals",intervals)]:
        table.to_csv(OUT / f"{filename}.csv", index=False)
    gains = pd.DataFrame([{"model":name,"gain":g,"MAE":row[f"{g}_mae"],"RMSE":row[f"{g}_rmse"],"R2":row[f"{g}_r2"]}
                          for name,row in metrics.iterrows() for g in ["Kp","Ki","Kd"]])
    gains.to_csv(OUT / "gain_errors.csv", index=False)
    selected = pd.read_csv(OUT / "selected_before_test.csv")
    curves = pd.read_csv(OUT / "learning_curves.csv").sort_values(["model","training_size"])
    old = pd.read_csv(ROOT / "results/dataset_20000/test_metrics.csv").set_index("model")
    comparison = pd.DataFrame({"20K_development_test_pct":old.success_pct,"10K_final_test_pct":metrics.success_pct})
    comparison.to_csv(OUT / "development_comparison.csv")
    lines = ["# Final 10,000-plant evaluation", "", "## Protocol", "",
        f"Dataset: {protocol['data_path']}. SHA-256: {protocol['dataset_sha256']}.", "",
        "Fresh independent cohort; exactly 7,000 training / 1,500 validation / 1,500 test plants. The prepared manifest is reused unchanged (seeds 42/43; zero/positive delay stratification). There are 719 zero-delay plants in training, 154 in validation and 154 in test. All six models share these rows.", "",
        "All models use only t_rise, t_settling, y_peak, y_ss and tau_est. Empirical-normalized targets are q=[y_ss*Kp, y_ss*t_rise*Ki, y_ss*Kd/t_rise]. Preprocessing is fitted only on training plants. True K, wn, zeta and tau enter controller simulation only.", "",
        "Linear, ridge alpha=100 and degree-two ridge alpha=1 retain the development configurations. Random forest compares 200 trees with (depth 12, leaf 3) versus (unlimited depth, leaf 1). XGBoost compares 150 versus 400 trees per target at depth 6, learning rate 0.08. Standard MLP is 5-128-64-32-3, SiLU, dropout 0.05, positive normalized outputs, standardized log loss, AdamW learning rate 0.001/weight decay 0.0001, batch 256, seed 42, max 500 epochs and patience 50; physics loss is disabled.", "",
        "Select settings by validation full-controller pass count, then median accepted J ratio. MLP early stopping uses validation supervised loss. Saved weights, settings and protocol hashes are locked before test scoring. No refit on validation labels, prediction clipping, fallback controller or post-test retuning is used.", "",
        "Reference and predicted controllers undergo the same independent checks: finite bounded gains, stable poles, refined overshoot <=10%, tail error <=0.001 and converged grids/horizons. J=wn^2*ITAE+0.01*((K*Kp)^2+(K*Ki/wn)^2+(K*wn*Kd)^2). Validation refers to simulated ideal PID with second-order Pade delay, not hardware safety.", "", "## Held-out results", ""]
    lines += markdown_table(metrics.reset_index(), ["model","passes","plants","success_pct","stable_pct","r2","median_ITAE_ratio","median_J_ratio","ratio_pairs"])
    best = metrics.success_pct.idxmax()
    best_row = metrics.loc[best]
    lines += ["", "## Main findings", "",
        f"{LABELS[best]} has the highest held-out acceptance: {int(best_row.passes)}/1500 ({best_row.success_pct:.2f}%). "
        f"Its median accepted-pair ITAE increase is {100*(best_row.median_ITAE_ratio-1):.2f}%, and median J increase is {100*(best_row.median_J_ratio-1):.2f}%.", ""]
    if "standard_mlp" in predictions:
        mf = predictions["standard_mlp"]
        counts = mf.reason.value_counts().to_dict()
        bins = over.loc["standard_mlp"]
        lines += [f"MLP outcomes: {counts}. All {int(mf.stable.sum())}/1500 are pole-stable; full acceptance remains a stricter test. "
            f"Overshoot misses split into {int(bins.over_10_to_12)} between 10-12%, {int(bins.over_12_to_15)} between 12-15%, "
            f"{int(bins.over_15_to_20)} between 15-20%, and {int(bins.over_20)} above 20%; maximum measured overshoot is {bins.max_overshoot_pct:.2f}%.", ""]
    lines += ["", "Ratios are conditional on both reference and prediction passing, with coverage shown. Failed controllers remain in the 1,500-plant acceptance denominator and are never assigned zero cost.", "", "## Per-gain errors (all test plants)", ""] + markdown_table(gains,list(gains.columns))
    lines += ["", "## ITAE tail analysis", ""] + markdown_table(tracking,list(tracking.columns))
    lines += ["", "Acceptance does not bound ITAE regret. Raw ITAE can improve while regularized J worsens; labels are best-found local numerical solutions, not global optima.", "", "## Overshoot severity", ""]
    lines += markdown_table(over.reset_index(), ["model","at_most_10","over_10_to_12","over_12_to_15","over_15_to_20","over_20","stable_unmeasured","unstable"])
    lines += ["", "Below-limit overshoot alone is not full acceptance. Unresolved numerical checks and unstable controllers remain separate.", "", "## Validation-only selections", ""] + markdown_table(selected,["model","setting","success_pct","median_J_ratio"])
    lines += ["", "## Learning curves", "", "Fixed selected settings; shared nested 1,000/3,000/5,000/7,000 training subsets and the same 500 stratified validation plants. MLP stopping uses the full validation set. These are development diagnostics, not repeated test evaluations or independent estimates of hyperparameter-selection performance.", ""]
    lines += markdown_table(curves,["model","training_size","success_pct","median_J_ratio"])
    lines += ["", "## Error analysis by plant group", "", "Descriptive post-test breakdown; no settings are changed using these results. Group counts matter; physical parameters are audit metadata, not model inputs.", ""] + markdown_table(groups,list(groups.columns))
    lines += ["", "## Primary failure outcomes", ""] + markdown_table(failures,list(failures.columns))
    lines += ["", "## Sampling uncertainty", "", "95% Wilson intervals describe finite-test uncertainty within this accepted-label simulation population. They do not cover seed variability, model mismatch, selection bias, or physical robustness.", ""] + markdown_table(intervals,list(intervals.columns))
    lines += ["", "## Development-study context", "", "Different cohorts and test sets: these percentages are descriptive, not a controlled causal comparison of training sizes. The within-cohort learning curves are more informative about sample-size trends.", ""] + markdown_table(comparison.reset_index(),["model","20K_development_test_pct","10K_final_test_pct"])
    lines += ["", "## Verification", "", f"All {verification['validation_reference_accepted']}/1500 validation and {verification['test_reference_accepted']}/1500 test numerical-reference controllers passed revalidation. All saved models reproduced their scored gains (max differences: {verification['saved_prediction_max_abs_difference']}). Dataset, prepared split, selected settings and saved-model hashes remained unchanged after locking. Final test scoring occurred once per model.", "",
        "## Limits", "", "One split and training seed; a noise-free stable positive-gain SOPDT family in fixed ranges; ideal differentiation and second-order Pade delay; no actuator saturation, exact-delay robustness or hardware validation. Accepted-label filtering excludes teacher failures and can bias the main population. Results do not establish a universal or safety-certified tuner.", "",
        "Artifacts: saved models, candidate/selected settings, immutable split and selection lock, per-plant test/validation predictions, reference checks, histories, learning curves and plots are stored alongside this report. The website retains the separately labeled 20K development results until explicitly updated."]
    (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(metrics[["passes","success_pct","stable_pct","r2","median_ITAE_ratio","median_J_ratio"]].to_string())


if __name__ == "__main__":
    report()
