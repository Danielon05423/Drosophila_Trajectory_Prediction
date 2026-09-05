# -*- coding: utf-8 -*-
"""
Step 4 - the figures for the project book and the presentation.

Processes all flies across train, val, and test splits to turn raw model
outputs into figures that answer how well the learned model generalizes.

A good presentation slide needs to show more than just a line on a graph; it
needs to prove that the model actually outperforms physics where it counts.
To make this clear at a glance, the per-fly figure uses a unified two-graph
layout:
  * Top: A spatial comparison showing the true flight path alongside the
    physics baseline and the step-by-step evolution of training epochs.
    Drift errors in millimeters are integrated directly into the legend so
    viewers can see the improvement instantly without hunting across axes.
  * Bottom: A cumulative distribution function (CDF) of the error magnitude
    across the entire bout, giving an honest look at the model's win rate
    versus the constant-velocity baseline.

Outputs:
  1. Per-fly figures (Unified 2-graph layout) under 'per_fly_trajectories/'
  2. per_fly_summary.csv - summary table of errors across all flies
  3. error_by_cluster.png & .csv - breakdown of model advantage by movement cluster
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fly_common as fc

try:
    import torch
except ImportError:
    sys.exit("PyTorch is required. Install with:  pip install torch")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROLLOUT_FRAMES = fc.env_int("FLY_ROLLOUT_FRAMES", 120)
MAX_EPOCH_PANELS = 4
FIGURE_DPI = fc.env_int("FLY_FIGURE_DPI", 110)
MAX_FLIES = fc.env_int("FLY_MAX_FIGURES", None)

CLUSTER_SAMPLE = fc.env_int("FLY_CLUSTER_SAMPLE", 1500)
NROWS_PER_FILE = fc.env_int("FLY_NROWS", None)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_INDEX_SPEED = 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() \
    else os.getcwd()


# ---------------------------------------------------------------------------
# Loading what step 3 saved
# ---------------------------------------------------------------------------
def load_model_module():
    path = os.path.join(SCRIPT_DIR, "03_continuous_model.py")
    if not os.path.exists(path):
        sys.exit(f"Cannot find '{path}' - it defines the model architecture.")
    spec = importlib.util.spec_from_file_location("continuous_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bundle():
    path = os.path.join(fc.OUTPUT_DIR, "next_step_lstm.pt")
    if not os.path.exists(path):
        sys.exit(f"'{path}' not found. Run 03_continuous_model.py first.")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if "checkpoints" not in bundle:
        sys.exit(f"'{path}' has no saved snapshots. Re-run 03_continuous_model.py.")
    return bundle


def build_predictor(model_module, bundle, state_dict):
    mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
    std = np.asarray(bundle["feature_std"], dtype=np.float64)

    model = model_module.NextStepLSTM(
        len(bundle["feature_names"]),
        hidden=bundle.get("hidden_size", 64),
        layers=bundle.get("num_layers", 2),
        dropout=bundle.get("dropout", 0.1),
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    @torch.no_grad()
    def predict(X):
        Xn = torch.from_numpy((np.asarray(X, np.float64) - mean) / std).float()
        return model(Xn.to(DEVICE)).cpu().numpy()

    return predict


# ---------------------------------------------------------------------------
# Turning one bout into arrays
# ---------------------------------------------------------------------------
def bout_arrays(bout, window):
    x = bout["position_x(mm)"].to_numpy(np.float64)
    y = bout["position_y(mm)"].to_numpy(np.float64)
    heading = bout["heading_unwrapped(rad)"].to_numpy(np.float64)
    speed = bout["speed(mm/s)"].to_numpy(np.float64)
    ang_vel = bout["angular_velocity(rad/s)"].to_numpy(np.float64)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    forward = dx * cos_h - dy * sin_h
    lateral = dx * sin_h + dy * cos_h
    d_heading = np.diff(heading, prepend=heading[0])

    feats = np.column_stack([forward, lateral, speed, ang_vel, d_heading])
    return x, y, heading, feats


def free_rollout(predict, feats_seed, start_xy, start_heading, n_steps, horizon):
    window = feats_seed.shape[0]
    feats = list(feats_seed)
    pos = np.array(start_xy, dtype=np.float64)
    heading = float(start_heading)
    path = [pos.copy()]

    for _ in range(n_steps):
        X = np.asarray(feats[-window:], dtype=np.float64)[None, :, :]
        df, dl = predict(X)[0].astype(np.float64)

        c, s = np.cos(heading), np.sin(heading)
        pos = pos + np.array([df * c - dl * s, df * s + dl * c])
        path.append(pos.copy())

        step_len = float(np.hypot(df, dl))
        d_theta = float(np.arctan2(dl, df)) if step_len > 1e-9 else 0.0
        heading += d_theta

        per_frame = np.array([df / horizon, dl / horizon,
                              step_len / (horizon * fc.DT),
                              d_theta / (horizon * fc.DT),
                              d_theta / horizon])
        feats.extend([per_frame] * horizon)

    return np.asarray(path)


def physics_rollout(x, y, last_idx, n_steps, horizon):
    vx = x[last_idx] - x[last_idx - 1]
    vy = y[last_idx] - y[last_idx - 1]
    steps = np.arange(n_steps + 1)[:, None]
    start = np.array([x[last_idx], y[last_idx]])
    return start + steps * np.array([vx, vy]) * horizon


def one_step_errors(predict, x, y, heading, feats, window, horizon):
    n = len(x)
    starts = np.arange(0, n - window - horizon)
    if len(starts) == 0:
        return None

    X = np.stack([feats[s:s + window] for s in starts])
    last = starts + window - 1
    target = last + horizon

    c, s = np.cos(-heading[last]), np.sin(-heading[last])
    gdx, gdy = x[target] - x[last], y[target] - y[last]
    Y = np.column_stack([gdx * c - gdy * s, gdx * s + gdy * c])

    B = np.column_stack([feats[last, 0] * horizon, feats[last, 1] * horizon])
    P = predict(X)

    return {
        "model": np.linalg.norm(P - Y, axis=1),
        "physics": np.linalg.norm(B - Y, axis=1),
    }


# ---------------------------------------------------------------------------
# The per-fly figure
# ---------------------------------------------------------------------------
def draw_fly_figure(fly_id, split_label, bout, predictors, epoch_list,
                    window, horizon, out_dir):
    x, y, heading, feats = bout_arrays(bout, window)
    if len(x) < window + horizon + 10:
        return None

    offset = 0
    if len(x) > ROLLOUT_FRAMES:
        speed = feats[:, FEATURE_INDEX_SPEED]
        rolling = np.convolve(speed, np.ones(ROLLOUT_FRAMES) / ROLLOUT_FRAMES,
                              mode="valid")
        offset = int(np.argmax(rolling))

    n = min(len(x) - offset, ROLLOUT_FRAMES)
    if n < window + horizon + 10:
        return None

    sl = slice(offset, offset + n)
    x, y, heading, feats = x[sl], y[sl], heading[sl], feats[sl]

    last_idx = window - 1
    n_steps = 1  # 3 frames horizon

    seed = feats[:window]
    start_xy = (x[last_idx], y[last_idx])

    prediction_frames = n_steps * horizon
    end_idx = min(last_idx + prediction_frames, n - 1)

    true_path = np.column_stack([
        x[last_idx:end_idx + 1],
        y[last_idx:end_idx + 1]
    ])
    true_end = true_path[-1]

    final_epoch = epoch_list[-1]
    model_path = free_rollout(predictors[final_epoch], seed, start_xy,
                              heading[last_idx], n_steps, horizon)
    phys_path = physics_rollout(x, y, last_idx, n_steps, horizon)

    errs = one_step_errors(predictors[final_epoch], x, y, heading, feats,
                           window, horizon)

    drift_model = float(np.linalg.norm(model_path[-1] - true_end))
    drift_phys = float(np.linalg.norm(phys_path[-1] - true_end))

    fig, (ax_comb, ax_err) = plt.subplots(2, 1, figsize=(10, 11))

    # --- TOP GRAPH: Combined Spatial Path with Physics and Epoch Errors in Legend ---
    ax_comb.plot(true_path[:, 0], true_path[:, 1], color="black", linewidth=3.0,
                 label="real path (ground truth)", zorder=5)
    ax_comb.scatter(*start_xy, color="black", s=80, marker="o", zorder=6, label="start")
    ax_comb.scatter(true_end[0], true_end[1], color="black", s=120, marker="*", zorder=6, label="true end")

    # Physics Path line
    ax_comb.plot(phys_path[:, 0], phys_path[:, 1], color="steelblue", linewidth=2.2,
                 linestyle="--", label=f"constant velocity — off by {drift_phys:.3f} mm", zorder=4)
    
    # Physics End Marker (Plus sign 'P')
    ax_comb.scatter(phys_path[-1, 0], phys_path[-1, 1], color="steelblue", s=90, marker="P", zorder=7,
                    label=f"Physics end")

    # Training Epochs progression with individual epoch error values included in legend labels
    shades = plt.cm.autumn(np.linspace(0.75, 0.0, len(epoch_list)))
    for colour, epoch in zip(shades, epoch_list):
        path = free_rollout(predictors[epoch], seed, start_xy, heading[last_idx], n_steps, horizon)
        epoch_drift = float(np.linalg.norm(path[-1] - true_end))
        label = ("epoch 0 (untrained)" if epoch == 0 else f"epoch {epoch}")
        ax_comb.plot(path[:, 0], path[:, 1], color=colour, linewidth=1.8, alpha=0.8,
                     label=f"LSTM {label} — off by {epoch_drift:.3f} mm", zorder=3)

    # Final Model Path Marker
    ax_comb.scatter(model_path[-1, 0], model_path[-1, 1], color="crimson", s=100, marker="X", zorder=7, 
                    label=f"Final LSTM end — off by {drift_model:.3f} mm")

    split_desc = {
        "test": "TEST fly (held-out, model never saw this one)",
        "val": "VALIDATION fly (used for tuning/early stopping)",
        "train": "TRAIN fly (model trained on this one)"
    }.get(split_label, f"{split_label.upper()} fly")

    ax_comb.set_title(f"Fly {fly_id} | {split_desc}\nSpatial Comparison with Drift Errors (3 frames / {horizon * fc.DT * 1000:.0f} ms ahead)", fontsize=11, fontweight="bold")
    ax_comb.set_xlabel("x (mm)")
    ax_comb.set_ylabel("y (mm)")
    ax_comb.grid(alpha=0.3)
    ax_comb.set_aspect("equal", adjustable="datalim")
    ax_comb.legend(fontsize=7.5, loc="best", ncol=2)

    # --- BOTTOM GRAPH: Cumulative Error Distribution ---
    if errs is not None:
        m_errs = np.sort(errs["model"])
        p_errs = np.sort(errs["physics"])
        
        cdf_y = np.linspace(0, 100, len(m_errs))

        med_m = np.median(m_errs)
        med_p = np.median(p_errs)

        ax_err.plot(m_errs, cdf_y, color="crimson", linewidth=2.2, label=f"LSTM (median error: {med_m:.3f} mm)")
        ax_err.plot(p_errs, np.linspace(0, 100, len(p_errs)), color="steelblue", linewidth=2.2, linestyle="--", label=f"Physics (median error: {med_p:.3f} mm)")
        
        win = 100 * float((errs["model"] < errs["physics"]).mean())
        ax_err.set_title(
            f"Cumulative Error Distribution (Millimeters)\n"
            f"LSTM model closer on {win:.0f}% of moments  |  median: {med_m:.3f} mm vs {med_p:.3f} mm",
            fontsize=11, fontweight="bold")
        ax_err.set_xlabel("Prediction Error Magnitude (mm)")
        ax_err.set_ylabel("Cumulative Percentage of Moments (%)")
        ax_err.grid(alpha=0.3)
        ax_err.legend(fontsize=9, loc="lower right")

    fig.tight_layout()

    path = os.path.join(out_dir, f"fly_{fly_id:04d}_{split_label}.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)

    row = {"fly": fly_id, "split": split_label,
           "start_time_s": float(bout["time(s)"].to_numpy()[offset]),
           "frames_drawn": int(n_steps * horizon),
           "rollout_end_error_model_mm": drift_model,
           "rollout_end_error_physics_mm": drift_phys}
    if errs is not None:
        row.update({
            "onestep_median_model_mm": float(np.median(errs["model"])),
            "onestep_median_physics_mm": float(np.median(errs["physics"])),
            "onestep_win_rate": float((errs["model"] < errs["physics"]).mean()),
        })
    return row


def longest_bout(fly_df):
    bouts = fc.extract_movement_bouts(fly_df)
    return max(bouts, key=len) if bouts else None


def per_fly_figures(model_module, bundle):
    window = int(bundle["window"])
    horizon = int(bundle["horizon"])
    checkpoints = bundle["checkpoints"]

    available = sorted(checkpoints)
    if len(available) <= MAX_EPOCH_PANELS:
        epoch_list = available
    else:
        picks = np.linspace(0, len(available) - 1, MAX_EPOCH_PANELS)
        epoch_list = sorted({available[int(round(p))] for p in picks})

    print(f"Rebuilding model at epochs {epoch_list}...")
    predictors = {e: build_predictor(model_module, bundle, checkpoints[e])
                  for e in epoch_list}

    split = bundle["fly_split"]
    label_of = {}
    for name, ids in split.items():
        for f in ids:
            label_of[int(f)] = name

    out_dir = os.path.join(fc.OUTPUT_DIR, "per_fly_trajectories")
    os.makedirs(out_dir, exist_ok=True)

    rows, drawn, skipped = [], 0, 0
    for fly_id, _condition, fly_df in fc.iter_flies(nrows_per_file=NROWS_PER_FILE):
        if MAX_FLIES is not None and drawn >= MAX_FLIES:
            break
        bout = longest_bout(fly_df)
        if bout is None:
            skipped += 1
            continue
        
        split_type = label_of.get(fly_id, "unused")
        row = draw_fly_figure(fly_id, split_type, bout,
                              predictors, epoch_list, window, horizon, out_dir)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        drawn += 1
        if drawn % 25 == 0:
            print(f"  {drawn} figures written...", flush=True)

    if not rows:
        sys.exit("No fly produced a long enough movement stretch to draw.")

    summary = pd.DataFrame(rows)
    path = os.path.join(fc.OUTPUT_DIR, "per_fly_summary.csv")
    summary.to_csv(path, index=False)
    print(f"\n{drawn} figures written across train/val/test splits, {skipped} skipped. Saved '{path}'")

    test_rows = summary[summary["split"] == "test"]
    if len(test_rows) and "onestep_win_rate" in test_rows:
        print(f"\nAcross the {len(test_rows)} TEST flies (the ones that count)[cite: 3]:")
        print(f"  model beats physics on "
              f"{100 * (test_rows['onestep_median_model_mm'] < test_rows['onestep_median_physics_mm']).mean():.0f}% "
              f"of flies")
        print(f"  median per-fly error: "
              f"{test_rows['onestep_median_model_mm'].median():.4f} mm (model) vs "
              f"{test_rows['onestep_median_physics_mm'].median():.4f} mm (physics)")

    return summary


# ---------------------------------------------------------------------------
# Error broken down by movement cluster
# ---------------------------------------------------------------------------
def error_by_cluster(model_module, bundle):
    centroid_path = os.path.join(fc.OUTPUT_DIR, "cluster_centroids.npy")
    if not os.path.exists(centroid_path):
        print(f"\nSkipping error_by_cluster.png: '{centroid_path}' not found.\n"
              f"Re-run 01_cluster_ksweep.py to write it, then run this script again.")
        return None

    try:
        from tslearn.metrics import cdist_soft_dtw_normalized
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        from tslearn.utils import to_time_series_dataset
    except ImportError:
        print("\nSkipping error_by_cluster.png: tslearn is not installed.")
        return None

    centroids = np.load(centroid_path)
    meta_path = os.path.join(fc.OUTPUT_DIR, "cluster_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    seg_len = int(meta.get("segment_length", centroids.shape[1]))

    window = int(bundle["window"])
    horizon = int(bundle["horizon"])
    predict = build_predictor(model_module, bundle, bundle["state_dict"])

    test_flies = set(int(f) for f in bundle["fly_split"]["test"])
    rng = np.random.default_rng(0)

    print(f"\nAssigning test moments to the {len(centroids)} movement clusters from step 1...")

    segments, err_model, err_phys = [], [], []
    for fly_id, _condition, fly_df in fc.iter_flies(nrows_per_file=NROWS_PER_FILE):
        if fly_id not in test_flies:
            continue
        for bout in fc.extract_movement_bouts(fly_df):
            x, y, heading, feats = bout_arrays(bout, window)
            n = len(x)
            lo, hi = max(seg_len, window), n - horizon - 1
            if hi <= lo:
                continue
            take = min(6, hi - lo)
            for i in rng.choice(np.arange(lo, hi), size=take, replace=False):
                seg = bout.iloc[i - seg_len:i]
                aligned = fc.align_window(seg)
                segments.append(fc.features_matrix(aligned))

                X = feats[i - window:i][None, :, :]
                c, s = np.cos(-heading[i - 1]), np.sin(-heading[i - 1])
                gdx = x[i - 1 + horizon] - x[i - 1]
                gdy = y[i - 1 + horizon] - y[i - 1]
                true = np.array([gdx * c - gdy * s, gdx * s + gdy * c])
                base = np.array([feats[i - 1, 0] * horizon, feats[i - 1, 1] * horizon])
                err_model.append(float(np.linalg.norm(predict(X)[0] - true)))
                err_phys.append(float(np.linalg.norm(base - true)))

            if len(segments) >= CLUSTER_SAMPLE:
                break
        if len(segments) >= CLUSTER_SAMPLE:
            break

    if len(segments) < 50:
        print("Skipping error_by_cluster.png: too few usable test moments.")
        return None

    scaled = TimeSeriesScalerMeanVariance().fit_transform(to_time_series_dataset(segments))
    labels = cdist_soft_dtw_normalized(scaled, centroids).argmin(axis=1)

    df = pd.DataFrame({"cluster": labels, "model_mm": err_model, "physics_mm": err_phys})
    stats = (df.groupby("cluster")
               .agg(n=("model_mm", "size"),
                    model_mm=("model_mm", "median"),
                    physics_mm=("physics_mm", "median"))
               .reset_index())
    stats["improvement_%"] = 100 * (stats["physics_mm"] - stats["model_mm"]) / stats["physics_mm"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    pos = np.arange(len(stats))
    ax.bar(pos - 0.2, stats["physics_mm"], width=0.4, color="steelblue", label="constant velocity")
    ax.bar(pos + 0.2, stats["model_mm"], width=0.4, color="crimson", label="LSTM")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"cluster {int(c)}\n(n={int(n)})" for c, n in zip(stats["cluster"], stats["n"])], fontsize=8)
    ax.set_ylabel("median error (mm)")
    ax.set_title("Error per movement cluster")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    ax = axes[1]
    colours = ["mediumseagreen" if v > 0 else "indianred" for v in stats["improvement_%"]]
    ax.bar(pos, stats["improvement_%"], color=colours)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"cluster {int(c)}" for c in stats["cluster"]], fontsize=9)
    ax.set_ylabel("how much better the model is (%)")
    ax.set_title("Where the learned model earns its place")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Prediction accuracy by movement type, on test flies only[cite: 3].", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    path = os.path.join(fc.OUTPUT_DIR, "error_by_cluster.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    stats_path = os.path.join(fc.OUTPUT_DIR, "error_by_cluster.csv")
    stats.to_csv(stats_path, index=False)
    print(f"Saved '{path}'")
    print(f"Saved '{stats_path}'")
    return stats


def main():
    os.makedirs(fc.OUTPUT_DIR, exist_ok=True)
    model_module = load_model_module()
    bundle = load_bundle()

    print(f"{'=' * 78}\nREPORT FIGURES (Combined Layout + Physics/Epoch Errors in Legend)\n{'=' * 78}")
    print(f"Model predicts {int(bundle['horizon']) * fc.DT * 1000:.0f} ms ahead "
          f"from {int(bundle['window'])} frames of history.")
    
    per_fly_figures(model_module, bundle)

    try:
        error_by_cluster(model_module, bundle)
    except Exception as exc:
        print(f"\nCould not build error_by_cluster.png: {exc}")

    print(f"\nDone. Everything is in '{fc.OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()