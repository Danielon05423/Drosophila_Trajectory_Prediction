# Project_Phase_B
Clustering and predicting fly movement trajectories using initial heading conditions and DTW-based centroid modeling.



# Fly Trajectory Prediction — Pipeline

Predicts where a fly will be a fraction of a second from now.

## Running in Google Colab (recommended)

Open `Fly_Project_Colab.ipynb` in Colab. It mounts Drive, finds this folder,
installs what is missing, and runs every stage in order. Set the runtime to GPU
first (`Runtime → Change runtime type → GPU`).

The notebook is configured for the **full run**: all 326 flies, predicting
50 ms ahead, writing to `outputs_full/`. The earlier quick-test results in
`outputs/` are left untouched, so the two can be compared.

## Running locally

```
pip install numpy pandas matplotlib seaborn tslearn scikit-learn torch

python 06_architecture_diagram.py  # the two diagrams (needs no data)
python 01_cluster_ksweep.py        # clustering + optimal k
python 02_prepare_trajlearn.py     # prepare TrajLearn input
python 03_continuous_model.py      # the predictor + baselines + snapshots
python 04_report_figures.py        # one figure per fly + cluster breakdown
python 05_horizon_figure.py        # error vs. horizon (needs the sweep first)
```

Output goes to `$FLY_OUTPUT_DIR`, or `outputs/` if that is unset. The scripts
find the CSV files themselves (recursive search from this folder) and skip the
metadata block and Arena table at the top of each raw file.

Order matters in two places: `04` needs the weights `03` saves, and `05` needs
`horizon_sweep.csv`, which the notebook's step 5 produces by running `03` at
several horizons.

### Configuration via environment variables

Every tunable setting can be overridden without editing code, which is how
the Colab notebook controls the scripts:

| Variable | Meaning | Default |
|---|---|---|
| `FLY_DATA_DIR` | where to search for raw CSVs | script folder |
| `FLY_OUTPUT_DIR` | where results are written | `./outputs` |
| `FLY_NROWS` | rows read per file (`none` = all) | all |
| `FLY_K_MIN` / `FLY_K_MAX` | k sweep range | 2 / 15 |
| `FLY_SWEEP_MAX` | segments used in the sweep | 1200 |
| `FLY_HORIZON` | frames ahead to predict (3 = 50 ms) | 1 |
| `FLY_WINDOW` | frames of history fed to the model | 10 |
| `FLY_HEX_SIZE_MM` | hexagon size for TrajLearn | 1.5 |
| `FLY_MAX_EPOCHS` | training epoch limit | 30 |
| `FLY_PATIENCE` | epochs without improvement before stopping | 5 |
| `FLY_CHECKPOINT_EVERY` | snapshot the model every N epochs | 5 |
| `FLY_ROLLOUT_FRAMES` | frames drawn per fly in step 4 | 120 |
| `FLY_MAX_FIGURES` | stop after N per-fly figures (for testing) | all |
| `FLY_FIGURE_DPI` | resolution of the per-fly figures | 110 |
| `FLY_CLUSTER_SAMPLE` | moments sampled for the cluster breakdown | 1500 |

---

## `fly_common.py` — shared module

Data loading, header detection, global fly IDs, and motion features.

**Key concept — movement bouts:** in this dataset the flies are stationary
most of the time. The median frame-to-frame displacement is about 0.02 mm,
and 68% of frames move less than 0.1 mm. Stages 2 and 3 therefore only use
stretches where the fly is genuinely moving (above 2 mm/s for at least a
second), which keeps roughly 27% of frames.

## `01_cluster_ksweep.py` — how many movement types are there?

Sweeps k = 2..15, running soft-DTW K-Means on 40-frame segments and scoring
each k with the silhouette coefficient, then keeps the best k.

**Reading the score:** silhouette runs from -1 to 1; higher means better
separated clusters. **Set expectations realistically:** fly movement is a
continuum (turns of every sharpness, speeds of every magnitude), not sharp
categories. A score of 0.2-0.4 already indicates useful structure, and a
value near 1 is not achievable here. A low-but-positive peak is a legitimate
finding to report — it says the clusters are convenient labels carved out of
a continuum, not naturally separate groups.

**Interruptions are safe.** Results are written after every k, and re-running
skips whatever is already in `outputs/ksweep_silhouette.csv`. Delete that
file to start over — and do delete it if you change the sample size, so
results from different settings don't get mixed.

Outputs: `ksweep_silhouette.csv/.png`, `fly_segments_clustered.csv`,
`clusters_plot.png`.

## `02_prepare_trajlearn.py` — adapting TrajLearn

TrajLearn is a transformer over sequences of discrete spatial cells. Two
adaptations were required:

1. **Scale.** TrajLearn uses the H3 geographic hex grid, whose finest
   resolutions are still metres across; our arena is 65-170 mm. This script
   replaces H3 with a millimetre-scale hex grid using axial coordinates.
   Everything else (vocabulary, neighbour lists, embeddings) matches what
   TrajLearn's own `preprocess.py` emits, so its training code runs
   unmodified.

2. **Sampling rate.** At 60 Hz the fly stays in the same cell for most
   frames, so a token model would learn to predict "same cell again" and
   score above 95% while being useless. Two filters fix this: keep only
   movement bouts, and collapse runs of the same cell. Every token is then a
   **new** cell — the assumption TrajLearn was built on.

The resulting model answers "which cell does the fly move to next" — the
direction it commits to, at cell resolution. The precise 17 ms prediction is
handled by stage 3.

**Train/test split:** TrajLearn slices `data.txt` purely by line order, so
this script writes lines grouped by fly (train first) and computes ratios
that reproduce that split exactly. `verify_split` replays TrajLearn's own
slicing and aborts on any mismatch, because even a one-line drift leaks a
fly between train and test.

**Then run TrajLearn itself:**

```
git clone https://github.com/amir-ni/Trajectory-prediction
cp -r outputs/trajlearn/flies  Trajectory-prediction/data/flies
cp outputs/trajlearn/configs.yaml  Trajectory-prediction/
cd Trajectory-prediction
python main.py configs.yaml           # train
python main.py configs.yaml --test    # evaluate
```

Training runs up to 50 epochs with patience 5. If training struggles,
increase `FLY_HEX_SIZE_MM` for a smaller vocabulary.

## `03_continuous_model.py` — the actual prediction

An LSTM that takes the last 10 frames and predicts the displacement over the
next frame (~17 ms), in millimetres. The displacement is expressed in the
fly's own reference frame (forward / sideways), so the model learns how the
fly turns and accelerates rather than memorising where in the arena things
happen.

Trains for up to 30 epochs with early stopping (patience 5).

**Comparisons:**

| Model | What it is |
|---|---|
| LSTM | the learned model |
| Constant velocity | fly keeps its current speed and heading — physical baseline |
| Stopped | fly does not move — shows how much accuracy is just the fly being slow |

The baseline comparison is the point. Over 17 ms the fly barely has time to
deviate from its current velocity, so the physical baseline may well win.
**That is a legitimate result to report**, not a failure — it says physics
is sufficient at that horizon. To show where the learned model starts to pay
off, raise `FLY_HORIZON` (6 = 0.1 s, 30 = 0.5 s) and compare again. A plot of
error against prediction horizon is a stronger result than any single number.

**Snapshots.** The model is saved at epoch 0 (untrained), 5, 10, 15… and every
snapshot is then scored on the *same* held-out test flies. That turns "the
model improves as it trains" into a measurement: the only thing changing
between points is how long it trained. `epoch_progression.png` plots it, and
`epoch_checkpoints.csv` has the numbers.

Outputs: `epoch_progression.png`, `learning_curve.png`,
`error_by_condition.png`, `prediction_examples.png`,
`prediction_error_plot.png`, `prediction_results.csv`,
`epoch_checkpoints.csv`, `training_history.csv`, `fly_split.json`,
`next_step_lstm.pt`.

## `04_report_figures.py` — per-fly report figures, all splits

For every fly with a long enough movement bout — train, validation, and test
alike — produces a two-panel figure:

* **Top: spatial comparison.** The true path over the bout's most active
  stretch (found the same way as elsewhere: highest rolling speed), plotted
  against the constant-velocity baseline and the model's own path at up to
  four training epochs (untrained through final). Each path's endpoint drift,
  in millimetres, is quoted directly in the legend.
* **Bottom: cumulative error distribution.** A CDF of one-step prediction
  error (mm) over the whole bout, model vs. physics baseline, with the
  model's win rate against physics in the title.

Files are named `fly_0042_test.png`, `fly_0015_train.png`, etc. — the split
label comes from `fly_split.json` (written by `03_continuous_model.py`), so
train/val/test flies are all drawn and distinguishable by filename. Only the
console summary restricts itself to test flies ("the ones that count").

Also produces `error_by_cluster.png`, which joins this stage to stage 1: it
assigns test moments to the movement clusters found there and reports the
error per cluster. It needs `cluster_centroids.npy`, which `01` writes into
the same output folder; without it that one figure is skipped and everything
else still runs.

Outputs: `per_fly_trajectories/*.png`, `per_fly_summary.csv`,
`error_by_cluster.png`, `error_by_cluster.csv`.

## `05_horizon_figure.py` — how far ahead is it worth predicting?

Reads `horizon_sweep.csv` and turns it into one slide. Predicting further
ahead is harder for everyone, but a straight-line assumption decays much
faster than a learned one, so the model's *advantage* rises, peaks, and falls.
The peak is a genuine design result.

Outputs: `horizon_efficiency.png`, `horizon_efficiency.csv`.

## `06_architecture_diagram.py` — the diagrams

`pipeline_diagram.png` (the whole project on one page) and
`model_diagram.png` (inside the predictor). Drawn from scratch, so they need
no data and can be regenerated any time.

**`prediction_examples.png` is the figure to put in the report.** Each panel is
one real moment from the test set, drawn in the fly's own frame: the fly sits
at the origin facing right, the grey line is where it came from, and the
markers are the true next position and the two predictions of it. Panels run
from the easiest case to the hardest. Each has a zoom inset, because at full
scale the history spans a couple of millimetres while the methods differ by
hundredths of one — without the zoom all three markers overlap.

---

## What was tested, and what was not

**Verified against the real data:** file loading and header detection;
movement-bout detection; the hex grid (confirmed every point maps to its
genuinely nearest hexagon, after fixing a cube-rounding bug); the TrajLearn
output format and the split verification (after fixing an off-by-one caused
by float truncation); window and baseline construction (confirmed no
information leaks from the future and that the rotations are invertible);
the k sweep (confirmed it recovers a known number of clusters on planted
test data); and the resume logic (confirmed an interrupted sweep continues
correctly).

Also verified against the real data, in the later round of work: the
free-rollout maths (a straight-ahead predictor produces a straight line, a
constant-turn predictor produces a closed circle of constant step length, a
zero predictor stays put, and no path contains NaN); that a predictor which
*is* the constant-velocity baseline reproduces the baseline column to machine
precision, and a random one loses on every fly; the per-fly figure end to end
on real flies; the cluster breakdown end to end against real soft-DTW
centroids, including its behaviour when the centroids are missing; and the
error-by-condition plot, including the degenerate case of too few moments,
which now prints an explanation instead of an empty pair of axes.

**Not executed:** the PyTorch training loop itself — torch could not be
installed in the test environment (disk limits), so everything torch-specific
was checked by review and everything around it was tested with a NumPy
stand-in predictor. The structure is standard, and two bugs were found and
fixed by review this way (inference over a million windows in one call, and a
loss `beta` mismatched to the millimetre scale of the targets).

**Run `FLY_NROWS=200000` first** to confirm the whole chain runs end to end
before committing hours to the full dataset.

## Runtimes

Measured on 2 CPU cores, which is what free Colab provides.

| Mode | Rows/file | k range | Sweep size | Stage 1 |
|---|---|---|---|---|
| quick | 200,000 | 2-6 | 400 | ~10 min |
| medium | all | 2-15 | 400 | ~50 min |
| full | all | 2-15 | 1200 | 2-2.5 h |

Stage 2 takes 10-25 minutes, dominated by reading the files. Stage 3 takes
10-20 minutes on a GPU, one to three hours without.

`medium` is the sweet spot: it uses the entire dataset, and only the sweep
resolution separates it from `full`.
****
