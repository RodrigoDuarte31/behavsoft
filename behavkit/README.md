# behavkit

A reusable pipeline for **DeepLabCut + Behavsoft** data, scoped to:
preprocessing, kinematic feature extraction, binary threshold detection
(e.g. immobility, with a minimum-duration filter and bout metrics), and
multiclass classification (Random Forest) with video-grouped cross-validation.

This package does **no plotting or statistics** itself — every result is
written as a plain CSV (or JSON for hyperparameters), so you can analyze and
plot it with whatever tool you already use (R, Prism, Python notebooks,
etc.). The one exception is annotated example videos, which the package
still generates directly.

Extracted and generalized from an original forced-swim-test (FST) pipeline —
bodypart/behavior names are never hardcoded, so it works for other paradigms
too (e.g. open field).

## Installation

```bash
cd behavkit
pip install -e .

# If you plan to use BalancedRandomForestClassifier (imbalanced classes):
pip install -e ".[balanced]"
```

## Quick start: config-driven (recommended)

1. Copy `config.example.yaml` to `config.yaml` and fill in your paths,
   bodyparts, and behavior labels.
2. Run:

```python
import behavkit as bk

results = bk.run_pipeline("config.yaml")
```

`run_pipeline` reads the config, discovers your video pairs, preprocesses
them, and runs whichever stages you enabled — saving CSVs to the paths
defined in the config, and returning a dict with every intermediate
artifact (`df_train`, `df_full`, the trained model, metrics DataFrames...)
for further interactive use.

See `examples/run_open_field.py` for a minimal usage example.

## What each stage writes

| Stage (config section) | CSV/JSON output |
|---|---|
| `modeling` | Per-class classification report, confusion matrix, best hyperparameters (JSON) |
| `sequences` | One row per behavior event (video_id, order, behavior, start/end/duration) — for Markov-chain or other sequence analysis in an external tool |
| `threshold_detection` | Per-fold CV metrics, raw per-frame activity+label data, per-video bout metrics (frequency/duration/latency), and bout-level detail (annotation vs. prediction, for reconstructing an alignment plot elsewhere) |

## Quick start: manual (function by function)

If you want more control than `run_pipeline()` gives you — e.g. inspecting
intermediate results before deciding the next step — every function is
importable individually. See `examples/run_manual_steps.py` for a complete
walkthrough.

## Configuration file reference

`config.example.yaml` documents every available parameter, and each stage's
block ends with a comment listing exactly which files it writes. You only
need to specify what you want to change from the defaults
(`behavkit/config.py`); anything omitted falls back automatically.

| Section | Controls |
|---|---|
| `project` | fps, random seed, project name |
| `paths` | input folders and every CSV/JSON output path |
| `preprocessing` | likelihood-based interpolation, label renaming |
| `features` | which bodyparts/pairs/triplets feed the feature engine, rolling windows |
| `modeling` | whether to run the RF classifier, hyperparameter grid, label fusing/exclusion |
| `sequences` | whether to export the ordered behavior-event CSV |
| `threshold_detection` | whether to run binary threshold detection + bout metrics |
| `video_annotation` | whether/how to generate an annotated example video |

## Package structure

| Module | Contents |
|---|---|
| `behavkit.config` | YAML config loading, merged with sensible defaults |
| `behavkit.run` | `run_pipeline()`, the main config-driven entry point |
| `behavkit.io` | Reading `.h5` (DLC) and `.xls` (manual scoring), likelihood-based interpolation |
| `behavkit.features` | Generic feature engine (velocity, distance, angle, oscillation, area) |
| `behavkit.pipeline` | Orchestrates batch preprocessing (multiple videos → one dataset) |
| `behavkit.modeling` | Multiclass Random Forest, video-grouped split/CV, optional `BalancedRandomForestClassifier` |
| `behavkit.sequences` | Ordered behavior-event export, for sequence/Markov analysis elsewhere |
| `behavkit.threshold` | CV-based threshold selection, minimum-duration filter, bout metrics and bout-level detail |
| `behavkit.video` | Annotated video with multiclass classification or binary detection overlay |

## Why bodypart parameters are explicit everywhere

The package never assumes fixed tracked-point names (`nose`, `head`, etc.) —
every function in `features` takes lists of bodyparts/pairs/triplets. This is
what lets the same package be reused across datasets with different tracking
schemes (FST vs. open field) without touching the source code, only the
arguments (or config values) passed in.

## Design decisions inherited from the original pipeline (and why)

- **CV is always video-grouped** (`StratifiedGroupKFold`), never frame-level —
  frames from the same video are correlated; a frame-level split leaks
  information between train and test.
- **Split frozen to disk** (`get_or_create_group_holdout`) — prevents
  changing the target (e.g. merging labels) from also silently changing
  which videos land in the holdout, which would invalidate comparisons
  between configurations.
- **Minimum-duration filter, not aggressive smoothing** — removes short
  behavioral noise without blurring the real onset/offset boundaries of a
  state, which would distort latency/duration metrics.
- **Sequences exported from `df_full`, including 'other'** — preserves true
  temporal contiguity between annotated events; filtering 'other' out (if
  desired) is left to the external sequence-analysis tool, since that's an
  analysis choice, not a preprocessing one.
