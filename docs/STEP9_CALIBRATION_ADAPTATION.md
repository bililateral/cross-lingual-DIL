# Step 9: Frozen-Score Calibration Adaptation

Status: synchronized on the Step 5 v2 cleaned boundary on `2026-04-15`; run this branch on the Linux runtime server only

This branch is separate from the pure few-shot retraining runner.

## Role

The calibration branch keeps a Step 7 scorer frozen and fits a calibration map on reviewed Chinese supervision.

It changes the question from:

- "can a new target-aware LightGBM help?"

to:

- "can a frozen zero-shot scorer be made more usable on the Chinese target domain without retraining tree structure?"

## Active Data Boundary

The active synchronized boundary is:

- `zh_target_strict train = 140`
- `zh_target_strict valid = 41`
- `zh_target_strict test = 38`

This branch does not reopen Step 5.

## Protocol

The active calibration branch uses:

- frozen Step 7 model weights
- Chinese strict `train` only for calibration fitting
- Chinese strict `valid` only for a diagnostic selected threshold
- Chinese strict `test` as the untouched evaluation split
- fixed calibrated threshold `0.5` as the branch primary threshold

## Synchronized Experiments

Current experiments:

- `core_calibrated_default`
- `core_calibrated_bge_m3`
- `identifier_augmented_calibrated_default`

## Remote Linux Commands

```bash
python3 scripts/step9_run_calibration_adaptation.py \
  --experiment core_calibrated_default \
  --experiment core_calibrated_bge_m3 \
  --experiment identifier_augmented_calibrated_default
```

## Current Reading

Current clean calibration results:

- `core_calibrated_default`
  - primary threshold `0.5`
  - `balanced_accuracy = 0.886905`
  - `roc_auc = 0.940476`
  - diagnostic selected-threshold `balanced_accuracy = 0.750000`
  - `parameter_scale = 1.594385`
- `core_calibrated_bge_m3`
  - primary threshold `0.5`
  - `balanced_accuracy = 0.860119`
  - `roc_auc = 0.952381`
  - diagnostic selected-threshold `balanced_accuracy = 0.833333`
  - `parameter_scale = 1.644724`
- `identifier_augmented_calibrated_default`
  - primary threshold `0.5`
  - `balanced_accuracy = 0.901786`

## Interpretation

The fresh calibration reading is:

- calibration is healthy again
- the default line remains the strongest clean calibration control
- the BGE calibration line is now close enough that calibration should be discussed as a two-family comparison
- calibration is no longer suffering from the earlier probability-saturation failure

## Step 11 Integration

The current fresh dynamic Step 11 calibration control is:

- `core_calibrated_default`

Its current graph reading is:

- graph threshold `0.8`
- threshold-pass edges `464`
- retained edges `119`
- clusters `28`
- largest cluster `10`
- `tree_cluster_share = 0.0`

So the current calibrated graph is now topologically healthy and should be kept as a conservative control against the larger few-shot BGE discovery graph.
