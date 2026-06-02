# Step 8: Chinese Zero-Shot Transfer

Status: synchronized to the Step 5 v2 cleaned boundary on `2026-04-15`

Step 8 is the strict external transfer check of the pipeline. It reads Step 7 scorers trained on English supervision and evaluates them on the fixed Chinese strict `test` split without Chinese training labels.

## Active Evaluation Boundary

The active Chinese strict supervision container is:

- `train = 140`
- `valid = 41`
- `test = 38`

Current Chinese strict test label counts:

- `14 positive`
- `24 negative`

## Inputs

- `reports/step7_training_summary.json`
- `reports/step7_core_zero_shot_default_predictions.zh_target_strict_test.csv`
- `reports/step7_core_zero_shot_bge_m3_predictions.zh_target_strict_test.csv`
- `reports/step7_identifier_augmented_default_predictions.zh_target_strict_test.csv`

## Current Zero-Shot Results

### `core_zero_shot_bge_m3`

- threshold `0.686852`
- `balanced_accuracy = 0.607143`
- `roc_auc = 0.952381`
- `average_precision = 0.919816`
- `precision = 1.000000`
- `recall = 0.214286`
- confusion `tp/tn/fp/fn = 3 / 24 / 0 / 11`

### `core_zero_shot_default`

- threshold `0.875934`
- `balanced_accuracy = 0.571429`
- `roc_auc = 0.940476`
- `average_precision = 0.916282`
- `precision = 1.000000`
- `recall = 0.142857`
- confusion `tp/tn/fp/fn = 2 / 24 / 0 / 12`

### `identifier_augmented_default`

- threshold `0.853834`
- `balanced_accuracy = 0.607143`
- `roc_auc = 0.940476`
- `average_precision = 0.915034`

## Interpretation

The active Step 8 record supports these claims:

1. Cross-lingual zero-shot transfer is still real.
   - both clean zero-shot families rank Chinese positives and negatives well
   - the fresh best clean zero-shot line is now `core_zero_shot_bge_m3`

2. The earlier GTE semantic-feature omission bug is no longer relevant to current interpretation.
   - the default/GTE line is now a real semantic model again
   - it is simply no longer the strongest clean zero-shot line on the cleaned boundary

3. The clean zero-shot reading is still conservative.
   - both active lines are high precision, low recall operating points
   - Step 8 is therefore a feasibility check, not the final endpoint of the project

## Relation To Later Steps

- Step 9 pure few-shot asks whether limited Chinese supervision improves this zero-shot baseline
- Step 9 calibration asks whether frozen zero-shot scores can be made more usable without retraining tree structure
- Step 11 projects selected Step 7 or Step 9 scorers into the Chinese candidate graph
