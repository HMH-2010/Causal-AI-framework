# Causal AI framework: core training and ITE estimation

This repository contains the core DeepSurv T-learner implementation for estimating
potential progression-free survival under surgery and neoadjuvant treatment.
Two treatment-specific models produce survival probabilities at 12, 24, 36 months:

```text
ITE(t) = S_neoadjuvant(t | X) - S_surgery(t | X)
```

A positive ITE indicates higher predicted PFS under neoadjuvant treatment.
Probabilities and ITE are expressed on the 0–1 scale; an ITE of 0.05 means
a predicted difference of 5 percentage points. No recommendation threshold is
applied by this core implementation.


## Files

| File | Purpose |
| --- | --- |
| `config.py` | Network, nested-validation, and training settings |
| `model.py` | DeepSurv, weighted Cox loss, weighted Breslow survival |
| `training.py` | Fold-local preprocessing, nested selection, final fitting |
| `predict.py` | Local model loading, potential survival, and ITE |
| `run.py` | Training and prediction commands |
| `requirements.txt` | Runtime dependencies |

## Installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for your CPU or GPU,
then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Training input

Provide one local CSV with these metadata columns:

| Column | Meaning |
| --- | --- |
| `patient_id` | Unique identifier across development and external datasets |
| `dataset` | `1`: development; `2`: external evaluation |
| `treatment` | `surgery` or `neoadjuvant` |
| `time` | Observed follow-up time in months, using a consistent endpoint origin |
| `event` | `1`: PFS event; `0`: censored |
| `weight` | Positive, precomputed training sample weight |

Use the following prefixes for numeric feature columns:

| Prefix | Input | Processing inside each training split |
| --- | --- | --- |
| `img_` | Imaging features, e.g. `img_0001` | Median imputation, feature selection, z-score scaling |
| `cont_` | Continuous clinical variables, e.g. `cont_age` | Median imputation, z-score scaling |
| `cat_` | Predefined binary or one-hot clinical variables, e.g. `cat_sex_male` | Mode imputation; retain the encoded scale |

Only columns with these prefixes enter the models. At least one usable imaging
feature is required. Specify a fixed categorical coding scheme before running
the code. Represent missing feature values with empty cells. Entirely missing
clinical columns are filled with zero; entirely missing imaging columns are
discarded by the variance filter.

The `weight` column is used as supplied, normalized to mean one within each
training arm/split. The code does not infer whether it represents standard
overlap weights or another weighting definition, and it does not estimate
propensity scores. Select the intended weights and covariates before running;
calendar year is included only if supplied as a feature such as `cont_year`.
External outcomes and weights are not used for training or hyperparameter selection.
Weights and features prepared outside this code must not use external outcomes.

## Training procedure

1. Split the development cohort into five outer folds, stratified by treatment
   and event. Both treatment models use the same held-out outer fold.
2. Within each outer training treatment arm, use three-fold inner validation
   to select a joint configuration of imaging-screening threshold, network
   architecture, dropout, initial learning rate, weight decay, and batch size,
   together with the training duration.
3. Fit imputation, feature selection, and scaling only on the corresponding
   training split. Refit using the selected settings on the outer training arm,
   then predict both potential survival curves for every held-out patient.
4. Concatenate paired out-of-fold predictions and compute OOF ITE.
5. Repeat inner selection on each full development treatment arm, fit the two
   final models, and predict the external cohort.

The imaging screen follows the local core training configuration: variance
filtering, weighted univariate Cox **score** tests, then correlation pruning in
ascending p-value order.

The default joint search space in `config.py` is:

| Hyperparameter | Candidates |
| --- | --- |
| Imaging-screening p-value threshold | 0.10, 0.05, 0.01 |
| Hidden-layer widths | (64, 32), (128, 64), (128, 64, 32) |
| Dropout | 0.3, 0.5, 0.7 |
| Initial learning rate | 0.0001, 0.0003, 0.001 |
| Weight decay | 0, 0.00001, 0.0001 |
| Batch size | 32, 64, 128 |

By default, 24 distinct joint configurations are sampled without replacement
from the 729 combinations using `SEARCH_SEED=42`. The same candidate list is
used for every outer fold and both treatment arms, while each training split
selects its own winner. This is a randomized search, not an exhaustive grid
search; set `SEARCH_ITERATIONS=729` to evaluate all default combinations.
Only inner validation scores determine the winner. Candidates use the same
inner folds and per-fold training seeds. Preprocessing is fitted separately
on each inner training split. Batch normalization remains enabled.

The inner selection criterion is the mean of the best validation C-indices
across inner folds, using unweighted concordance for strictly ordered follow-up
times, with half credit for equal risk scores. Ties retain the first sampled
configuration. The selected epoch is the median inner-fold best epoch for the
winning configuration. Inner training uses early stopping and learning-rate
reduction; refitting uses the selected architecture and optimizer settings for
the selected epoch count, starting at the selected initial learning rate without
validation-driven scheduling. The final full-development fits repeat this search
independently in each treatment arm. Search scores are tuning scores, not reported
validation performance; performance evaluation uses held-out outer-fold or
external predictions.

The defaults require 24 x 3 = 72 inner fits per training arm/split (864 inner
fits plus 12 refits across five outer folds and final full-development fitting).
Edit `SEARCH_SPACE` and `SEARCH_ITERATIONS` to change the computational budget.
This expanded search changes the training procedure; reproducing prior numerical
results requires their original code, configuration, and fitted models.

The Cox objective weights both event terms and risk-set denominators. Training
uses mini-batch risk sets with the selected batch size; the baseline hazard is
estimated using the full fitted training arm. Tied follow-up times share a common
risk set using the Breslow convention. Log-space calculations stabilize the
loss and survival predictions. These implementation details can affect numerical
results relative to earlier code versions; no equivalence to historical
manuscript estimates is asserted.

```bash
python run.py train --data data.csv --output outputs --device cpu
```

For a CUDA-enabled installation, use `--device cuda`. Adequate event counts and
comparable survival pairs are required in each inner validation fold.

Local outputs:

```text
outputs/
├── oof_predictions.csv
├── external_predictions.csv
└── models/
    ├── surgery.pt
    └── neoadjuvant.pt
```

Each model bundle records its selected network architecture, training
configuration, training duration, search seed, and all candidate inner scores
and best epochs, alongside the fitted preprocessing and baseline hazard.
Prediction reconstructs each arm from its saved architecture, so the two arms
can have different networks. Existing bundles remain loadable.
Outer-fold candidate scores are printed during training but only the final
full-development model bundles are saved by this core workflow.

Use both bundles from the same release. They must follow the bundle format
written by `training.py`; a bare neural-network state dictionary is insufficient
for absolute survival prediction. Earlier checkpoints may require packaging
with their original preprocessing and baseline hazard before use.

For new patients, provide `patient_id` and the feature columns named in the
bundles. Outcome, treatment, dataset, and weight columns are not needed:

```bash
python run.py predict --data new_patients.csv --models models --output predictions --device cpu
```

The resulting `predictions/predictions.csv` contains both potential survival
estimates and ITE. Keep input feature definitions and coding consistent with the
training release.
