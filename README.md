# Causal AI framework: core training and ITE estimation

This repository contains the core DeepSurv T-learner implementation for estimating
potential progression-free survival under surgery and neoadjuvant treatment.
Two treatment-specific models produce survival probabilities at 12, 24, 36,
and 60 months:

```text
ITE(t) = S_neoadjuvant(t | X) - S_surgery(t | X)
```

A positive ITE indicates higher predicted PFS under neoadjuvant treatment.
Probabilities and ITE are expressed on the 0–1 scale; an ITE of 0.05 means
a predicted difference of 5 percentage points. No recommendation threshold is
applied by this core implementation.

## Scope

Only model training and direct ITE prediction are included. The code starts from
prepared imaging features, numerically encoded clinical covariates, and supplied
sample weights. Image extraction, source-record processing, propensity-score
estimation, benchmarking, sensitivity analyses, SHAP, and figure generation are
outside this repository. This is a core implementation, not a complete pipeline
for reproducing every manuscript figure or the historical fitted models.

No patient data, patient-level sample weights, or trained model files are hosted
in this GitHub repository. Trained model bundles will be distributed separately.

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
   to select the imaging-screening threshold and training epoch.
3. Fit imputation, feature selection, and scaling only on the corresponding
   training split. Refit using the selected settings on the outer training arm,
   then predict both potential survival curves for every held-out patient.
4. Concatenate paired out-of-fold predictions and compute OOF ITE.
5. Repeat inner selection on each full development treatment arm, fit the two
   final models, and predict the external cohort.

The imaging screen follows the local core training configuration: variance
filtering, weighted univariate Cox **score** tests, then correlation pruning in
ascending p-value order. The default p-value candidates are 0.10, 0.05, and 0.20;
if fewer than eight features pass, the top available features are retained before
correlation pruning. These are training settings, not alternative analysis
branches; edit them in `config.py` for the intended training specification.

The inner selection criterion is unweighted concordance for strictly ordered
follow-up times, with half credit for equal risk scores. The selected epoch is
the median inner-fold best epoch. Inner training uses early stopping and learning
rate reduction; refitting uses a fixed epoch count and initial learning rate.

The Cox objective weights both event terms and risk-set denominators. Training
uses mini-batch risk sets (default batch size 128); the baseline hazard is
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

Each CSV contains patient identifiers and paired `S_surgery_*m`,
`S_neoadjuvant_*m`, and `ITE_*m` columns; the OOF file also contains `fold`.
Model bundles contain network parameters, selected feature names and order,
fitted imputation/scaling parameters, prediction horizons, and the fitted
baseline cumulative hazard. These locally generated files are ignored by Git.

## Models hosted separately

**External model download URL: pending publication by the authors.**

After obtaining compatible bundles from the external hosting site, place them in
a local directory:

```text
models/
├── surgery.pt
└── neoadjuvant.pt
```

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
