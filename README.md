# Causal Survival Treatment Effect

Core code for treatment-specific survival modeling and individualized treatment effect estimation.

## Overview

The workflow trains separate DeepSurv models for:

- surgery
- neoadjuvant treatment

It then estimates potential survival under both treatment strategies and computes the individual treatment effect:

\[ITE(t) = \hat{S}_{neoadjuvant}(t \mid X) - \hat{S}_{surgery}(t \mid X)\]

## Repository Structure

```text
├── config.py
├── model.py
├── training.py
├── counterfactual.py
├── run.py
├── shap_analysis.py
├── requirements.txt
└── README.md
```

## Main Components

- `config.py`: model and training settings
- `model.py`: DeepSurv, weighted Cox loss, weighted Breslow estimation, survival prediction
- `training.py`: fixed-epoch training and five-fold cross-fitting
- `counterfactual.py`: potential survival prediction and ITE calculation
- `run.py`: main workflow
- `shap_analysis.py`: SHAP analysis for treatment-specific risk models and ITE surrogate models

## Input Data

The main script expects:

```text
data.csv
```

Required columns:

| Column | Description |
|---|---|
| `patient_id` | Unique patient identifier |
| `dataset` | `1` or `2` |
| `treatment` | `surgery` or `neoadjuvant` |
| `time` | Follow-up time |
| `event` | Event indicator |
| `weight` | Positive sample weight |

All other columns are treated as model features.

- Dataset 1 is used for model development and cross-fitting.
- Dataset 2 is used for external evaluation.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main workflow:

```bash
python run.py
```

Run SHAP analysis:

```bash
python shap_analysis.py
```
