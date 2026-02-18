---
name: causality-handbook
description: Reference guide to the Causality Handbook — conceptual and methodological chapters on causal inference
---

# Causality Handbook

The Causality Handbook contains conceptual, methodological, and worked-example chapters on causal inference. Use it to answer questions about theory, methods, assumptions, and interpretation.

## Location

All chapters are in `/project/docs/libraries/causality-handbook/`.

## Chapter Index

| File | Topic |
|------|-------|
| `01-Introduction-To-Causality.md` | What causality means; Rubin potential outcomes framework |
| `02-Randomised-Experiments.md` | RCTs, randomisation, ATE estimation |
| `03-Stats-Review-The-Most-Dangerous-Equation.md` | Statistical foundations |
| `04-Graphical-Causal-Models.md` | DAGs, d-separation, backdoor criterion |
| `05-The-Unreasonable-Effectiveness-of-Linear-Regression.md` | OLS for causal inference |
| `06-Grouped-and-Dummy-Regression.md` | Fixed effects, grouped estimators |
| `07-Beyond-Confounders.md` | Mediators, colliders, selection bias |
| `08-Instrumental-Variables.md` | IV estimation, relevance and exclusion |
| `09-Non-Compliance-and-LATE.md` | Intent-to-treat, LATE, complier effects |
| `10-Matching.md` | Exact and approximate matching |
| `11-Propensity-Score.md` | PSM, IPW |
| `12-Doubly-Robust-Estimation.md` | AIPW, double robustness |
| `13-Difference-in-Differences.md` | DiD, parallel trends assumption |
| `14-Panel-Data-and-Fixed-Effects.md` | Two-way FE, within estimator |
| `15-Synthetic-Control.md` | Synthetic control method |
| `16-Regression-Discontinuity-Design.md` | Sharp and fuzzy RDD |
| `17-Predictive-Models-101.md` | ML for causal inference |
| `18-Heterogeneous-Treatment-Effects-and-Personalization.md` | CATE, HTE |
| `19-Evaluating-Causal-Models.md` | Model validation, sensitivity analysis |
| `20-Plug-and-Play-Estimators.md` | Off-the-shelf causal estimators |

## How to Use

1. Identify the relevant chapter from the index above.
2. Use `read_file` with the full path, e.g. `read_file("/project/docs/libraries/causality-handbook/04-Graphical-Causal-Models.md")`.
3. Quote or summarise the relevant section in your answer.
