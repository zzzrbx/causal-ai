---
name: causal-inference-patterns
description: Common causal inference workflows and decision patterns for choosing the right approach and library
---

# Causal Inference Patterns

Common workflows and decision patterns for causal inference tasks.

## Choosing the Right Approach

### "What causes what?" → Causal Discovery
- **Library**: causal-learn
- **When**: You have observational data and want to learn the causal graph
- **Key algorithms**: PC (no latent confounders), FCI (latent confounders possible), GES (score-based)

### "What is the average effect of X on Y?" → ATE Estimation
- **Library**: DoWhy
- **When**: You have a causal graph (or can specify one) and want to estimate the average treatment effect
- **Workflow**: Define graph → Identify → Estimate → Refute

### "How does the effect vary across individuals?" → CATE / HTE Estimation
- **Library**: EconML
- **When**: You want heterogeneous treatment effects — how the effect of X on Y varies with individual characteristics
- **Key methods**: DML (Double ML), DR Learner, Meta-learners, Causal Forests

### "Is my causal estimate robust?" → Sensitivity / Refutation
- **Library**: DoWhy refuters
- **Methods**: Placebo treatment, random common cause, data subset, bootstrap

## Common Pipelines

### Pipeline 1: Full Causal Analysis
1. **Discover** causal graph with causal-learn (if unknown)
2. **Identify** causal effect with DoWhy
3. **Estimate** ATE with DoWhy or CATE with EconML
4. **Refute** with DoWhy refutation tests

### Pipeline 2: Quick ATE
1. Specify DAG manually
2. Use DoWhy: `CausalModel` → `identify_effect` → `estimate_effect` → `refute_estimate`

### Pipeline 3: Heterogeneous Effects
1. Use EconML: choose estimator (DML, DR, meta-learner)
2. Fit on data with treatment, outcome, features, confounders
3. Get CATE estimates and confidence intervals

## Reference Documentation

- DoWhy docs: `/project/docs/libraries/dowhy/`
- EconML docs: `/project/docs/libraries/econml/`
- causal-learn docs: `/project/docs/libraries/causal-learn/`
- Causality Handbook: `/project/docs/libraries/causality-handbook/` — tutorials and worked examples
