---
name: econml-reference
description: Guide to EconML library documentation for heterogeneous treatment effect estimation (CATE)
---

# EconML Reference

EconML is a library for estimating heterogeneous treatment effects (CATE) using machine learning.

## Documentation Location

EconML source is in `/project/.venv/lib/python3.12/site-packages/econml/`.

### Key Locations

- **Source code**: `/project/.venv/lib/python3.12/site-packages/econml/` — Python source

### Key Estimators

| Estimator | Module | Use Case |
|-----------|--------|----------|
| `LinearDML` | `econml.dml` | Double ML with linear final stage |
| `CausalForestDML` | `econml.dml` | Double ML with causal forest |
| `DRLearner` | `econml.dr` | Doubly robust learner |
| `ForestDRLearner` | `econml.dr` | DR with forest |
| `SLearner`, `TLearner`, `XLearner` | `econml.metalearners` | Meta-learners |
| `OrthoIV`, `DRIV` | `econml.iv` | Instrumental variable methods |
| `DynamicDML` | `econml.dynamic` | Dynamic/panel treatment effects |

### Common Workflow

1. Prepare data: Y (outcome), T (treatment), X (features for heterogeneity), W (confounders)
2. Choose estimator: `est = LinearDML(model_y=..., model_t=...)`
3. Fit: `est.fit(Y, T, X=X, W=W)`
4. Predict CATE: `est.effect(X)` or `est.const_marginal_effect(X)`
5. Confidence intervals: `est.effect_inference(X)`

### Search Tips

- **Always use `glob: '*.py'` when grepping** — notebooks (.ipynb) are JSON and won't match
- Grep in `/project/.venv/lib/python3.12/site-packages/econml/` for `LinearDML` or `CausalForestDML` for DML examples
- Grep for `effect_inference` for confidence interval examples
