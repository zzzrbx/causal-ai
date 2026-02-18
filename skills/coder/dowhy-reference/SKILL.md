---
name: dowhy-reference
description: Guide to DoWhy library documentation for causal inference identification, estimation, and refutation
---

# DoWhy Reference

DoWhy is a causal inference library for identification, estimation, and refutation.

## Documentation Location

DoWhy source is in `/project/.venv/lib/python3.12/site-packages/dowhy/`.

### Key Locations

- **Source code**: `/project/.venv/lib/python3.12/site-packages/dowhy/` — Python source for understanding internals

### Key Modules

| Module | Purpose |
|--------|---------|
| `dowhy.CausalModel` | Main entry point — specify graph, identify, estimate, refute |
| `dowhy.causal_estimators` | Estimation methods (propensity score, IV, regression, etc.) |
| `dowhy.causal_identifier` | Identification strategies (backdoor, frontdoor, IV) |
| `dowhy.causal_refuters` | Refutation tests (placebo, random common cause, subset) |
| `dowhy.gcm` | Graphical causal models (structural causal models, root cause analysis) |
| `dowhy.graph_learners` | Causal discovery integration |

### Common Workflow

1. Define causal graph (DAG) as a string or networkx graph
2. Create `CausalModel(data, treatment, outcome, graph)`
3. Identify causal effect: `model.identify_effect()`
4. Estimate: `model.estimate_effect(identified_estimand, method_name=...)`
5. Refute: `model.refute_estimate(identified_estimand, estimate, method_name=...)`

### Search Tips

- **Always use `glob: '*.py'` when grepping** — notebooks (.ipynb) are JSON and won't match
- Grep in `/project/.venv/lib/python3.12/site-packages/dowhy/` for `CausalModel` to find usage examples
- Grep for `method_name` to find available estimation methods
- Grep for `refute_estimate` to find refutation patterns
