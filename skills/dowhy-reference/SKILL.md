---
name: dowhy-reference
description: Guide to DoWhy library documentation for causal inference identification, estimation, and refutation
---

# DoWhy Reference

DoWhy is a causal inference library for identification, estimation, and refutation.

## Documentation Location

All DoWhy docs are in `/project/docs/libraries/dowhy/`.

### Key Locations

- **User guide**: `/project/docs/libraries/dowhy/docs/source/docs/` — tutorials and how-to guides
- **Source code**: `/project/docs/libraries/dowhy/dowhy/` — Python source for understanding internals
- **Tests**: `/project/docs/libraries/dowhy/tests/` — useful for understanding API usage patterns

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
- Grep for `CausalModel` to find usage examples
- Grep for `method_name` to find available estimation methods
- Grep for `refute_estimate` to find refutation patterns
