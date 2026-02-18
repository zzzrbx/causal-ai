---
name: causal-learn-reference
description: Guide to causal-learn library documentation for causal discovery algorithms (PC, FCI, GES)
---

# causal-learn Reference

causal-learn is a library for causal discovery — learning causal structure from observational data.

## Documentation Location

All causal-learn docs are in `/project/docs/libraries/causal-learn/`.

### Key Locations

- **Source code**: `/project/docs/libraries/causal-learn/causallearn/` — Python source
- **Tests**: `/project/docs/libraries/causal-learn/tests/` — usage examples in test form

### Key Algorithms

| Algorithm | Module | Type |
|-----------|--------|------|
| PC | `causallearn.search.ConstraintBased.PC` | Constraint-based |
| FCI | `causallearn.search.ConstraintBased.FCI` | Constraint-based (latent confounders) |
| GES | `causallearn.search.ScoreBased.GES` | Score-based |
| LiNGAM | `causallearn.search.FCMBased.lingam` | Functional causal model |
| NOTEARS | `causallearn.search.ScoreBased.NOTEARS` | Continuous optimization |

### Common Workflow

1. Load data as numpy array
2. Choose algorithm based on assumptions:
   - No latent confounders → PC or GES
   - Possible latent confounders → FCI
   - Linear non-Gaussian data → LiNGAM
3. Run: `cg = pc(data, alpha=0.05, indep_test='fisherz')`
4. Visualize: `cg.draw_pydot_graph()`
5. Extract adjacency matrix: `cg.G.graph`

### Key Subpackages

- `causallearn.search.ConstraintBased` — PC, FCI
- `causallearn.search.ScoreBased` — GES, exact search
- `causallearn.search.FCMBased` — LiNGAM variants
- `causallearn.utils.cit` — conditional independence tests

### Search Tips

- **Always use `glob: '*.py'` when grepping** — notebooks (.ipynb) are JSON and won't match
- Grep for `def pc(` or `def fci(` to find algorithm entry points
- Grep for `indep_test` to find available independence tests
- Check `tests/` directory for concise usage examples
