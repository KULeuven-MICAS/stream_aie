# Phase 4: Variable SSIS + FIFO Constraints - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-07
**Phase:** 04-variable-ssis-fifo-constraints
**Areas discussed:** SSIS loop size dependency, Linearization pattern, SSIS invariant check

---

## SSIS Loop Size Dependency on Tile

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-compute per candidate | Add utility functions that compute SSIS loop sizes per candidate. _init_transfer_fire_helpers becomes coefficient builder. | |
| Recompute SSIS on the fly | Call generate_steady_state_iteration_spaces per candidate. Heavier but more accurate. | |
| Tile-aware SSIS class | Replace existing SSIS class with one where each dimension has K, S, T loops. K=tile, T=workload/(S*K). Single candidate = fixed scalars. | ✓ |

**User's choice:** Tile-aware SSIS class (user-proposed approach)
**Notes:** User specified that the SSIS should have no pre-defined loop sizes — K (kernel) and T (temporal) are set based on the chosen tile size. S (spatial) is fixed. K × S × T = workload_size. For dimensions with only one candidate, values are already fixed. The existing class is replaced (not duplicated) with backward compatibility for single-candidate case.

---

## Linearization Pattern for reuse_levels/tiles_needed/bds_needed

| Option | Description | Selected |
|--------|-------------|----------|
| Direct sum with pre-computed coefficients | sum_k(coeff[k] * joint_binary_var[k]) — pure linear expression, multiply by z_stop via _add_binary_product | |
| Continuous auxiliary with big-M | Same pattern as Phase 3 memory constraints. Continuous variable gated by z_stop. Consistent code shape. | ✓ |

**User's choice:** Continuous auxiliary with big-M (Option B)
**Notes:** Chosen for consistency with Phase 3's memory constraint pattern. Same code shape, same tight bounds strategy.

---

## _ensure_same_ssis_for_all_transfers Invariant

| Option | Description | Selected |
|--------|-------------|----------|
| Keep as model construction check | Verify per candidate that SSIS totals match. | |
| Move to post-solve verification | Check after solver resolves w[dim,k]. Invariant holds by construction since K × S × T = workload_size. | ✓ |

**User's choice:** Move to post-solve verification
**Notes:** User confirmed the invariant holds by construction. Also corrected the decomposition: K × S × T = workload_size (not K × T), where S is the fixed spatial unrolling across multiple cores.

---

## Claude's Discretion

- Internal structure of refactored SSIS class
- How _init_transfer_fire_helpers is restructured
- Unit test design
- Whether FIFO and buffer descriptor constraints share linearization code

## Deferred Ideas

None — discussion stayed within phase scope
