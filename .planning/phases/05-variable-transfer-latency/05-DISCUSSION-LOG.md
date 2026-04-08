# Phase 5: Variable Transfer Latency - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 05-variable-transfer-latency
**Areas discussed:** Latency numerator linearization, Triple product: y x latency_expr / reuse_factor, Scope boundary: what else is tile-dependent?

---

## Latency Numerator Linearization

| Option | Description | Selected |
|--------|-------------|----------|
| Same pattern (Recommended) | Reuse _joint_candidates_for_tensor, compute latency coefficient per candidate. Straightforward extension of Phase 3. | Y |
| Additional complexity | Something else about how latency depends on tile sizes. | |

**User's choice:** Same pattern -- reuse joint candidate enumeration with ceil(tensor_size[k] / min_bw) per candidate.
**Notes:** Straightforward extension, no additional complexity flagged.

---

## Triple Product: y x latency_expr / reuse_factor

### Initial approach selection

| Option | Description | Selected |
|--------|-------------|----------|
| Option C: enumerate (tile, stop) pairs | Pre-compute amortized_latency[k,s] for each (candidate, stop_level) pair. Pure MILP. | |
| Option A: approximate with tile-only ratio | Pre-compute ratio[k] assuming fixed reuse factor per candidate. Simpler but ignores z_stop. | |
| Keep existing ratio structure | Adapt _add_binary_times_const_over_linexpr to accept linear expression numerator. | Y |

**User's choice:** Keep existing ratio structure -- adapt the helper for linexpr numerator.

### Follow-up: NL constraint discovery

| Option | Description | Selected |
|--------|-------------|----------|
| Extend addGenConstrNL to linexpr/linexpr | Replace float(numerator)/den with num_var/den. Same Gurobi NL mechanism. | |
| Eliminate addGenConstrNL entirely | Replace division with pure MILP linearization. Makes model a true MILP. | Y |

**User's choice:** Eliminate addGenConstrNL entirely -- make the model a true MILP.

### Follow-up: Pure MILP formulation

| Option | Description | Selected |
|--------|-------------|----------|
| Enumerate (k,s) pairs | Pre-compute all amortized latency coefficients, select via binary products. Eliminates addGenConstrNL. | Y |
| Different approach | Something else in mind for eliminating NL constraint. | |

**User's choice:** Yes, enumerate (tile_candidate, stop_level) pairs with pre-computed amortized_latency[k,s] = ceil(size[k]/bw) / reuse[k,s]. Pure MILP.
**Notes:** This is effectively Option C from the initial selection, arrived at after discovering the addGenConstrNL constraint needed elimination.

---

## Scope Boundary: What Else Is Tile-Dependent?

| Option | Description | Selected |
|--------|-------------|----------|
| That covers it | Transfer latency is the last tile-dependent scalar. | |
| There's more | Additional tile-dependent quantities need linearization. | |
| Double-check the code | User requested thorough verification before confirming. | Y |

**User's choice:** Requested a thorough code scan before confirming scope.
**Notes:** Deep scan of all constraint methods confirmed: transfer latency is the ONLY remaining tile-dependent scalar. All other quantities (memory, SSIS, reuse, fires, tiles_needed, bds_needed, FIFO, BD) properly linearized in Phases 3-4. DMA, force_nonconstant_reuse_levels, objective, and variable bounds are tile-independent.

---

## Claude's Discretion

- Whether to keep or remove _add_const_over_linexpr / _add_binary_times_const_over_linexpr after they become dead code
- Internal structure of (k, s) pair enumeration loop
- Latency cache restructuring
- Unit test design

## Deferred Ideas

None -- discussion stayed within phase scope.
