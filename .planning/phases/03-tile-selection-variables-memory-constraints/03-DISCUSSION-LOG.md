# Phase 3: Tile Selection Variables + Memory Constraints - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-07
**Phase:** 03-tile-selection-variables-memory-constraints
**Areas discussed:** MILP formulation, Triple product linearization, Tile utility scope, Big-M strategy

---

## MILP Formulation for Tile-Dependent Sizes

| Option | Description | Selected |
|--------|-------------|----------|
| Pure binary expansion | Pre-compute tensor_size(t,k) per candidate, express as sum_k(size[t,k] * w[dim,k]). Tile value implicit. | |
| INTEGER auxiliary variable | Create tile_var[dim] as INTEGER, constrain via tile_var[dim] = sum_k(tile_value[k] * w[dim,k]). More readable model. | ✓ |

**User's choice:** INTEGER auxiliary variable (Option B)
**Notes:** User preferred the more readable model with explicit tile size representation.

---

## Triple Product Linearization (u × z_stop × w)

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-expand triple product | For each (t, stop, k), compute uwz = binary_product(binary_product(u, z_stop), w[dim,k]). Scalar coefficients only but O(candidates × stops × tensors × cores) auxiliaries. | |
| Nested two-stage | First uz = binary_product(u, z_stop), then sum_k(size[t,k] * binary_product(uz, w[dim,k])). Same variable count as A, grouped differently. | |
| Big-M continuous auxiliary | Continuous load_contrib[t,c,stop] equals linear_expr * uz via big-M activation. Fewer binary auxiliaries, needs tight big-M bounds. | ✓ |

**User's choice:** Big-M continuous auxiliary (Option C)
**Notes:** Trades binary auxiliaries for continuous ones with big-M. Keeps binary variable count manageable.

**Additional requirement:** User emphasized that multi-dimensional tensors (tensors whose size depends on multiple tiled dimensions) must be handled via recursive joint-candidate enumeration. The linearization must work cleanly for any number of dimensions without special-casing, keeping the model a pure MILP.

---

## Scope of Tile Utility Functions in Phase 3

| Option | Description | Selected |
|--------|-------------|----------|
| Tensor size + big-M only | Phase 3 adds tensor-size-per-candidate and max-over-candidates helper. SSIS, reuse, latency deferred to Phases 4-5. | ✓ |
| Pull in SSIS too | Also add SSIS loop size utilities in Phase 3. | |

**User's choice:** Tensor size + big-M only
**Notes:** Matches TILE-05 incremental strategy. Each phase adds only the utilities it needs.

---

## Tight Big-M Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Computed once at model construction | Iterate SearchSpace to compute max_size[t] before adding constraints. Byproduct of joint candidate enumeration. | ✓ |
| Computed inline per constraint | Each constraint queries SearchSpace for tight bound on the fly. Repeated computation. | |

**User's choice:** Computed once at model construction (Option A)
**Notes:** Natural fit since max bounds fall out of the same joint candidate enumeration used for linearization coefficients.

---

## Claude's Discretion

- Method placement within TransferAndTensorAllocator
- Variable naming conventions for w, tile_var, and auxiliaries
- Internal structure of joint candidate enumeration
- Unit test design

## Deferred Ideas

None — discussion stayed within phase scope
