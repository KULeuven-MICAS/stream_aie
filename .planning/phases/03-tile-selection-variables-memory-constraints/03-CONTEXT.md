# Phase 3: Tile Selection Variables + Memory Constraints - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Introduce binary tile selection variables w[dim,k] and INTEGER auxiliary tile_var[dim] into the Gurobi MILP model. Linearize memory capacity constraints over tile selection using continuous auxiliaries with big-M activation. Handle multi-dimensional tensors via recursive joint-candidate enumeration to keep the model a pure MILP. Regression test must pass with single-candidate degenerate input recovering the fixed-tile baseline.

</domain>

<decisions>
## Implementation Decisions

### MILP Formulation
- **D-01:** INTEGER auxiliary variable `tile_var[dim]` created for each unique workload dimension, constrained via `tile_var[dim] = sum_k(tile_value[k] * w[dim,k])`. The tile size value is explicitly represented in the model for readability.
- **D-02:** Binary w[dim,k] variables with one-hot constraint (`sum_k w[dim,k] == 1`) for each unique workload dimension. Candidates come from SearchSpace passed via StageContext.

### Multi-Dimensional Tensor Linearization
- **D-03:** For tensors whose size depends on multiple tiled dimensions, enumerate all joint candidate combinations across those dimensions. Each joint combination gets a pre-computed tensor size coefficient. The joint selection is represented as a product of the individual w[dim,k] binaries, linearized recursively via binary product auxiliaries (nested `_add_binary_product` or equivalent). This keeps the model a pure MILP with scalar coefficients x binary variables, regardless of tensor dimensionality.
- **D-04:** The recursive enumeration must work cleanly for any number of tiled dimensions (1, 2, 3+) without special-casing. The combinatorial blow-up is acceptable because candidate lists are small (typically 3-10 per dimension) and tensor dimensionality is low (2-3).

### Memory Constraint Linearization (Triple Product)
- **D-05:** The interaction between tensor-uses-core (u), reuse level (z_stop), and tile selection (w) is handled via continuous auxiliary variables with big-M activation (Option C). This avoids creating one binary auxiliary per (tensor, stop, candidate, core) combination. Instead, a continuous `load_contrib[t,c,stop]` equals `(sum_k size[t,k] * w[k]) * uz` via big-M constraints gated on the uz binary product.
- **D-06:** Big-M bounds for these continuous auxiliaries are derived from the tight per-constraint max over joint candidate tensor sizes, not the legacy scalar `self.big_m`.

### Tight Big-M Strategy
- **D-07:** Max tensor size bounds computed once at model construction time, as a byproduct of the joint candidate enumeration in D-03. Stored in a dict keyed by tensor (or tensor + relevant constraint key). No extra pass needed — the max falls out of the same enumeration that produces the linearization coefficients.
- **D-08:** These tight per-constraint bounds replace the legacy `self.big_m = len(nodes) + 5` for memory-related constraints. Other big-M uses (idle indicators, etc.) retain the legacy value until Phases 4-5 address them.

### Tile Utility Functions (Phase 3 Scope)
- **D-09:** Phase 3 adds only the utilities needed for memory constraints: a tensor-size-per-candidate function that computes tensor size given a specific candidate tile value for a dimension, and a max-over-candidates helper for big-M bounds.
- **D-10:** SSIS loop sizes, reuse levels, fire counts, and transfer latencies are deferred to Phases 4-5 per TILE-05 incremental strategy.

### Claude's Discretion
- Exact method placement within TransferAndTensorAllocator (new methods vs modifying existing ones)
- Variable naming conventions for w, tile_var, and auxiliary variables in the Gurobi model
- Internal structure of the joint candidate enumeration (generator vs materialized list)
- Unit test design for the new CO variables and constraints

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### CO Allocator (Primary Modification Target)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `TransferAndTensorAllocator` class; `_create_vars()` (line 520), `__create_reuse_vars()` (line 530), `_memory_capacity_constraints()` (line 720), `_object_fifo_depth_constraints()` (line 745), `_add_binary_product()` helper
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 720-743 — Current memory capacity constraint implementation with scalar tensor_size and uz binary product pattern

### Search Space & Tile Utilities (Input to Phase 3)
- `stream/opt/search_space.py` — `SearchSpace` and `TileSizeOption` dataclasses; keyed by unique LayerDim
- `stream/opt/tile_size_utils.py` — `tensor_size_bits()`, `is_divisible_candidate()`, `passes_single_tensor_memory_check()`

### Tensor Size Computation Chain
- `stream/workload/tensor.py` — `Tensor.size_bits()` computes tensor size as `bitwidth * prod(shape)`
- `stream/workload/workload.py` — `get_tensor_shape_with_tiling()` applies inter-core tiling; `get_dimension_size()` returns workload dimension size; `unique_dimensions()` for dimension grouping

### Pipeline & Stage Context
- `stream/api.py` — `optimize_allocation_co()` stage pipeline; SearchSpace available via StageContext
- `stream/stages/generation/candidate_filter_stage.py` — `CandidateFilterStage` builds SearchSpace and sets it in ctx

### Regression Baseline
- `tests/regression/test_baseline.py` — Regression tests; single-candidate degenerate input must still pass after Phase 3 changes

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_add_binary_product(a, b)`: Existing linearization helper for product of two binary variables — reuse for nested/recursive joint candidate products
- `SearchSpace.get(dim)` → `list[TileSizeOption]`: Direct access to candidate tiles per dimension; each option has `.tile` and `.workload_size`
- `tensor_size_bits(workload, tensor, inter_core_tiling)`: Existing tensor size computation; needs a variant that substitutes a candidate tile value for a specific dimension

### Established Patterns
- `z_stop[(t, stop)]` binary variables with one-hot constraint per transfer — w[dim,k] follows the same pattern
- `self.reuse_levels[(t, stop)]` pre-computed per reuse level — joint candidate coefficients follow the same pre-computation pattern
- `self.core_load[c] += req_size * uz` — memory accumulation pattern; will change to use continuous auxiliary with big-M

### Integration Points
- SearchSpace retrieved from `StageContext` in the allocator's `__init__` or setup
- w[dim,k] variables created in `_create_vars()` alongside existing z_stop, x_tensor_choice, y_path_choice
- Memory constraints modified in `_memory_capacity_constraints()` to use linearized tile-dependent expressions
- Workload dimension → tensor mapping needed to know which w[dim,k] variables affect which tensor sizes

</code_context>

<specifics>
## Specific Ideas

- The recursive joint-candidate linearization for multi-dimensional tensors is the most critical design element. A tensor with dimensions [embedding, hidden] each having 5 candidates yields 25 joint candidates — each with a pre-computed size. The joint binary selection is `w[emb,k1] AND w[hid,k2]`, linearized via binary product auxiliaries. This generalizes cleanly to 3+ dimensions.
- For the degenerate single-candidate case, w[dim,0] = 1 is forced by the one-hot constraint, tile_var[dim] = the single tile value, and all coefficients collapse to the fixed-tile scalar — ensuring regression compatibility.
- The big-M tight bound for a tensor is simply `max(size[t, joint_k] for joint_k in all_joint_candidates)` — computed during the same enumeration that produces linearization coefficients.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-tile-selection-variables-memory-constraints*
*Context gathered: 2026-04-07*
