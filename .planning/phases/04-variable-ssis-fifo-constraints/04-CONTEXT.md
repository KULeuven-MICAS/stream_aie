# Phase 4: Variable SSIS + FIFO Constraints - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Make SSIS loop sizes (kernel and temporal), reuse levels, fire counts, tiles_needed, bds_needed, and object FIFO depth constraints tile-dependent linear expressions in the CO model. Refactor the SteadyStateIterationSpace class to be tile-aware: each dimension has K (kernel), S (spatial, fixed), and T (temporal) loops where K × S × T = workload_size. Regression test must pass with single-candidate degenerate input.

</domain>

<decisions>
## Implementation Decisions

### Tile-Aware SSIS Class
- **D-01:** Replace (not duplicate) the existing SteadyStateIterationSpace class with a tile-aware version that handles both fixed and variable tile cases. When a dimension has one candidate, K and T are scalar constants (backward compatible with current behavior). When a dimension has multiple candidates, K and T are candidate-indexed coefficients that become linear expressions in the CO model.
- **D-02:** Each dimension's loop decomposition follows K × S × T = workload_size, where S (spatial unrolling across cores) is fixed per dimension. Therefore K × T = workload_size / S. The kernel loop size K equals the candidate tile size; the temporal loop size T = workload_size / (S × K).
- **D-03:** The existing `generate_steady_state_iteration_spaces` function is refactored to produce the new tile-aware objects, populating them with fixed values when no SearchSpace is present. Same entry point, richer output.

### Linearization Pattern for SSIS-Derived Quantities
- **D-04:** `reuse_levels`, `tiles_needed_levels`, `bds_needed_levels`, and fire counts become linear expressions over tile selection variables. For each (tensor, stop_level), the value is `sum_k(coeff[k] * joint_binary_var[k])` where coefficients are pre-computed per candidate combination.
- **D-05:** The interaction with z_stop is handled via continuous auxiliary variables with big-M activation — same pattern as Phase 3's memory constraints (D-05 from Phase 3). Continuous variable gated by z_stop. No triple product with `u` here, so it's simpler than the memory case.
- **D-06:** Tight per-constraint big-M bounds follow Phase 3's strategy: max over candidate combinations, computed as a byproduct of the coefficient enumeration.

### SSIS Invariant Check
- **D-07:** `_ensure_same_ssis_for_all_transfers` (currently line 245) moves to post-solve verification. With variable tiles, the actual loop sizes aren't known until w[dim,k] is resolved by the solver. The structural property (all transfers share the same loop nest) still holds by construction since K × S × T = workload_size for each dimension regardless of tile selection.

### Utility Functions (TILE-05 Scope for Phase 4)
- **D-08:** Phase 4 adds utility functions for computing SSIS loop sizes (K, T), reuse levels, fire counts, and tiles_needed per candidate tile combination. These are the TILE-05 utilities deferred from Phase 3.
- **D-09:** These utilities work with the tile-aware SSIS class to produce the candidate-indexed coefficients used in linearization.

### Claude's Discretion
- Internal structure of the refactored SSIS class (method signatures, properties)
- How `_init_transfer_fire_helpers` is restructured to produce linear expressions
- Unit test design for tile-aware SSIS and linearized FIFO constraints
- Whether `_object_fifo_depth_constraints` and `_buffer_descriptor_constraints` share linearization code or remain separate

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### SSIS (Primary Modification Target)
- `stream/workload/steady_state/iteration_space.py` — `SteadyStateIterationSpace` class; holds kernel/temporal loop sizes, `reuse_factor()`, `get_applicable_temporal_variables()`, `get_temporal_sizes()`, `get_temporal_reuses()`
- `stream/workload/utils.py` — `generate_steady_state_iteration_spaces()` creates SSIS objects from fusion splits; `determine_fusion_splits()` maps intra_core_tiling to split factors; `collect_spatial_unrollings()` returns spatial unrolling per dimension

### CO Allocator (Consumer of SSIS)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `_init_transfer_fire_helpers()` (line 259) computes `reuse_levels`, `tiles_needed_levels`, `bds_needed_levels` from SSIS; `_object_fifo_depth_constraints()` (line 842) uses `tiles_needed_levels`; `_buffer_descriptor_constraints()` (line 867) uses both `tiles_needed_levels` and `bds_needed_levels`; `_ensure_same_ssis_for_all_transfers()` (line 245) validates SSIS consistency; `_force_nonconstant_reuse_levels()` (line 942)

### Phase 3 Infrastructure (Reusable Patterns)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `__create_tile_selection_vars()` creates w[dim,k] + tile_var[dim]; `_joint_candidates_for_tensor()` enumerates joint candidate combinations; `_joint_binary_for_combo()` recursive binary product; `_memory_capacity_constraints()` continuous auxiliary with big-M pattern
- `stream/opt/tile_size_utils.py` — `tensor_size_bits_for_candidate()`, `max_tensor_size_bits()`; Phase 4 adds SSIS loop size utilities here
- `stream/opt/search_space.py` — `SearchSpace` and `TileSizeOption` dataclasses

### Mapping & Workload
- `stream/mapping/mapping.py` — `FusedGroup.intra_core_tiling` stores (LayerDim, tile_size) tuples
- `stream/workload/workload.py` — `get_dimension_size()`, `unique_dimensions()`, `get_tensor_shape_with_tiling()`

### Tests
- `tests/unit/test_co_tile_variables.py` — Existing CO tile variable tests; Phase 4 extends these
- `tests/regression/test_baseline.py` — Regression tests; degenerate single-candidate must pass

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_joint_candidates_for_tensor(tensor, transfer)`: Returns `list[(size_bits, joint_binary_var)]` — same enumeration pattern needed for SSIS coefficients
- `_add_binary_product(a, b)`: Binary product linearization helper — reuse for z_stop × joint_binary products
- Continuous auxiliary with big-M pattern from `_memory_capacity_constraints`: Three constraints (lc <= expr, lc <= M*binary, lc >= expr - M*(1-binary)) — reuse for SSIS-derived quantities × z_stop
- `collect_spatial_unrollings(workload, mapping)`: Returns spatial unrolling per dimension — needed for K × S × T = workload_size decomposition

### Established Patterns
- `self.reuse_levels[(t, stop)]` = `(fires, size_factor)` tuple — will become candidate-indexed coefficients
- `self.tiles_needed_levels[(t, stop)]` = scalar int — will become linear expression
- `self.bds_needed_levels[(t, stop)]` = scalar int — will become linear expression
- Loop over `range(-1, len(ssis[tr].get_applicable_temporal_variables()))` for stop levels

### Integration Points
- `_init_transfer_fire_helpers` is the primary method to refactor — transforms from scalar computation to candidate-indexed coefficient computation
- `_object_fifo_depth_constraints` and `_buffer_descriptor_constraints` consume the linearized values
- `_force_nonconstant_reuse_levels` may need adaptation for tile-dependent reuse
- Post-solve result extraction (`get_tensor_reuse_levels`, `get_object_fifo_depths_per_transfer`) must extract values from solved w[dim,k] selections

</code_context>

<specifics>
## Specific Ideas

- The tile-aware SSIS class should not require knowing the selected tile at construction time. It should provide methods to compute K and T for any given candidate tile value, allowing the CO model builder to pre-compute coefficients for all candidates.
- For the degenerate single-candidate case, K and T collapse to fixed scalars, and all downstream quantities (reuse_levels, tiles_needed, etc.) become the same scalar constants as today — ensuring regression compatibility.
- The `_ensure_same_ssis_for_all_transfers` check should move to post-solve, where the actual tile selection is known and the concrete loop sizes can be compared.
- K × S × T = workload_size is the fundamental invariant. S is spatial unrolling (fixed). K = tile_size (variable). T = workload_size / (S × K).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-variable-ssis-fifo-constraints*
*Context gathered: 2026-04-07*
