# Phase 8: Latency Computation Parity - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix TileAwareLatencyEstimator and `_slot_latency_constraints` to produce identical per-node latency (MACs/cycles) as the old CoreCostLUT + AIECostEstimator path for the same tile sizes. Restore the single-candidate regression test to match the original Phase 1 baseline exactly (latency_total=922357343, latency_per_iteration=10716). The CO model must remain feasible and optimal for the degenerate single-candidate case.

</domain>

<decisions>
## Implementation Decisions

### Double-Counting Fix
- **D-01:** The fix is localized to `_slot_latency_constraints` in `transfer_and_tensor_allocation.py`. The method must compute inter_core_tiling factors correctly for the untiled workload (since Phase 7 moved TilingGenerationStage post-solve). The tiling factor should be derived as `workload_size / (tile * spatial)`, not read from a pre-tiled mapping that no longer exists at CO time.
- **D-02:** `fusion_splits` in a fusion group are actually the temporal T loop sizes (`workload_size / (tile * spatial)`). They must be derived correctly from tile candidates, not stored as fixed values. If currently defined as fixed, the mapping representation needs updating so the CO can compute them from variable tile sizes.

### MACs Parity
- **D-03:** MACs are computed from first principles: `MACs = prod(node_dimension_sizes) / tiling_factor`, where `tiling_factor = prod(spatial * tile)` per tiled dimension. Any mismatch with the old path indicates a bug in understanding, not in code. No attempt to replicate old-path quirks.

### Regression Test
- **D-04:** Validate using hardcoded baseline values: assert `latency_total=922357343` and `latency_per_iteration=10716` for the single-candidate BIG BOY config. Matches success criteria directly.

### Debugging Workflow
- **D-05:** Add temporary per-node MACs/latency logging to both the estimator and `_slot_latency_constraints`. Run the single-candidate case and compare node-by-node. Remove logging after fix is confirmed.

### Claude's Discretion
- How to represent variable fusion_splits in the mapping (D-02 implementation detail)
- Whether to refactor `get_unique_dims_inter_core_tiling` or bypass it with a local computation
- Internal organization of the fix (single plan or multiple)
- Whether temporary logging is committed or kept as debug-only

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### CO Allocator (Primary Fix Target)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1249-1298 -- `_slot_latency_constraints()` where temporal split double-counting occurs
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1254-1278 -- Variable tile mode: candidate enumeration and tiling factor substitution

### Latency Estimator
- `stream/cost_model/tile_aware_latency.py` -- `TileAwareLatencyEstimator.estimate()` and MACs formula
- `stream/stages/estimation/aie_cost_estimator.py` -- Original `AIECostEstimator` formula for reference parity

### Mapping & Fusion Groups
- `stream/mapping/mapping.py` -- `FusedGroup`, `intra_core_tiling`, fusion_splits representation
- `stream/workload/workload.py` -- `get_unique_dims_inter_core_tiling()`, `get_dimension_size()`

### Formulation Reference
- `docs/variable_tile_co_formulation.md` -- Variable tile CO formulation documentation

### Tests
- `tests/unit/test_co_tile_variables.py` lines 1846-1931 -- `_make_slot_latency_stub` and existing slot latency tests
- `tests/regression/test_baseline.py` -- Regression test with baseline values

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TileAwareLatencyEstimator`: Already computes `ceil(MACs / ops_per_cycle)` -- the formula itself is correct, the issue is what MACs value gets fed in
- `_joint_binary_for_combo` / `_joint_candidates_for_tensor`: Candidate enumeration pattern reused in `_slot_latency_constraints`
- `_make_slot_latency_stub` in tests: Existing test infrastructure for slot latency constraint testing

### Established Patterns
- Per-candidate coefficient computation: `quicksum(coeff[k] * jw[k])` -- already in place in `_slot_latency_constraints`
- `workload.get_unique_dims_inter_core_tiling(n, mapping)` returns `(dim, factor)` pairs -- but these may be incorrect for untiled workload

### Integration Points
- `_slot_latency_constraints()` is the primary method to fix
- `TileAwareLatencyEstimator.estimate()` receives `inter_core_tiling` as argument -- may need correct values passed in
- Regression test in `test_baseline.py` validates end-to-end correctness

</code_context>

<specifics>
## Specific Ideas

- The root cause is the Phase 7 pipeline reorder: CO now runs on untiled workload, but `_slot_latency_constraints` still reads tiling factors as if the workload were pre-tiled. The candidate tile substitution then double-divides.
- fusion_splits = temporal T loop sizes = `workload_size / (tile * spatial)`. These are NOT fixed constants -- they depend on the selected tile size. The mapping must represent this correctly so the CO can derive them from variable tiles.
- Per-node MACs logging (temporary) will reveal exactly which node(s) have the discrepancy and by what factor, making the fix straightforward.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 08-latency-computation-parity*
*Context gathered: 2026-04-08*
