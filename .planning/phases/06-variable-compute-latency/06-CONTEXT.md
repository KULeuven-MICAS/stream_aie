# Phase 6: Variable Compute Latency - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Make computation node latency in the CO slot constraints tile-dependent. Remove `CoreCostLUT` and `CoreCostEstimationStage` entirely from the codebase, replacing all 18 consumer files with a new tile-aware interface that computes latency on demand from node dimensions, kernel utilization, and tile sizes. The CO model uses per-candidate linear expressions for compute latency following the established joint candidate pattern.

</domain>

<decisions>
## Implementation Decisions

### CO Compute Latency Linearization
- **D-01:** Per-candidate compute latency is computed inline during CO model construction, not pre-computed in a LUT. For each tile candidate combination, `latency[k] = ceil(MACs[k] / ops_per_cycle)` where `MACs[k] = prod(node_dims) / inter_core_tiling[k]` and `ops_per_cycle = floor(ideal_ops_per_cycle * utilization / 100)`. The utilization comes from the Kernel object associated with the node.
- **D-02:** The CO constraint becomes `slot_latency[s] >= sum_k(latency[k] * jw[k])` — a direct linear expression over joint candidate binary variables. No big-M auxiliary needed because there is no z_stop or path-choice interaction (compute latency is independent of reuse depth and transfer path).
- **D-03:** For multi-dimensional computation nodes, the joint candidate enumeration follows the same `_joint_candidates_for_tensor` pattern from Phase 3. Each candidate combination determines the inter-core tiling factors, which determine MACs, which determine latency.

### Removal of CoreCostLUT
- **D-04:** `CoreCostLUT` (`stream/cost_model/core_cost_lut.py`) and `CoreCostEstimationStage` (`stream/stages/estimation/core_cost_estimation.py`) are removed entirely. No backward compatibility layer. The cost_lut parameter is removed from all function signatures and constructors that accept it.
- **D-05:** All 18 files currently consuming cost_lut are migrated to use the new tile-aware interface (see D-06). This includes: the CO allocator, GA allocator, fitness evaluator, steady-state scheduler, visualization modules (perfetto, traces, heatmaps), and pipeline pass-throughs.

### New Tile-Aware Latency Interface
- **D-06:** A new class/module replaces CoreCostLUT. It takes a computation node, core, and tile sizes as input and returns latency (and optionally energy, ideal_cycle) on demand. It encapsulates the `ceil(MACs / ops_per_cycle)` formula and Kernel utilization lookup. All current cost_lut consumers call this instead.
- **D-07:** The interface works for both fixed-tile (concrete tile sizes passed in) and variable-tile (CO calls it per candidate during model build) cases. There is no separate "variable mode" — the same interface serves both.

### Kernel Utilization Access
- **D-08:** The Kernel object (with its `utilization` attribute) is accessed via the node's associated kernel in the mapping. The tile-aware interface accepts or looks up the kernel for each computation node to get `utilization` and `ideal_ops_per_cycle`.

### Claude's Discretion
- Exact class name and module location for the new tile-aware interface
- Whether to keep `CoreCostEntry` as a return type or replace with a simpler structure
- Internal organization of the migration across 18 files (one plan or multiple)
- How to handle the ZigZag cost estimator path (if it exists as an alternative to AIE estimator)
- Whether visualization consumers should cache computed values or always compute on demand
- The slot_latency upper bound computation (line 1397) also needs per-candidate max

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### CO Allocator (Primary Modification Target)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1247-1252 — `_slot_latency_constraints()` where compute node latency is consumed as scalar
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1393-1399 — slot_latency upper bound computation using cost_lut
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` line 71 — `cost_lut` parameter in `__init__`

### Cost Model (To Be Removed/Replaced)
- `stream/cost_model/core_cost_lut.py` — `CoreCostLUT` class; `get_cost()`, `get_cores()`, `add_cost()` methods
- `stream/cost_model/core_cost.py` — `CoreCostEntry` dataclass; `latency_total`, `ideal_cycle`, `energy_total` fields; `get_cycles_scaled(utilization)` method
- `stream/stages/estimation/core_cost_estimation.py` — `CoreCostEstimationStage` that populates cost_lut
- `stream/cost_model/aie_cost_estimator.py` — AIE latency formula: `ceil(macs / floor(ideal_ops * utilization/100))`

### Kernel (Utilization Source)
- `stream/compiler/kernels/aie_kernel.py` — `AIEKernel` base class with `utilization: float`; subclasses: `GemmKernel`, `SiluKernel`, `EltwiseMulKernel`, etc.

### Non-CO Consumers (All Must Migrate)
- `stream/cost_model/steady_state_scheduler.py` — reads `latency_total` for scheduling
- `stream/opt/allocation/genetic_algorithm/fitness_evaluator.py` — reads `ideal_cycle`, `energy_total` for GA fitness
- `stream/stages/allocation/genetic_algorithm_allocation.py` — passes cost_lut to GA
- `stream/stages/set_fixed_allocation_performance.py` — reads `ideal_cycle`, scales by utilization
- `stream/visualization/perfetto.py`, `stream/visualization/utils.py`, `stream/visualization/steady_state_trace.py`, `stream/visualization/cost_model_evaluation_lut.py` — visualization consumers
- `stream/stages/allocation/constraint_optimization_allocation.py`, `stream/opt/allocation/constraint_optimization/allocation.py`, `stream/opt/allocation/constraint_optimization/utils.py` — CO pipeline pass-throughs
- `stream/api.py` — threads cost_lut through pipeline
- `stream/stages/set_fixed_allocation.py` — validates fixed allocations exist in cost_lut
- `stream/stages/estimation/zigzag_cost_estimator.py` — ZigZag estimator creating CoreCostEntry

### Phase 3-5 Infrastructure (Reusable Patterns)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `_joint_candidates_for_tensor()`, `_joint_binary_for_combo()`, `_add_binary_scaled_continuous()`
- `stream/opt/search_space.py` — `SearchSpace` and `TileSizeOption`

### Tests
- `tests/unit/test_co_tile_variables.py` — Existing CO tile variable tests; Phase 6 extends
- `tests/regression/test_baseline.py` — Regression tests; degenerate single-candidate must pass

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_joint_candidates_for_tensor(tensor, transfer)`: Enumeration pattern for multi-dim candidates — reuse for compute node dimension enumeration
- Joint candidate binary products via `_add_binary_product` and `_joint_binary_for_combo`
- AIE latency formula in `aie_cost_estimator.py`: `ceil(macs / floor(ideal_ops * utilization/100))` — extract into the new interface

### Established Patterns
- `isinstance(rl_check, list)` for detecting variable vs scalar mode — may not be needed here if cost_lut is removed entirely
- Per-candidate coefficient computation followed by `quicksum(coeff[k] * jw[k])` — direct application for compute latency
- No z_stop interaction — compute latency is simpler than transfer latency (Phase 5)

### Integration Points
- `_slot_latency_constraints()` is the primary CO method to modify
- Slot latency upper bound (line 1397) also needs the per-candidate max for variable bounds
- Pipeline stages in `api.py` thread cost_lut — need to be updated to thread the new interface instead
- GA allocator path is separate from CO — needs its own migration

</code_context>

<specifics>
## Specific Ideas

- The compute latency linearization is the simplest of all phases — just `sum_k(lat[k] * jw[k])` as a direct linear expression in the slot latency constraint. No big-M auxiliary, no z_stop interaction, no path choice gating.
- The new tile-aware interface should encapsulate the full AIE cost estimation formula so it can be called both during CO model build (per candidate) and post-solve (with resolved tiles).
- The ZigZag estimator path creates CoreCostEntry with different fields — the new interface should handle both AIE and ZigZag estimation, or at minimum document which estimator is used for SwiGLU.
- Removing CoreCostEstimationStage from the pipeline means the pipeline no longer needs a "cost estimation" step before allocation. The allocation stage itself computes what it needs.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 06-variable-compute-latency*
*Context gathered: 2026-04-08*
