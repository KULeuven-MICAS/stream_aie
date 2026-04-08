# Phase 06: Variable Compute Latency - Research

**Researched:** 2026-04-08
**Domain:** Gurobi MILP linearization, AIE cost estimation, CoreCostLUT removal
**Confidence:** HIGH

## Summary

Phase 6 has two goals: (1) replace the fixed-scalar compute latency in `_slot_latency_constraints` with a per-candidate linear expression over joint binary variables, and (2) remove `CoreCostLUT` / `CoreCostEstimationStage` entirely and migrate all 15 consumer files to a new tile-aware latency interface.

The CO constraint change is the simplest linearization in the whole project. Compute latency has no z_stop interaction and no path-choice gating — it is a pure `quicksum(lat[k] * jw[k])` expression directly used in `addConstr(slot_latency[s] >= expr)`. No auxiliary variable is needed. The same joint candidate enumeration infrastructure (`_joint_candidates_for_tensor`, `_joint_binary_for_combo`) from Phases 3–5 applies here, but driven by the node's dimensions and inter-core tiling factors rather than by a transfer tensor.

The CoreCostLUT removal is the larger structural work: 15 files currently import or accept `cost_lut`. The GA path is already `raise NotImplementedError` in production, so its consumers can be stripped of the `cost_lut` parameter without functional risk. Visualization consumers need compute latency only for display; the new tile-aware interface satisfies them if called post-solve with the resolved tile. The `steady_state_scheduler.py` has an `update_cost_lut()` method that renames nodes in the LUT after the transfer graph is built; this will be replaced by doing nothing (or a no-op) once the LUT is gone.

**Primary recommendation:** Build a single `TileAwareLatencyEstimator` module alongside `aie_cost_estimator.py`, keep the AIE formula in one place, wire it into `_slot_latency_constraints` for the CO per-candidate path, and thread it (instead of `cost_lut`) through the pipeline stages.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Per-candidate compute latency is computed inline during CO model construction, not pre-computed in a LUT. For each tile candidate combination, `latency[k] = ceil(MACs[k] / ops_per_cycle)` where `MACs[k] = prod(node_dims) / inter_core_tiling[k]` and `ops_per_cycle = floor(ideal_ops_per_cycle * utilization / 100)`. The utilization comes from the Kernel object associated with the node.
- **D-02:** The CO constraint becomes `slot_latency[s] >= sum_k(latency[k] * jw[k])` — a direct linear expression over joint candidate binary variables. No big-M auxiliary needed because there is no z_stop or path-choice interaction.
- **D-03:** For multi-dimensional computation nodes, the joint candidate enumeration follows the same `_joint_candidates_for_tensor` pattern from Phase 3.
- **D-04:** `CoreCostLUT` (`stream/cost_model/core_cost_lut.py`) and `CoreCostEstimationStage` (`stream/stages/estimation/core_cost_estimation.py`) are removed entirely. No backward compatibility layer.
- **D-05:** All 15 files currently consuming cost_lut are migrated to use the new tile-aware interface.
- **D-06:** A new class/module replaces CoreCostLUT. It takes a computation node, core, and tile sizes as input and returns latency (and optionally energy, ideal_cycle) on demand. It encapsulates the `ceil(MACs / ops_per_cycle)` formula and Kernel utilization lookup.
- **D-07:** The interface works for both fixed-tile and variable-tile cases with no separate mode.
- **D-08:** Kernel utilization accessed via `mapping.get(node).kernel.utilization`; `ideal_ops_per_cycle` from `AIECostEstimator.ops_per_cycle()`.

### Claude's Discretion

- Exact class name and module location for the new tile-aware interface
- Whether to keep `CoreCostEntry` as a return type or replace with a simpler structure
- Internal organization of the migration across 15 files (one plan or multiple)
- How to handle the ZigZag cost estimator path
- Whether visualization consumers should cache computed values or always compute on demand
- The slot_latency upper bound computation (line 1397) also needs per-candidate max

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CO-06 | Computation node latency in slot constraints becomes a linear expression over tile selection variables, using kernel size and Kernel utilization to compute per-candidate cycle counts | `_slot_latency_constraints` modification (lines 1247–1252) + slot_latency UB (lines 1393–1399) + per-candidate MACs formula from `AIECostEstimator.estimate()` |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gurobipy | project default | MILP constraint building | Used throughout CO allocator |
| math.ceil / math.floor | stdlib | AIE latency formula `ceil(MACs / floor(ops * util/100))` | Exact formula from `aie_cost_estimator.py` |
| itertools.product | stdlib | Joint candidate enumeration over node dims | Same as Phase 3–5 pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | stdlib | Return type for new tile-aware estimator | Lightweight replacement for CoreCostEntry |

**No new packages needed.** All required logic is already present in the repository.

---

## Architecture Patterns

### New Module: TileAwareLatencyEstimator

**Location (discretion):** `stream/cost_model/tile_aware_latency.py`  
This keeps it adjacent to the files it replaces (`core_cost.py`, `core_cost_lut.py`).

**Interface:**
```python
# stream/cost_model/tile_aware_latency.py
from math import ceil, floor, prod
from dataclasses import dataclass
from stream.stages.estimation.aie_cost_estimator import AIECostEstimator

@dataclass
class LatencyEstimate:
    latency_total: int      # ceil(MACs / effective_ops_per_cycle)
    ideal_cycle: int        # ceil(MACs / ideal_ops_per_cycle)
    energy_total: float     # 0 for AIE (TODO)

class TileAwareLatencyEstimator:
    def __init__(self, workload: Workload, mapping: Mapping):
        self._aie = AIECostEstimator(workload, mapping)

    def estimate(
        self,
        node: ComputationNode,
        core: Core,
        inter_core_tiling: InterCoreTiling,
    ) -> LatencyEstimate:
        dim_sizes = [workload.get_dimension_size(d) for d in workload.get_dims(node)]
        tiling_factor = prod(f for _, f in inter_core_tiling)
        macs = prod(dim_sizes) // tiling_factor
        kernel = mapping.get(node).kernel
        utilization = kernel.utilization
        ideal_ops = self._aie.ops_per_cycle(node, core)
        ideal_cycles = ceil(macs / ideal_ops)
        ops_per_cycle = floor(ideal_ops * utilization / 100.0)
        cycles = ceil(macs / ops_per_cycle)
        return LatencyEstimate(latency_total=cycles, ideal_cycle=ideal_cycles, energy_total=0.0)
```

**Key point:** This is exactly the existing `AIECostEstimator.estimate()` logic, extracted into a standalone callable that accepts explicit `inter_core_tiling` rather than reading it from the mapping. This is necessary so the CO can call it per-candidate with hypothetical tilings.

### CO Change: `_slot_latency_constraints` (lines 1247–1252)

**Current code:**
```python
# line 1250
runtimes = [self.cost_lut.get_cost(n, c).latency_total for c in self.cost_lut.get_cores(n)]
runtime = ceil(max(runtimes)) if runtimes else 0
self.model.addConstr(self.slot_latency[s] >= runtime, name=f"ssc_lat_{n.name}")
```

**Replacement pattern (variable tile mode):**
```python
for n in self.ssc_nodes:
    s = self.slot_of[n]
    if self.search_space is not None and not self.search_space.is_empty():
        # --- Variable tile mode: per-candidate linear expression ---
        tiled_dims = [d for d in self.search_space.dims()
                      if d in {dim for dim, _ in self.workload.get_unique_dims_inter_core_tiling(n, self.mapping)}]
        if tiled_dims:
            per_dim_options = [(dim, self.search_space.get(dim)) for dim in tiled_dims]
            lat_coeffs = []  # list of (latency_int, joint_w_var)
            for combo in iproduct(*[opts for _, opts in per_dim_options]):
                # Build inter_core_tiling for this candidate
                base_tiling = list(self.workload.get_unique_dims_inter_core_tiling(n, self.mapping))
                current_tiling = list(base_tiling)
                for (dim, _opts), opt in zip(per_dim_options, combo):
                    wdim_size = self.workload.get_dimension_size(dim)
                    new_factor = wdim_size // opt.tile
                    for i, (td, _) in enumerate(current_tiling):
                        if td == dim:
                            current_tiling[i] = (dim, new_factor)
                            break
                core = next(iter(self.mapping.get(n).resource_allocation[0]))
                lat_est = self.latency_estimator.estimate(n, core, tuple(current_tiling))
                jw = self._joint_binary_for_combo(per_dim_options, combo, base_name=f"jw_ssc_{n.name}")
                lat_coeffs.append((lat_est.latency_total, jw))

            expr = quicksum(lat * jw for lat, jw in lat_coeffs)
            self.model.addConstr(self.slot_latency[s] >= expr, name=f"ssc_lat_{n.name}")
        else:
            # No tiled dims for this node: scalar fallback
            core = next(iter(self.mapping.get(n).resource_allocation[0]))
            lat_est = self.latency_estimator.estimate(n, core, self.workload.get_unique_dims_inter_core_tiling(n, self.mapping))
            self.model.addConstr(self.slot_latency[s] >= lat_est.latency_total, name=f"ssc_lat_{n.name}")
    else:
        # Scalar mode (no search space)
        core = next(iter(self.mapping.get(n).resource_allocation[0]))
        lat_est = self.latency_estimator.estimate(n, core, self.workload.get_unique_dims_inter_core_tiling(n, self.mapping))
        self.model.addConstr(self.slot_latency[s] >= lat_est.latency_total, name=f"ssc_lat_{n.name}")
```

**MILP validity:** `lat_coeffs` are precomputed integer scalars multiplied by binary `jw` variables — this is purely linear. No auxiliary variable needed.

### CO Change: Slot Latency Upper Bound (lines 1393–1399)

The `_create_idle_latency_vars` method computes `slot_latency_ub` from cost_lut. This needs to be replaced with:
```python
# For variable mode: take max over all candidates
for n in self.ssc_nodes:
    # enumerate all candidates as above, take max latency
    max_lat = max(lat for lat, _ in lat_coeffs_for_node[n])
    slot_latency_ub = max(slot_latency_ub, max_lat)
```
This requires either caching `lat_coeffs` built during `_slot_latency_constraints`, or extracting the candidate enumeration into a helper that both methods call.

**Recommendation:** Cache per-node latency coefficients in `self._ssc_node_lat_coeffs: dict[ComputationNode, list[tuple[int, gp.Var]]]` built during `_slot_latency_constraints`, then reuse in `_create_idle_latency_vars`.

### ThreadedThrough: `latency_estimator` on `TransferAndTensorAllocator`

Add `latency_estimator: TileAwareLatencyEstimator` as a constructor parameter (replacing `cost_lut`). All call sites that currently pass `cost_lut=...` pass `latency_estimator=...` instead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ops_per_cycle for data types | Custom dispatch | `AIECostEstimator.ops_per_cycle(node, core)` | Already handles bf16 aie2/aie dispatch; extensible |
| Joint binary product for node dims | New AND-variable helper | `_joint_binary_for_combo(per_dim_options, combo)` | Identical structure to tensor candidates, already correct |
| Per-candidate enumeration | Custom loop | `iproduct` over `search_space.get(dim)` lists | Same pattern as Phase 3–5 |

---

## Common Pitfalls

### Pitfall 1: `_slot_latency_constraints` and `_create_idle_latency_vars` both iterate `ssc_nodes` independently
**What goes wrong:** `_create_idle_latency_vars` (line 1397) has a parallel loop over `ssc_nodes` to compute `slot_latency_ub`. If that loop is updated to compute `max(latency_per_candidate)` but `_slot_latency_constraints` already enumerated candidates, you need the results from that enumeration. Without caching, you enumerate twice.
**How to avoid:** Store `self._ssc_node_lat_coeffs` during `_slot_latency_constraints`; read it in `_create_idle_latency_vars`.
**Warning signs:** Both methods appear sequential in `_create_constraints()` → `_overlap_and_objective()` — the order is `_slot_latency_constraints` first, so caching works.

### Pitfall 2: Scalar mode after CoreCostLUT removal
**What goes wrong:** When `search_space` is None or the node has no tiled dims, the code previously called `cost_lut.get_cost(n, c)`. After removal this crashes unless the scalar path calls `latency_estimator.estimate()` with the fixed tiling.
**How to avoid:** The `else` branches in `_slot_latency_constraints` and `_create_idle_latency_vars` must call `latency_estimator.estimate(n, core, fixed_tiling)` using the current mapping's `inter_core_tiling`.
**Warning signs:** Any `AttributeError: 'TransferAndTensorAllocator' object has no attribute 'cost_lut'` during scalar-mode test runs.

### Pitfall 3: `update_cost_lut()` in `SteadyStateScheduler`
**What goes wrong:** `SteadyStateScheduler.run()` calls `self.cost_lut = self.update_cost_lut()` which renames workload nodes inside the LUT after the transfer graph is rebuilt. Without this the mapping lookup for new transfer-graph nodes would fail.
**How to avoid:** After removing `cost_lut`, delete `update_cost_lut()` and instead confirm that `TileAwareLatencyEstimator` always resolves nodes via the workload passed to it (or is re-constructed with the updated workload/mapping after `build_transfer_graph()`).
**Warning signs:** `AssertionError: No mapping found for node` when running the scheduler.

### Pitfall 4: `SteadyStateScheduler.__init__` accepts `cost_lut` as positional parameter
**What goes wrong:** `ConstraintOptimizationAllocationStage.find_best_tensor_transfer_allocation()` (line 78) constructs `SteadyStateScheduler(...)` with `cost_lut=self.cost_lut`. This must be replaced everywhere; the signature update must be coordinated across `steady_state_scheduler.py` and its two callers.
**How to avoid:** Track call sites:
- `stream/stages/allocation/constraint_optimization_allocation.py` line 78
- Any test fixtures that construct `SteadyStateScheduler` directly.
**Warning signs:** `TypeError: __init__() got an unexpected keyword argument 'cost_lut'` or `TypeError: __init__() missing positional argument 'cost_lut'`.

### Pitfall 5: `SetFixedAllocationStage` and `SetFixedAllocationPerformanceStage` validate against cost_lut
**What goes wrong:** `SetFixedAllocationStage.set_fixed_allocation()` calls `cost_lut.get_equal_node(node)` and `cost_lut.get_equal_core(...)` as sanity checks. `SetFixedAllocationPerformanceStage.set_fixed_allocation_performance()` calls `cost_lut.get_cost(equal_node, core)` to get `ideal_cycle` for scaling.
**How to avoid:** `SetFixedAllocationStage` validation can be replaced by asserting the node exists in the workload mapping directly. `SetFixedAllocationPerformanceStage` must call `latency_estimator.estimate()` with the node's fixed tiling.
**Warning signs:** These stages are only in the GA pipeline path (which is not used for CO); however they must still compile cleanly after the removal.

### Pitfall 6: `ZigZag` estimator path creates `CoreCostEntry` objects
**What goes wrong:** `ZigZagCostEstimator.estimate()` returns a `CoreCostEntry`. If `CoreCostEntry` is removed, ZigZag consumers break. But `CoreCostEntry` carries a `cme: CostModelEvaluation` field used by visualization.
**How to avoid:** Keep `CoreCostEntry` as a data type even after removing `CoreCostLUT`. Only remove the LUT (the store), not the entry class. Alternatively, keep `CoreCostEntry` in `core_cost.py` but remove `CoreCostLUT` from `core_cost_lut.py`. `ZigZagCostEstimator` is in the GA path (currently `raise NotImplementedError`) — leave its return type unchanged.
**Warning signs:** Import errors in `zigzag_cost_estimator.py` if `CoreCostEntry` import is removed.

### Pitfall 7: Visualization consumers use `tta.cost_lut` directly
**What goes wrong:** `steady_state_trace.py` (lines 290–296, 422–428) accesses `tta.cost_lut.get_equal_node(node)` and `.get_cores(eq_node)` and `.get_cost(eq_node, c).latency_total` to get node duration for display.
**How to avoid:** After removal, these should fall back to the solved `slot_lat[slot]` value which is already in the same block's else branch: `else: slot_lat[slot]`. Remove the `if eq_node is not None` branch and always use `slot_lat[slot]`. This is simpler and correct post-solve.

---

## Code Examples

### Existing AIE formula (source: `stream/stages/estimation/aie_cost_estimator.py`)
```python
# Exact formula from AIECostEstimator.estimate()
macs = prod(dim_sizes) // total_inter_core_tiling
utilization = kernel.utilization        # float, e.g. 75.0
ideal_ops_per_cycle = self.ops_per_cycle(node, core)  # e.g. 32
ideal_cycles = ceil(macs / ideal_ops_per_cycle)
ops_per_cycle = floor(ideal_ops_per_cycle * (utilization / 100.0))
cycles = ceil(macs / ops_per_cycle)     # latency_total
```

### Direct linear expression for slot latency (no auxiliary, per D-02)
```python
# Source: D-02 in CONTEXT.md; pattern mirrors Phase 5 scalar quicksum
expr = quicksum(lat_int * jw for lat_int, jw in lat_coeffs)
self.model.addConstr(self.slot_latency[s] >= expr, name=f"ssc_lat_{n.name}")
```

### Existing joint candidate pattern for transfer (source: `_joint_candidates_for_tensor` in TTA)
The same `per_dim_options` / `iproduct` / `_joint_binary_for_combo` pattern is used; for compute nodes the "tensor size" is replaced by the "latency value" as the per-candidate scalar coefficient.

### CoreCostLUT consumers — full file list (source: `grep cost_lut stream/**`)
15 files confirmed by grep:
1. `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py`
2. `stream/visualization/steady_state_trace.py`
3. `stream/stages/allocation/constraint_optimization_allocation.py`
4. `stream/cost_model/steady_state_scheduler.py`
5. `stream/stages/allocation/genetic_algorithm_allocation.py`
6. `stream/stages/estimation/core_cost_estimation.py`
7. `stream/opt/allocation/constraint_optimization/utils.py`
8. `stream/opt/allocation/genetic_algorithm/fitness_evaluator.py`
9. `stream/opt/allocation/constraint_optimization/allocation.py`
10. `stream/visualization/perfetto.py`
11. `stream/visualization/utils.py`
12. `stream/stages/set_fixed_allocation_performance.py`
13. `stream/stages/set_fixed_allocation.py`
14. `stream/visualization/constraint_optimization.py`
15. `stream/visualization/cost_model_evaluation_lut.py`

Note: CONTEXT.md says 18 consumers; grep finds 15. Three files listed in CONTEXT.md (`stream/api.py`, plus possibly 2 others) show no `cost_lut` matches in the current codebase — they may have already been cleaned up or were in the GA path. The actual migration target is 15 files.

---

## Migration Grouping Strategy

Given the scope, split across two or three plans is recommended:

| Plan | Files | Scope |
|------|-------|-------|
| Plan 1 | `core_cost_lut.py` (delete), new `tile_aware_latency.py` (create), `aie_cost_estimator.py` (extract ops_per_cycle), `core_cost_estimation.py` (delete) | New interface + delete core files |
| Plan 2 | `transfer_and_tensor_allocation.py`, `steady_state_scheduler.py`, `constraint_optimization_allocation.py` | CO allocator + scheduler (primary path) |
| Plan 3 | All visualization + GA + `set_fixed_allocation*` + `utils.py` + `allocation.py` | Non-CO consumers |

Plans 1+2 must be ordered before Plan 3. Plans 2 and 3 can share the same regression test run.

---

## Open Questions

1. **Should `CoreCostEntry` be retained or replaced?**
   - What we know: `ZigZagCostEstimator` returns `CoreCostEntry`; visualization accesses `.cme` on it.
   - What's unclear: Whether ZigZag estimator is exercised in any active test path.
   - Recommendation (Claude's discretion): Keep `CoreCostEntry` in `core_cost.py` — only delete `CoreCostLUT` from `core_cost_lut.py`. Avoids breaking ZigZag path silently.

2. **`slot_latency_ub` cache vs. recompute**
   - What we know: `_create_idle_latency_vars` (line 1395) uses `slot_latency_ub` as a big-M bound. Must be `>= max(slot_latency[s])`.
   - What's unclear: Whether the planner should cache lat_coeffs in a dict on self, or just re-enumerate inline.
   - Recommendation: Cache `self._ssc_node_lat_coeffs` during `_slot_latency_constraints` for clarity and to avoid double Gurobi variable creation.

3. **Multi-core dispatch for ops_per_cycle**
   - What we know: `AIECostEstimator.ops_per_cycle()` requires `node` and `core` to dispatch on datatypes + core type.
   - What's unclear: In variable tile mode, we iterate candidates for tiling factors but the core is fixed by `mapping.get(n).resource_allocation[0]`. Confirm there is always exactly one core entry and it is the AIE compute core.
   - Recommendation: Assert `len(mapping.get(n).resource_allocation[0]) == 1` and that the core is an AIE compute core before calling estimator.

---

## Environment Availability

Step 2.6: SKIPPED (no external dependencies — this is a pure code change; all required runtime libraries are already available from prior phases).

---

## Validation Architecture

> `workflow.nyquist_validation` key absent from `.planning/config.json` — treating as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml or pytest.ini (project root) |
| Quick run command | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x` |
| Full suite command | `.venv/bin/pytest tests/unit/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CO-06 | `_slot_latency_constraints` uses per-candidate linear expression, not scalar cost_lut | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x -k "slot_latency"` | ❌ Wave 0 — new test needed |
| CO-06 | Degenerate single-candidate input produces same result as fixed scalar | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x -k "degenerate"` | ❌ Wave 0 — new test needed |
| CO-06 | Model remains pure MILP (no nonlinear terms) after change | unit | verify model.NumNZs == 0 in quadratic check | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x`
- **Per wave merge:** `.venv/bin/pytest tests/unit/ -x`
- **Phase gate:** Full unit suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_co_tile_variables.py` — add `test_slot_latency_variable_mode`, `test_slot_latency_degenerate_single_candidate`, `test_slot_latency_scalar_fallback`
- [ ] `tests/unit/test_tile_aware_latency.py` — unit tests for `TileAwareLatencyEstimator.estimate()` covering bf16/aie2 path and formula correctness

---

## Sources

### Primary (HIGH confidence)
- `stream/stages/estimation/aie_cost_estimator.py` — AIE latency formula: `ceil(macs / floor(ideal_ops * utilization/100))`
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `_slot_latency_constraints` (lines 1247–1252), UB (lines 1393–1399), `__init__` signature (line 71), `_joint_candidates_for_tensor` (line 1738), `_joint_binary_for_combo` (line 1806), `_active_transfer_latency` variable mode (lines 1854–1929)
- `stream/cost_model/core_cost_lut.py` — full `CoreCostLUT` class; confirmed 15 consumers via grep
- `stream/cost_model/core_cost.py` — `CoreCostEntry` dataclass
- `stream/stages/estimation/core_cost_estimation.py` — `CoreCostEstimationStage.run()` and `update_cost_lut()`
- `stream/cost_model/steady_state_scheduler.py` — `update_cost_lut()` side effect and `cost_lut` constructor param
- `stream/compiler/kernels/aie_kernel.py` — `AIEKernel.utilization: float`
- `stream/mapping/mapping.py` — `NodeMapping.kernel: AIEKernel | None`

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions D-01 through D-08 — authoritative for design choices

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are already in use; no new packages
- Architecture: HIGH — formula verified directly in source code; linearization pattern verified in Phase 5
- Pitfalls: HIGH — all verified by reading actual source code of affected files

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable codebase; no fast-moving external dependencies)
