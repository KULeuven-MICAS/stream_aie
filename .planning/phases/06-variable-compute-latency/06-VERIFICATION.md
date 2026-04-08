---
phase: 06-variable-compute-latency
verified: 2026-04-08T00:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 06: Variable Compute Latency — Verification Report

**Phase Goal:** Computation node latency in the CO slot constraints becomes a tile-dependent linear expression, using the node's total kernel size (product of kernel loop dimensions) and the Kernel object's utilization to compute per-candidate cycle counts.

**Verified:** 2026-04-08
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Requirement Coverage

**CO-06** (REQUIREMENTS.md line 30): "Computation node latency in slot constraints becomes a linear expression over tile selection variables, using kernel size and Kernel utilization to compute per-candidate cycle counts"

Status: REQUIREMENTS.md marks CO-06 as **Complete** (line 75). All three plans in this phase declare `requirements: [CO-06]`. Verification below confirms it is actually satisfied in code.

---

## Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `_slot_latency_constraints` uses per-candidate `quicksum(lat[k]*jw[k])` expression instead of a scalar constant in variable tile mode | VERIFIED | TTA lines 1252-1281: branch on `search_space is not None and latency_estimator is not None`; builds `lat_coeffs` list then `quicksum(lat * jw for lat, jw in lat_coeffs)` |
| 2 | `_create_idle_latency_vars` computes `slot_latency_ub` from max over cached candidate latencies | VERIFIED | TTA lines 1437-1454: `max(lat for lat, _ in self._ssc_node_lat_coeffs[n])` used when node is in cache |
| 3 | `TileAwareLatencyEstimator.estimate()` returns correct latency using `ceil(MACs/floor(ideal_ops*util/100))` formula | VERIFIED | `tile_aware_latency.py` lines 87-96: formula matches exactly; 6 passing unit tests in `test_tile_aware_latency.py` |
| 4 | Degenerate single-candidate variable mode produces same constraint RHS as scalar mode | VERIFIED | `test_slot_latency_degenerate_single_candidate` passes (55 total passing tests) |
| 5 | Model remains pure MILP (no nonlinear terms) | VERIFIED | `grep "addGenConstrNL\|addGenConstrPow\|addGenConstrExp" transfer_and_tensor_allocation.py` returns 0 matches; `test_no_genconstr_nl_in_model` passes |
| 6 | CO primary pipeline threads `latency_estimator` end-to-end: `CoreCostEstimationStage -> ctx -> CO Allocation Stage -> SteadyStateScheduler -> TTA` | VERIFIED | All four files wire the parameter; see Key Links section below |
| 7 | `CoreCostLUT` class file is deleted from the codebase | VERIFIED | `test ! -f stream/cost_model/core_cost_lut.py` returns DELETED; zero import references in stream/ |
| 8 | `CoreCostEstimationStage` creates only `TileAwareLatencyEstimator` (no LUT) | VERIFIED | `core_cost_estimation.py` lines 65, 71: `TileAwareLatencyEstimator(workload, mapping)` + `ctx.set(..., latency_estimator=...)`; zero `cost_lut` references |
| 9 | All consumer files compile without `CoreCostLUT` import errors | VERIFIED | `grep -r "from stream.cost_model.core_cost_lut import"` returns zero matches; full 93-test suite passes |

**Score:** 9/9 truths verified

---

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `stream/cost_model/tile_aware_latency.py` | VERIFIED | Contains `TileAwareLatencyEstimator` and `LatencyEstimate`; 97 lines; substantive implementation |
| `tests/unit/test_tile_aware_latency.py` | VERIFIED | 6 test functions covering formula correctness, tiling reduction, ops_per_cycle dispatch |
| `tests/unit/test_co_tile_variables.py` | VERIFIED | 4 `test_slot_latency_*` tests; all 55 tests in file pass |
| `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` | VERIFIED | `_slot_latency_constraints` linearized; `_create_idle_latency_vars` uses cached coefficients |
| `stream/cost_model/steady_state_scheduler.py` | VERIFIED | Accepts `latency_estimator`; reconstructs after transfer graph build (Pitfall 3); passes to TTA |
| `stream/stages/estimation/core_cost_estimation.py` | VERIFIED | Creates `TileAwareLatencyEstimator`; sets on ctx; zero `cost_lut` references |
| `stream/stages/allocation/constraint_optimization_allocation.py` | VERIFIED | Reads `latency_estimator` from ctx; passes to `SteadyStateScheduler` |
| `stream/cost_model/core_cost_lut.py` | VERIFIED DELETED | File does not exist |

---

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `transfer_and_tensor_allocation.py` | `tile_aware_latency.py` | `self.latency_estimator.estimate(n, core, tuple(current_tiling))` (line 1276) | WIRED |
| `transfer_and_tensor_allocation.py` | `_ssc_node_lat_coeffs` cache | Written at lines 1281/1287/1298; read at line 1443-1444 | WIRED |
| `core_cost_estimation.py` | `tile_aware_latency.py` | `TileAwareLatencyEstimator(workload, mapping)` + `ctx.set(latency_estimator=...)` (lines 65, 71) | WIRED |
| `constraint_optimization_allocation.py` | `steady_state_scheduler.py` | `SteadyStateScheduler(..., latency_estimator=self.latency_estimator, ...)` (line 80) | WIRED |
| `steady_state_scheduler.py` | `transfer_and_tensor_allocation.py` | `TransferAndTensorAllocator(..., latency_estimator=self.latency_estimator)` (line 126) | WIRED |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `_slot_latency_constraints` | `lat_coeffs` (per-candidate latency * joint binary) | `self.latency_estimator.estimate(n, core, tuple(current_tiling))` → `ceil(macs / ops_per_cycle)` | Yes — formula uses `workload.get_dimension_size`, `kernel.utilization`, `AIECostEstimator.ops_per_cycle` | FLOWING |
| `_create_idle_latency_vars` | `slot_latency_ub` | `max(lat for lat, _ in self._ssc_node_lat_coeffs[n])` | Yes — reads from populated cache set by `_slot_latency_constraints` | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All Plan 01 unit tests pass | `pytest tests/unit/test_tile_aware_latency.py tests/unit/test_co_tile_variables.py -x` | 55 passed | PASS |
| Full unit suite passes | `pytest tests/unit/ -x -q` | 93 passed | PASS |
| No nonlinear constraints in model | `grep "addGenConstrNL" transfer_and_tensor_allocation.py` | 0 matches | PASS |
| CoreCostLUT deleted | `test ! -f stream/cost_model/core_cost_lut.py` | DELETED | PASS |
| Zero CoreCostLUT imports in stream/ | `grep -r "from stream.cost_model.core_cost_lut import" stream/` | 0 matches | PASS |
| Zero cost_lut references in core estimation stage | `grep "cost_lut" stream/stages/estimation/core_cost_estimation.py` | 0 matches | PASS |

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CO-06 | 06-01, 06-02, 06-03 | Computation node latency in slot constraints becomes a linear expression over tile selection variables, using kernel size and Kernel utilization to compute per-candidate cycle counts | SATISFIED | `_slot_latency_constraints` uses `quicksum(lat[k]*jw[k])` in variable mode; `TileAwareLatencyEstimator.estimate()` implements `ceil(MACs/floor(ideal_ops*util/100))`; all consumer files migrated; 93 unit tests pass |

No orphaned requirements — REQUIREMENTS.md maps CO-06 to Phase 6 only, and all three plans in this phase claim it.

---

## Anti-Patterns Found

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| `stream/visualization/cost_model_evaluation_lut.py` | Still contains `cost_lut` parameter in internal functions | Info | Intentional: plan 03 explicitly chose to leave this file for ZigZag compatibility; it accepts a generic dict, has no `CoreCostLUT` import, and is never called by the CO pipeline |
| `stream/visualization/utils.py` | `cost_lut=None` backward-compat params in `get_spatial_utilizations`, `get_energy_breakdown` | Info | Intentional: API preserved for visualization callers; functions return `np.nan` when `cost_lut` is None; plan 03 accepted this |
| `tile_aware_latency.py` line 32 | `energy_total: float = 0.0` (placeholder) | Info | Intentional: explicitly noted in summary as a `# TODO` in upstream `AIECostEstimator` code; does not affect constraint model correctness |

No blockers or warnings found.

---

## Human Verification Required

None — all must-haves can be verified programmatically. The constraint model structure (quicksum expression vs. scalar) is directly observable in code and confirmed by unit tests.

---

## Summary

Phase 06 fully achieves its goal. The CO slot constraint linearization is complete:

1. `TileAwareLatencyEstimator` correctly implements the `ceil(MACs/floor(ideal_ops*util/100))` formula, accepting explicit `inter_core_tiling` so the optimizer can evaluate each candidate tile size at model-construction time.
2. `_slot_latency_constraints` builds a `quicksum(lat[k]*jw[k])` linear expression in variable tile mode — the constraint is genuinely tile-dependent, not a scalar.
3. `_create_idle_latency_vars` takes the upper bound from cached per-candidate coefficients rather than a stale LUT lookup.
4. The full CO pipeline threads `latency_estimator` from `CoreCostEstimationStage` through context to `SteadyStateScheduler` to `TTA`.
5. `CoreCostLUT` is deleted with zero import references remaining.
6. All 93 unit tests pass; model contains zero nonlinear constraints.

---

_Verified: 2026-04-08_
_Verifier: Claude (gsd-verifier)_
