---
phase: 06-variable-compute-latency
plan: 03
subsystem: cost-model-cleanup
tags: [cost-model, refactoring, CoreCostLUT, TileAwareLatencyEstimator, cleanup]
dependency_graph:
  requires: ["06-01", "06-02"]
  provides: ["CoreCostLUT deleted", "all consumers migrated to latency_estimator"]
  affects: ["CoreCostEstimationStage", "all CO/GA/visualization consumers"]
tech_stack:
  added: []
  patterns: ["TileAwareLatencyEstimator replaces CoreCostLUT everywhere", "latency_estimator on ctx"]
key_files:
  created: []
  modified:
    - stream/stages/estimation/core_cost_estimation.py
    - stream/cost_model/steady_state_scheduler.py
    - stream/opt/allocation/constraint_optimization/allocation.py
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - stream/opt/allocation/constraint_optimization/utils.py
    - stream/opt/allocation/genetic_algorithm/fitness_evaluator.py
    - stream/stages/allocation/constraint_optimization_allocation.py
    - stream/stages/allocation/genetic_algorithm_allocation.py
    - stream/stages/set_fixed_allocation.py
    - stream/stages/set_fixed_allocation_performance.py
    - stream/visualization/constraint_optimization.py
    - stream/visualization/perfetto.py
    - stream/visualization/steady_state_trace.py
    - stream/visualization/utils.py
    - tests/unit/test_co_tile_variables.py
  deleted:
    - stream/cost_model/core_cost_lut.py
decisions:
  - "CoreCostLUT deleted entirely per D-04; no backward compatibility layer kept"
  - "TTA scalar fallback uses latency_estimator.estimate() instead of cost_lut.get_cost()"
  - "visualization utils gracefully return np.nan when cost_lut=None (functions kept for API compat but no-op)"
  - "SetFixedAllocationStage sanity check replaced with accelerator.get_core() assertion"
  - "GA fitness evaluator uses latency_estimator.estimate() for node performance"
metrics:
  duration: 22 minutes
  completed_date: "2026-04-08"
  tasks: 2
  files_changed: 15
  files_deleted: 1
---

# Phase 6 Plan 3: CoreCostLUT Removal and Consumer Migration Summary

Delete CoreCostLUT entirely and migrate all 15 consumer files to use TileAwareLatencyEstimator or solved values; CoreCostEstimationStage now creates only a TileAwareLatencyEstimator.

## What Was Built

Completed the structural cleanup for Phase 6 by removing `CoreCostLUT` from the entire codebase:

**Task 1: Consumer file migration (13 files)**
- `stream/visualization/steady_state_trace.py`: Replaced `tta.cost_lut` lookups with direct `slot_lat[slot]` values
- `stream/visualization/perfetto.py`: Removed `cost_lut` parameter; `get_dataframe_from_scme` called without cost_lut
- `stream/visualization/utils.py`: Removed `CoreCostLUT` import; `get_spatial_utilizations`/`get_energy_breakdown` return `np.nan` gracefully (API preserved)
- `stream/visualization/constraint_optimization.py`: Removed `CoreCostLUT` param from `to_perfetto_json`
- `stream/opt/allocation/constraint_optimization/utils.py`: `get_latencies`/`get_energies`/`get_node_latencies`/`get_partitioned_nodes` all accept `latency_estimator` instead of `cost_lut`
- `stream/opt/allocation/constraint_optimization/allocation.py`: `ComputeAllocator` constructor accepts `latency_estimator`; `get_optimal_allocations` facade updated
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py`: Removed `cost_lut` parameter; scalar fallback uses `latency_estimator.estimate()`
- `stream/opt/allocation/genetic_algorithm/fitness_evaluator.py`: `FitnessEvaluator`/`StandardFitnessEvaluator` accept `latency_estimator`; `set_node_core_allocations` uses `latency_estimator.estimate()`
- `stream/stages/allocation/genetic_algorithm_allocation.py`: REQUIRED_FIELDS updated from `cost_lut` to `latency_estimator`; `valid_allocations` derived from `possible_core_allocation` instead of LUT
- `stream/stages/allocation/constraint_optimization_allocation.py`: Removed `cost_lut` pass-through to `SteadyStateScheduler`
- `stream/stages/set_fixed_allocation.py`: Removed `cost_lut`; sanity check uses `accelerator.get_core()` assertion
- `stream/stages/set_fixed_allocation_performance.py`: Uses `latency_estimator.estimate()` for `ideal_cycle`/`latency_total`
- `stream/cost_model/steady_state_scheduler.py`: Removed `cost_lut` parameter and `update_cost_lut()` method

**Task 2: CoreCostLUT deletion and CoreCostEstimationStage rewrite**
- Deleted `stream/cost_model/core_cost_lut.py` entirely
- Rewrote `CoreCostEstimationStage.run()`: creates `TileAwareLatencyEstimator(workload, mapping)` and sets it on ctx; no LUT creation, no pickle I/O, no `update_cost_lut()`
- `CoreCostEntry` in `core_cost.py` is preserved (ZigZag estimator returns it)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated unit test for scalar fallback path**
- **Found during:** Task 1 verification (test_slot_latency_scalar_fallback)
- **Issue:** Test asserted `latency_estimator.estimate` was NOT called in scalar path, but the new code uses `latency_estimator.estimate()` in the scalar fallback instead of `cost_lut.get_cost()`
- **Fix:** Updated test to assert `latency_estimator.estimate` IS called and returns latency=15
- **Files modified:** `tests/unit/test_co_tile_variables.py`
- **Commit:** 20fb85b

**2. [Rule 3 - Blocking] transfer_and_tensor_allocation.py still had cost_lut**
- **Found during:** Task 1 - file not in the 11-file list but still had CoreCostLUT import and `cost_lut` attribute
- **Issue:** TTA had backward-compat scalar fallback using `cost_lut.get_cost()` and `CoreCostLUT` import; acceptance criteria required zero imports outside core_cost_lut.py
- **Fix:** Removed `CoreCostLUT` import and `cost_lut` parameter from TTA; replaced scalar fallback with `latency_estimator.estimate()`
- **Files modified:** `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py`
- **Commit:** 20fb85b

**3. [Rule 3 - Blocking] steady_state_scheduler.py still had CoreCostLUT import**
- **Found during:** Task 1 grep verification
- **Issue:** SSS still imported CoreCostLUT and passed `cost_lut=self.cost_lut` to TTA
- **Fix:** Removed `CoreCostLUT` import and `cost_lut` parameter; removed `update_cost_lut()` method
- **Files modified:** `stream/cost_model/steady_state_scheduler.py`
- **Commit:** 20fb85b

## Commits

- `20fb85b`: feat(06-03): migrate all CoreCostLUT consumer files to TileAwareLatencyEstimator
- `0a6cff1`: feat(06-03): delete CoreCostLUT and strip LUT logic from CoreCostEstimationStage

## Verification

- `grep -r "CoreCostLUT" stream/ | grep "import"` — zero matches
- `grep -r "core_cost_lut" stream/` — zero matches
- `test ! -f stream/cost_model/core_cost_lut.py` — DELETED confirmed
- `grep "TileAwareLatencyEstimator" stream/stages/estimation/core_cost_estimation.py` — 3 matches
- `grep "cost_lut" stream/stages/estimation/core_cost_estimation.py` — zero matches
- All 93 unit tests pass

## Self-Check: PASSED

Files created/modified verified. Commits 20fb85b and 0a6cff1 confirmed in git log.
