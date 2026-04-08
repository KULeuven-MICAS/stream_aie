---
phase: 06-variable-compute-latency
plan: "02"
subsystem: cost_model, stages, constraint_optimization
tags: [latency, milp, tile-aware, pipeline-migration, latency_estimator]
dependency_graph:
  requires:
    - "stream/cost_model/tile_aware_latency.py (TileAwareLatencyEstimator from Plan 01)"
    - "stream/cost_model/steady_state_scheduler.py"
    - "stream/stages/estimation/core_cost_estimation.py"
    - "stream/stages/allocation/constraint_optimization_allocation.py"
    - "stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py"
  provides:
    - "SteadyStateScheduler accepting latency_estimator with cost_lut optional"
    - "CoreCostEstimationStage creating and threading TileAwareLatencyEstimator via ctx"
    - "ConstraintOptimizationAllocationStage threading latency_estimator to SteadyStateScheduler"
    - "TransferAndTensorAllocator with optional cost_lut parameter"
  affects:
    - "CO primary pipeline: CoreCostEstimationStage -> ctx -> CO Allocation -> SSS -> TTA"
tech_stack:
  added: []
  patterns:
    - "Pitfall 3 mitigation: reconstruct TileAwareLatencyEstimator after transfer graph build"
    - "Keyword-only parameter passing for cost_lut in SSS/TTA constructors (backward compat)"
    - "ctx.get(key) pattern for optional pipeline context values with None fallback"
key_files:
  created: []
  modified:
    - "stream/cost_model/steady_state_scheduler.py"
    - "stream/stages/estimation/core_cost_estimation.py"
    - "stream/stages/allocation/constraint_optimization_allocation.py"
    - "stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py"
decisions:
  - "cost_lut made optional (None default) on both SteadyStateScheduler and TransferAndTensorAllocator — backward compat with non-CO consumers"
  - "latency_estimator reconstructed after transfer graph build per Pitfall 3: workload and mapping change after build_transfer_graph()/update_mapping()"
  - "cost_lut removed from REQUIRED_FIELDS in ConstraintOptimizationAllocationStage — stage no longer requires it when latency_estimator is present"
  - "SteadyStateScheduler uses keyword args for both cost_lut and latency_estimator in TTA constructor for clarity"
metrics:
  duration_minutes: 8
  completed_date: "2026-04-08"
  tasks_completed: 2
  files_changed: 4
---

# Phase 06 Plan 02: CO Pipeline Migration — cost_lut to latency_estimator Summary

**One-liner:** Wired TileAwareLatencyEstimator end-to-end through the CO primary pipeline (CoreCostEstimationStage -> ctx -> ConstraintOptimizationAllocationStage -> SteadyStateScheduler -> TTA), making cost_lut optional on SSS and TTA.

## What Was Built

### Task 1: Migrate SteadyStateScheduler from cost_lut to latency_estimator

Modified `stream/cost_model/steady_state_scheduler.py`:

- Added `from stream.cost_model.tile_aware_latency import TileAwareLatencyEstimator` import
- Changed `cost_lut: CoreCostLUT` to `cost_lut: CoreCostLUT | None = None` in constructor
- Added `latency_estimator: TileAwareLatencyEstimator | None = None` keyword parameter
- Stored `self.latency_estimator = latency_estimator` alongside `self.cost_lut`
- In `run()`: wrapped `self.cost_lut = self.update_cost_lut()` under `if self.cost_lut is not None:` guard
- In `run()`: added reconstruction of latency_estimator after transfer graph build:
  ```python
  if self.latency_estimator is not None:
      self.latency_estimator = TileAwareLatencyEstimator(self.ssw, self.mapping)
  ```
- Added `latency_estimator=self.latency_estimator` to `TransferAndTensorAllocator(...)` call

Modified `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py`:

- Changed `cost_lut: CoreCostLUT` to `cost_lut: CoreCostLUT | None = None` (positional parameter now has default)

### Task 2: Migrate ConstraintOptimizationAllocationStage and CoreCostEstimationStage

Modified `stream/stages/estimation/core_cost_estimation.py`:

- Added `from stream.cost_model.tile_aware_latency import TileAwareLatencyEstimator` import
- In `run()`: created `latency_estimator = TileAwareLatencyEstimator(self.workload, self.mapping)` after `update_cost_lut()`
- Added `latency_estimator=latency_estimator` to `self.ctx.set(...)` call
- Kept `cost_lut=self.cost_lut` in ctx.set for non-CO consumers (GA, visualization, SetFixedAllocation)

Modified `stream/stages/allocation/constraint_optimization_allocation.py`:

- Removed `"cost_lut"` from `REQUIRED_FIELDS` tuple
- Added `self.latency_estimator = self.ctx.get("latency_estimator")` in `__init__`
- Switched `SteadyStateScheduler(...)` call from positional to keyword args for cost_lut:
  ```python
  scheduler = SteadyStateScheduler(
      ...,
      cost_lut=self.cost_lut,
      ...,
      latency_estimator=self.latency_estimator,
  )
  ```

## Verification

All verifications pass:

- `pytest tests/unit/ -x` — 93 passed
- `grep "latency_estimator" steady_state_scheduler.py allocation.py core_cost_estimation.py` — all three files wire it through
- CO path: CoreCostEstimationStage -> ctx -> ConstraintOptimizationAllocationStage -> SteadyStateScheduler -> TTA threads latency_estimator end-to-end

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Fix] Worktree was behind arne/variable-tilesize branch**
- **Found during:** Initial file discovery (tile_aware_latency.py missing from worktree)
- **Issue:** Worktree was on 92b4ec3 (origin/main), missing all Phase 6 Plan 01 work (TileAwareLatencyEstimator, slot latency linearization)
- **Fix:** `git merge arne/variable-tilesize` to fast-forward worktree to feature branch
- **Impact:** No code change, git state correction only
- **Commit:** N/A (merge commit)

## Known Stubs

None — all pipeline wiring is complete. `cost_lut` remains in ctx and is still passed to SSS/TTA as keyword arg for backward compatibility with non-CO consumers (Plan 03 will remove it from those paths).

## Self-Check: PASSED

- FOUND: stream/cost_model/steady_state_scheduler.py (modified with latency_estimator)
- FOUND: stream/stages/estimation/core_cost_estimation.py (TileAwareLatencyEstimator import and ctx.set)
- FOUND: stream/stages/allocation/constraint_optimization_allocation.py (latency_estimator threaded)
- FOUND: stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py (cost_lut optional)
- FOUND: commit 341dc60 (feat: migrate SteadyStateScheduler)
- FOUND: commit ebdf7b2 (feat: migrate CO allocation and cost estimation stages)
