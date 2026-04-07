---
phase: 04-variable-ssis-fifo-constraints
plan: 01
subsystem: ssis-tile-aware
tags: [ssis, tile-aware, candidate-loop-sizes, fire-helpers, variable-tile, co-model]
dependency_graph:
  requires: []
  provides: [ssis-candidate-loop-sizes, ssis-loop-size-utilities, init-fire-helpers-variable-mode]
  affects: [transfer-and-tensor-allocation, tile-size-utils, steady-state-iteration-space]
tech_stack:
  added: []
  patterns: [candidate-indexed-coefficient-lists, scalar-fallback-backward-compat, itertools.product-joint-enumeration]
key_files:
  created:
    - tests/unit/test_ssis_tile_aware.py
  modified:
    - stream/workload/steady_state/iteration_space.py
    - stream/workload/utils.py
    - stream/opt/tile_size_utils.py
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - tests/unit/test_co_tile_variables.py
    - tests/unit/test_tile_size_utils.py
decisions:
  - "candidate_loop_sizes returns empty dict for dims not in applicable temporal dims (including ABSENT)"
  - "_joint_binary_for_combo accepts optional base_name param for SSIS-specific naming (ssis_jw)"
  - "reuse_levels variable mode stores (fires, sf, jw) triples; scalar mode stores (int, int) tuple"
  - "_ensure_same_ssis_for_all_transfers renamed to _verify_same_ssis_post_solve and not wired in __init__"
metrics:
  duration: "approx 30 min"
  completed: "2026-04-07"
  tasks: 2
  files: 6
requirements: [CO-02]
---

# Phase 4 Plan 1: Tile-aware SSIS + Fire Helpers Summary

Tile-aware SteadyStateIterationSpace with candidate_loop_sizes, refactored generate_ssis and _init_transfer_fire_helpers producing candidate-indexed coefficient lists for variable tile mode.

## What Was Built

### Task 1: Tile-aware SSIS, generate_ssis refactor, utility functions

**`stream/workload/steady_state/iteration_space.py`** (D-01):
- Added `search_space` and `spatial_unrollings` optional attributes to `SteadyStateIterationSpace.__init__`
- Added `candidate_loop_sizes(dim, candidates, workload_size, S=1) -> dict[int, tuple[int,int]]` method
  - Per D-02: K × S × T = workload_size; returns `{candidate_tile: (K, T)}`
  - Returns empty dict if dim not in applicable temporal dimensions (handles ABSENT dims correctly)
  - Raises AssertionError if workload_size % (S*K) != 0

**`stream/workload/utils.py`** (D-03):
- Refactored `generate_steady_state_iteration_spaces` to accept optional `search_space` parameter
- Refactored `_create_steady_state_iteration_spaces` to attach `search_space` and `spatial_unrollings` to each SSIS object

**`stream/opt/tile_size_utils.py`** (D-08/D-09):
- Added `ssis_loop_sizes_for_candidate(dim, candidate_tile, workload, mapping) -> (K, T)`
  - Pure computation convenience wrapper for K × S × T decomposition
- Added `reuse_coefficients_for_sizes(sizes, relevancies) -> (fires, sf, tn, bn) dicts`
  - Returns per-stop-level coefficient dicts keyed by stop level (−1 = no stop, 0..N-1)
  - Used by both scalar path and variable tile path

**Tests**: 9 tests in `test_ssis_tile_aware.py`, 8 new tests in `test_tile_size_utils.py`

### Task 2: Refactored _init_transfer_fire_helpers with candidate-indexed coefficients

**`stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py`**:
- Removed `_ensure_same_ssis_for_all_transfers()` call from `__init__`
- Added `_verify_same_ssis_post_solve()`: post-solve version of the check, not yet wired (Phase 6)
- Added `_ssis_tiled_dims_for_transfer(tr) -> list[LayerDim]`: intersection of search_space.dims() with SSIS applicable temporal dims (distinct from `_tiled_dims_for_tensor` which filters by inter-core tensor shape)
- Added `_ssis_coefficients_for_transfer(tr) -> dict | None`: enumerates joint candidate combinations using itertools.product, computes per-stop coefficients via `reuse_coefficients_for_sizes`, creates joint binary variables via `_joint_binary_for_combo`, returns None for scalar fallback
- Refactored `_init_transfer_fire_helpers`: delegates to `_ssis_coefficients_for_transfer`; populates `reuse_levels` as `list[(fires_c, sf_c, jw)]` in variable mode and `(int, int)` in scalar mode
- Updated type annotations for `reuse_levels`, `tiles_needed_levels`, `bds_needed_levels` to reflect union types
- Added `_ssis_max_coefficients` dict for big-M bounds
- Added `base_name` parameter to `_joint_binary_for_combo` for distinct naming

**Tests**: 7 new tests in `test_co_tile_variables.py`

## Test Results

- 63 unit tests pass (41 existing + 22 new)
- `tests/unit/test_ssis_tile_aware.py` created with 9 tests
- `tests/unit/test_tile_size_utils.py` extended with 8 new tests
- `tests/unit/test_co_tile_variables.py` extended with 7 new tests

## Commits

- `c065c35`: feat(04-01): extend SSIS with candidate_loop_sizes, refactor generate_ssis, add utilities
- `002e438`: feat(04-01): refactor _init_transfer_fire_helpers with candidate-indexed coefficients, move _ensure_same_ssis to post-solve

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] utils.py API drift from plan docs**
- **Found during:** Task 1 implementation
- **Issue:** Plan docs referenced `_derive_temporal_unrollings` and `split_factors` param, but the actual merged file uses `fusion_splits` and `_add_temporal_iteration_variables` directly
- **Fix:** Updated implementation and tests to match actual API
- **Files modified:** `stream/workload/utils.py`, `tests/unit/test_ssis_tile_aware.py`

**2. [Rule 1 - Bug] _joint_binary_for_combo needed base_name parameter**
- **Found during:** Task 2 implementation
- **Issue:** `_ssis_coefficients_for_transfer` calls `_joint_binary_for_combo` with a different naming prefix than the tensor-based caller; the method had no `base_name` parameter
- **Fix:** Added optional `base_name` parameter (default "jw") to `_joint_binary_for_combo`
- **Files modified:** `transfer_and_tensor_allocation.py`

**3. [Rule 1 - Bug] Test for generate_ssis search_space attribute needed mock-aware design**
- **Found during:** Task 1 testing
- **Issue:** Mocking `_create_steady_state_iteration_spaces` prevented testing that `search_space` attribute was set on SSIS objects; test had to be redesigned to verify the kwarg was passed through instead
- **Fix:** Changed test to verify call args to `_create_steady_state_iteration_spaces` include `search_space` keyword

## Known Stubs

None — the new utility functions and methods are fully implemented with real computation logic, not placeholders.

## Self-Check: PASSED

- `stream/workload/steady_state/iteration_space.py` has `candidate_loop_sizes` method: FOUND
- `stream/workload/utils.py` has `search_space` parameter: FOUND
- `stream/opt/tile_size_utils.py` has `ssis_loop_sizes_for_candidate` and `reuse_coefficients_for_sizes`: FOUND
- `transfer_and_tensor_allocation.py` has `_ssis_coefficients_for_transfer` and `_ssis_tiled_dims_for_transfer`: FOUND
- `tests/unit/test_ssis_tile_aware.py` exists with 9 tests: FOUND
- Commits c065c35 and 002e438: VERIFIED
- 63 unit tests pass: VERIFIED
