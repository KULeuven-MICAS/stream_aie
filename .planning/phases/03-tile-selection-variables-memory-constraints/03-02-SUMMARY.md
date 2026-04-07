---
phase: 03-tile-selection-variables-memory-constraints
plan: 02
subsystem: constraint-optimization
tags: [tile-selection, milp, gurobi, binary-variables, joint-candidates, linearization]

requires:
  - phase: 03-tile-selection-variables-memory-constraints
    plan: 01
    provides: SearchSpace threaded to TransferAndTensorAllocator as self.search_space optional kwarg

provides:
  - w[dim,k] binary decision variables and tile_var[dim] INTEGER variables in TransferAndTensorAllocator
  - One-hot constraint enforcing exactly one w selected per dim
  - _joint_candidates_for_tensor returning (size_bits, joint_binary_var) pairs for multi-dim tensors
  - _joint_binary_for_combo using recursive _add_binary_product linearization
  - _tiled_dims_for_tensor to determine SearchSpace dims affecting a tensor's size

affects:
  - 03-03-PLAN.md (uses _joint_candidates_for_tensor and _tensor_max_size for big-M memory constraints)

tech-stack:
  added: []
  patterns:
    - "Double-underscore private method __create_tile_selection_vars wired as last step of _create_vars for build order consistency"
    - "Method stub pattern: bind unbound methods via __get__ to test private methods without full constructor"
    - "Recursive binary AND via _add_binary_product fold for arbitrary-dim joint binary linearization"

key-files:
  created: []
  modified:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - tests/unit/test_co_tile_variables.py

key-decisions:
  - "Test private methods via types.SimpleNamespace + manual __get__ binding avoids constructing real TransferAndTensorAllocator"
  - "_joint_candidates_for_tensor returns empty list (not None) to signal scalar fallback path to Plan 03"
  - "base_tiling is replaced per-combo rather than per-dim independently to ensure correct joint tiling"

patterns-established:
  - "Mock allocator stub: types.SimpleNamespace with method binding for testing private MILP methods"
  - "Joint candidate caching via _tensor_joint_candidates dict prevents duplicate Gurobi variable creation"

requirements-completed: [TILE-03]

duration: 6min
completed: 2026-04-07
---

# Phase 03 Plan 02: Tile Selection Variables and Joint Candidate Enumeration Summary

**w[dim,k] binary variables, tile_var[dim] INTEGER variables, one-hot constraints, and _joint_candidates_for_tensor joint enumeration added to TransferAndTensorAllocator with 14 unit tests**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-04-07T09:49:29Z
- **Completed:** 2026-04-07T09:55:22Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `__create_tile_selection_vars` creates w[dim,k] BINARY and tile_var[dim] INTEGER variables for each dim in SearchSpace, with one-hot constraint `sum_k w[dim,k] == 1` and equality constraint `tile_var[dim] = sum_k(tile[k] * w[dim,k])` per D-01/D-02
- `_tiled_dims_for_tensor` determines which SearchSpace dims affect a tensor's size by intersecting search space dims with the successor node's inter-core tiling dimensions
- `_joint_candidates_for_tensor` enumerates all joint (Cartesian product) combinations using `itertools.product`, computes per-combination tensor size, and returns `(size_bits, joint_binary_var)` pairs per D-03/D-04; sets `_tensor_max_size[tensor]` as side-effect for Plan 03 big-M use
- `_joint_binary_for_combo` recursively ANDs w[dim,k] variables via `_add_binary_product` fold, building unique constraint names with `per_dim_options[:i+1]` indexing (avoids Pitfall 4 naming collisions)
- `__create_tile_selection_vars` called as last step of `_create_vars` for backward compatibility; no-op when `search_space is None`
- 14 unit tests covering variable creation, constraints, joint binary for single/multi dim, degenerate case, caching, and scalar fallback

## Task Commits

Each task was committed atomically:

1. **Task 1: Add tile selection variables and joint candidate enumeration** - `75d7354` (feat)
2. **Task 2: Unit tests for tile selection variables and joint candidate enumeration** - `d1e74ef` (test)

## Files Created/Modified

- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` - Added imports (itertools.product, LayerDim), instance vars (w, tile_var, _tensor_max_size, _tensor_joint_candidates), methods (__create_tile_selection_vars, _tiled_dims_for_tensor, _joint_candidates_for_tensor, _joint_binary_for_combo), wired __create_tile_selection_vars into _create_vars
- `tests/unit/test_co_tile_variables.py` (new) - 14 unit tests using real gp.Model with method-binding stub pattern

## Decisions Made

- Test private methods via `types.SimpleNamespace` with manual `__get__` binding — avoids constructing full TransferAndTensorAllocator while testing the actual method logic
- `_joint_candidates_for_tensor` returns empty list (not None) to signal scalar fallback; Plan 03 checks `len(results) == 0` to decide whether to use variable or fixed tensor size in memory constraints
- base_tiling copied per-combo and dimension factors replaced in-place to build correct joint inter-core tiling for each candidate combination

## Deviations from Plan

None - plan executed exactly as written. The `_tiled_dims_for_tensor` method uses `tr.outputs.index(tensor)` for multi-successor transfers, matching the pattern from `get_tensor_of_transfer_to_single_core` in workload.py.

## Known Stubs

None — all methods are fully implemented. The `_joint_candidates_for_tensor` scalar fallback path (empty list return) is intentional design, not a stub; Plan 03 will use this signal.

## Issues Encountered

- Worktree branch was behind `arne/variable-tilesize`; resolved by fast-forward merging before implementing.
- Pre-existing test `test_core_cost_lut_caching.py::test_core_cost_lut_caches_and_loads` fails due to missing ONNX fixture (confirmed unrelated per Plan 01 summary).

## Next Phase Readiness

- `self.w`, `self.tile_var`, `_tensor_joint_candidates`, `_tensor_max_size` are populated and ready for Plan 03 memory constraints
- `_joint_candidates_for_tensor` returns `(size_bits, joint_binary_var)` pairs; Plan 03 multiplies `size_bits * joint_binary_var` in the memory capacity expression
- Empty list return signals scalar path; Plan 03 falls back to `tensor.size_bits()` when `_joint_candidates_for_tensor` returns `[]`

---
*Phase: 03-tile-selection-variables-memory-constraints*
*Completed: 2026-04-07*
