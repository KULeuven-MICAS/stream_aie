---
phase: 03-tile-selection-variables-memory-constraints
plan: 03
subsystem: constraint-optimization
tags: [milp, memory-constraints, big-m, variable-tiles, linearization]
dependency-graph:
  requires: [03-02]
  provides: [memory-constraints-variable-tiles]
  affects: [transfer_and_tensor_allocation]
tech-stack:
  added: []
  patterns: [continuous-auxiliary-linearization, tight-big-m, scalar-fallback]
key-files:
  created: []
  modified:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - tests/unit/test_co_tile_variables.py
decisions:
  - "Continuous auxiliary lc vars linearize triple product u*z_stop*tile_size_expr (D-05)"
  - "Big-M = ceil(size_factor * _tensor_max_size[t]) for tight per-constraint bounds (D-06/D-07)"
  - "Scalar fallback path preserved when joint_candidates is empty (search_space None or no tiled dims)"
metrics:
  duration: ~15 minutes
  completed: 2026-04-07
  tasks: 2
  files: 2
---

# Phase 03 Plan 03: Memory Capacity Constraints with Variable Tiles Summary

Rewrote `_memory_capacity_constraints` to use continuous auxiliary `load_contrib` variables with big-M activation for the linearized triple product u * z_stop * tile_size_expr, enabling variable tensor sizes in memory constraints while preserving the scalar fallback for backward compatibility.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Rewrite _memory_capacity_constraints for variable tile mode | d11e750 | transfer_and_tensor_allocation.py |
| 2 | Unit tests for memory constraint rewrite and regression verification | c574759 | tests/unit/test_co_tile_variables.py |

## What Was Built

### Task 1: _memory_capacity_constraints rewrite

The method was extended with a `use_variable_tiles` flag (True when `search_space` is set and non-empty). For each (tensor, transfer, core, stop) triple:

- **Variable tile path**: calls `_joint_candidates_for_tensor(t, tr)` to get joint candidates. For each stop level, computes tight big-M = `ceil(size_factor * _tensor_max_size[t])`, builds a tile-dependent linear expression `tile_expr = sum(ceil(size_factor * sz) * jw for sz, jw in joint_candidates)`, creates a CONTINUOUS auxiliary variable `lc` (lb=0, ub=M), and adds 3 big-M activation constraints forcing `lc = tile_expr` when `uz=1` and `lc = 0` when `uz=0`.

- **Scalar fallback path**: when `joint_candidates` is empty (no search space or no tiled dims for this tensor), uses original `req_size * uz` accumulation unchanged.

Key correctness properties:
- When `uz=0`: `lc <= 0` via `lc <= M*uz=0`, combined with `lb=0`, forces `lc=0`.
- When `uz=1`: `lc <= tile_expr` and `lc >= tile_expr - M*(1-1) = tile_expr`, forces `lc=tile_expr`.
- `size_factor` scaling applied inside `ceil()` per (tensor, stop) pair (avoiding Pitfall 3).

### Task 2: Unit tests

Four new tests in `tests/unit/test_co_tile_variables.py`:

1. `test_memory_constraint_uses_load_contrib`: With 2 joint candidates, verifies `lc_*` variables and `lc_ub_expr_*`, `lc_ub_m_*`, `lc_lb_*` constraints exist.
2. `test_memory_constraint_scalar_fallback`: With `search_space=None`, verifies no `lc_*` variables created, `mem_cap_*` constraint present.
3. `test_tight_bigm_not_legacy`: Verifies `lc.UB == ceil(size_factor * max_size)` (tight), not the legacy `len(nodes)+5` value.
4. `test_single_candidate_regression_compat`: Degenerate 1-candidate-per-dim case produces feasible model with `w[dim,0].X == 1.0` and `tile_var[dim].X == 16.0`.

Test stub uses `MagicMock(spec=Core)` with explicit `.id = 42` to pass `isinstance(c, Core)` and `_resource_key(c)` checks.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all memory constraint logic is fully wired with no placeholder values.

## Self-Check: PASSED

Files exist:
- stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py: FOUND
- tests/unit/test_co_tile_variables.py: FOUND

Commits exist:
- d11e750: feat(03-03): rewrite _memory_capacity_constraints for variable tile mode - FOUND
- c574759: test(03-03): add unit tests for memory constraint rewrite and regression verification - FOUND

All 41 unit tests pass.
