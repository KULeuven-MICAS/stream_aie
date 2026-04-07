---
phase: 04-variable-ssis-fifo-constraints
plan: 02
subsystem: constraint-optimization
tags: [gurobi, milp, big-m, linearization, variable-tiles, ssis, fifo, buffer-descriptors]

requires:
  - phase: 04-variable-ssis-fifo-constraints
    plan: 01
    provides: "candidate-indexed coefficient dicts in reuse_levels, tiles_needed_levels, bds_needed_levels"

provides:
  - "Linearized _transfer_fire_rate_constraints with fire_lc_ big-M auxiliaries for variable tile mode"
  - "Linearized _reuse_factor_rate_constraints with rf_lc_ big-M auxiliaries for variable tile mode"
  - "Linearized _object_fifo_depth_constraints with fifo_lc_ big-M auxiliaries for variable tile mode"
  - "Linearized _buffer_descriptor_constraints with bd_lc_ big-M auxiliaries for variable tile mode"
  - "All four methods preserve scalar fallback when reuse_levels/tiles_needed_levels/bds_needed_levels are plain scalars"
  - "Degenerate single-candidate regression test confirming model feasibility"

affects:
  - 04-variable-ssis-fifo-constraints
  - future-co-integration

tech-stack:
  added: []
  patterns:
    - "isinstance(x, list) type detection at stop=-1 to branch variable vs scalar path"
    - "fire_lc_/rf_lc_/fifo_lc_/bd_lc_ lc auxiliary variable pattern for LinExpr*binary linearization"
    - "Tight big-M from _ssis_max_coefficients[(t,stop)][key] for per-constraint bound"

key-files:
  created: []
  modified:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - tests/unit/test_co_tile_variables.py

key-decisions:
  - "isinstance(rl_check, list) on stop=-1 entry determines variable vs scalar mode — avoids separate flag, consistent with Plan 01 type contract"
  - "force_double_buffering +1 offset applied to both tiles_expr_with_db and M bound in variable fifo mode — preserves tight big-M"
  - "_buffer_descriptor_constraints uses factor_dict + max_key pattern to share code between compute (tiles_needed) and memory (bds_needed) core types"

patterns-established:
  - "Pattern: Variable tile constraint linearization — check isinstance(level, list), build quicksum(coeff*jw), add lc aux with 3 big-M constraints"
  - "Pattern: Degenerate regression test — force jw=1 and z=1 for single-candidate, verify fires/reuse_factor match scalar baseline"

requirements-completed: [CO-02, CO-04]

duration: 7min
completed: 2026-04-07
---

# Phase 4 Plan 02: Linearized FIFO and BD Constraints for Variable Tile Mode Summary

**Four CO constraint methods (fire rate, reuse factor, FIFO depth, BD) linearized via big-M lc auxiliaries when reuse_levels/tiles_needed_levels/bds_needed_levels hold candidate-indexed lists; scalar path preserved for backward compatibility.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-07T11:05:37Z
- **Completed:** 2026-04-07T11:12:25Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Refactored `_transfer_fire_rate_constraints` to detect variable tile mode via `isinstance(rl, list)` and build `fire_lc_` continuous auxiliary variables with three big-M constraints per stop level
- Refactored `_reuse_factor_rate_constraints` analogously with `rf_lc_` prefix using size_factor coefficients
- Refactored `_object_fifo_depth_constraints` with `fifo_lc_` auxiliaries including force_double_buffering +1 offset in both expression and M bound
- Refactored `_buffer_descriptor_constraints` with `bd_lc_` auxiliaries, shared code for compute (tiles_needed) and memory (bds_needed) core types
- Added 16 new TDD unit tests covering scalar fallback, variable mode lc/constraint creation, double-buffering M adjustment, and degenerate single-candidate feasibility regression

## Task Commits

1. **Task 1: Linearize _transfer_fire_rate_constraints and _reuse_factor_rate_constraints** - `0d18e5f` (feat)
2. **Task 2: Linearize _object_fifo_depth_constraints and _buffer_descriptor_constraints + degenerate regression** - `1a9276b` (feat)

## Files Created/Modified

- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` - Four constraint methods refactored with variable tile linearization and scalar fallback
- `tests/unit/test_co_tile_variables.py` - 16 new TDD tests for Task 1 and Task 2 behaviors (79 total passing)

## Decisions Made

- Type detection via `isinstance(rl_check, list)` on `reuse_levels[(t, -1)]` entry — stop=-1 is always present in both modes, providing a reliable sentinel
- `force_double_buffering` offset added to both `tiles_expr_with_db` and `M` in the `fifo_lc_` path — keeps big-M tight while correctly modeling the +1 tile
- `_buffer_descriptor_constraints` uses `factor_dict`/`max_key` variables (not separate if-blocks per core type) to share linearization code between compute and memory core types

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — the worktree branch was not rebased onto `arne/variable-tilesize` initially (it was on the base commit), requiring a rebase before Phase 04-01 work was visible. This was handled automatically.

## Next Phase Readiness

- CO-02 and CO-04 requirements complete: all four SSIS-related constraint methods now support variable tile selection
- Full unit test suite (79 tests) green, including degenerate single-candidate regression
- Ready for Phase 04 integration testing or next phase combining variable tile CO with end-to-end SwiGLU solve

---
*Phase: 04-variable-ssis-fifo-constraints*
*Completed: 2026-04-07*
