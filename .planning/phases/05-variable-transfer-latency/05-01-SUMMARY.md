---
phase: 05-variable-transfer-latency
plan: 01
subsystem: constraint-optimization
tags: [gurobi, milp, transfer-latency, linearization, pure-milp]

# Dependency graph
requires:
  - phase: 04-variable-ssis-fifo-constraints
    provides: "reuse_levels variable mode with (fires, sf, jw) triples; _add_binary_product; _add_binary_scaled_continuous; isinstance(rl_check, list) detection pattern"
  - phase: 03-tile-selection-variables-memory-constraints
    provides: "_joint_candidates_for_tensor returning (size_bits, jw) pairs; joint binary variable enumeration"
provides:
  - "Pure MILP _active_transfer_latency: enumeration over (tile_candidate k, stop_level s) pairs with pre-computed amortized_latency[k,s] scalar coefficients"
  - "Eliminated addGenConstrNL from the CO model — zero NL constraints remain"
  - "Removed _add_const_over_linexpr and _add_binary_times_const_over_linexpr dead NL helper methods"
  - "Unit tests confirming pure MILP latency in variable mode, degenerate case, scalar fallback, and NL-free model"
affects: [06-end-to-end-validation, any phase using _active_transfer_latency]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Big-M linearization for stop-level gating: lc_s = z_stop[s] * sum(amort[k,s] * jw[k]) via three-constraint pattern"
    - "Pre-compute amortized_latency[k,s] = ceil(size_bits[k]/min_bw) / sf_coeff[k,s] as scalar coefficients before model construction"
    - "size_by_jw dict keyed by id(jw) for safe co-indexing of tensor and SSIS candidates across potentially mismatched dim sets"
    - "Gate final result by path choice y via _add_binary_scaled_continuous (established Phase 4 pattern)"

key-files:
  created:
    - tests/unit/test_co_tile_variables.py (4 new test functions added)
  modified:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py

key-decisions:
  - "Pure MILP latency: amortized_latency[k,s] = ceil(size_bits[k]/min_bw) / sf_coeff[k,s] — all scalars, selected by joint binary products"
  - "Big-M per stop level: M_s = max(amort[k,s] for k); M_total = sum(M_s for s) — tight bounds reduce LP relaxation slack"
  - "id(jw) as dict key for safe co-indexing of tensor candidates vs SSIS candidates without assuming matching dim sets"
  - "_add_const_over_linexpr and _add_binary_times_const_over_linexpr removed entirely (only caller was old _active_transfer_latency)"
  - "Scalar fallback still pure MILP: latency_constant / sf_c scalar used with z_stop binary — no NL constraint"

patterns-established:
  - "TDD RED/GREEN: write 4 failing tests first (no import errors, fail with AttributeError on old impl), then implement"
  - "Latency enumeration mirrors Phase 3/4 pattern: iterate stop_levels, build lc_s auxiliaries, sum to lat_sum, gate by y"

requirements-completed: [CO-03]

# Metrics
duration: 18min
completed: 2026-04-08
---

# Phase 05 Plan 01: Variable Transfer Latency Summary

**Pure MILP _active_transfer_latency via (tile_candidate, stop_level) enumeration with pre-computed amortized coefficients, eliminating the only addGenConstrNL call in the CO model**

## Performance

- **Duration:** 18 min
- **Started:** 2026-04-08T11:29:59Z
- **Completed:** 2026-04-08T11:47:59Z
- **Tasks:** 2 (TDD: 1 test commit + 1 implementation commit)
- **Files modified:** 2

## Accomplishments

- Replaced NL `_active_transfer_latency` (which used `addGenConstrNL` via `_add_binary_times_const_over_linexpr`) with pure MILP enumeration over `(tile_candidate k, stop_level s)` pairs
- Pre-computed scalar coefficients `amortized_latency[k,s] = ceil(size_bits[k]/min_bw) / sf_coeff[k,s]`; binary products select the active pair
- Removed `_add_const_over_linexpr` and `_add_binary_times_const_over_linexpr` — dead NL helper methods with no remaining callers
- The entire CO model is now a pure MILP: `grep -c "addGenConstrNL" transfer_and_tensor_allocation.py` returns 0
- All 45 tests in `test_co_tile_variables.py` pass including 4 new latency-specific tests and existing degenerate regression

## Task Commits

1. **Task 1: Add unit tests for pure MILP transfer latency (RED)** - `97f5171` (test)
2. **Task 2: Refactor _active_transfer_latency to pure MILP and remove NL helpers (GREEN)** - `607ef89` (feat)

## Files Created/Modified

- `tests/unit/test_co_tile_variables.py` — 4 new test functions: `test_active_transfer_latency_variable_mode`, `test_active_transfer_latency_degenerate`, `test_active_transfer_latency_scalar_fallback`, `test_no_genconstr_nl_in_model`
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — Refactored `_active_transfer_latency`; removed `_add_const_over_linexpr` and `_add_binary_times_const_over_linexpr`

## Decisions Made

- Used `id(jw)` as dict key for the `size_by_jw` lookup to safely co-index tensor candidates and SSIS candidates — avoids assumption that the same `jw` object appears in both enumerations with the same dim sets
- Kept Big-M tight: `M_s = max(amort[k,s] for k)` per stop level; `M_total = sum(M_s for s)` — ensures LP relaxation bound is tight without over-approximating
- Scalar fallback also converted to pure MILP: uses `amort = latency_constant / sf_c` as scalar coefficient multiplied by z_stop binary — no NL constraint needed even in scalar mode

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- CO-03 satisfied: transfer latency is now a linear expression over tile selection variables
- The CO model is a complete pure MILP — no tile-dependent quantity remains as a fixed scalar
- Ready for Phase 6 end-to-end validation: the full variable tile pipeline can now be tested with a real BIG BOY config

---
*Phase: 05-variable-transfer-latency*
*Completed: 2026-04-08*
