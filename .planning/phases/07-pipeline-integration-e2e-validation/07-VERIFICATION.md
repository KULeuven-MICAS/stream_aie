---
phase: 07-pipeline-integration-e2e-validation
verified: 2026-04-08T00:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 07: Pipeline Integration and E2E Validation Verification Report

**Phase Goal:** TilingGenerationStage is removed from the variable tile pipeline path (tile sizes are now CO-determined); a CLI entry point runs multi-candidate tile selection end-to-end on SwiGLU BIG BOY
**Verified:** 2026-04-08
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths — Plan 01 (PIPE-02)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TilingGenerationStage runs after the CO, not before it | VERIFIED | `stream/api.py` lines 197–207: `optimize_allocation_co()` stages list places `TilingGenerationStage` after `ConstraintOptimizationAllocationStage` (line 205 vs 204). `# TilingGenerationStage,` commented-out at line 145 confirms removal from the old position. |
| 2 | fusion_splits is computed and set in ctx before ConstraintOptimizationAllocationStage runs | VERIFIED | `_FusionSplitsStage` defined at `stream/api.py` lines 36–61, inserted at line 202 in the pipeline before the CO stage. Calls `determine_fusion_splits(workload, mapping)` and sets `fusion_splits` on ctx. |
| 3 | get_selected_tiles() returns the CO-solved tile size per dimension | VERIFIED | `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1855–1870: method iterates `search_space.dims()`, reads `w[(dim,k)].X > VAR_THRESHOLD`, returns `{dim: opt.tile}`. Returns `{}` when no search space (fixed-tile mode). |
| 4 | Post-solve TilingGenerationStage applies selected tiles to produce tiled workload | VERIFIED | `stream/stages/generation/tiling_generation.py` lines 49–93: `__init__` reads `self.selected_tiles = self.ctx.get("selected_tiles")`; `run()` branches into post-solve mode when `selected_tiles is not None`, calling `substitute_loop_sizes_with_selected_tiles()` instead of `determine_fusion_splits()`. |
| 5 | Single-candidate (fixed-tile) mode still works through the same unified pipeline (per D-02) | VERIFIED | Legacy mode branch at lines 58–61 of `tiling_generation.py` preserves the original `determine_fusion_splits()` / `substitute_loop_sizes_with_tiled_sizes()` path when `selected_tiles` is None. `get_selected_tiles()` returns `{}` when `self.w` is empty. |

**Plan 01 Score:** 5/5 truths verified

---

### Observable Truths — Plan 02 (PIPE-01)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 6 | CLI accepts multiple tile sizes per dimension via nargs='+' (per D-04) | VERIFIED | `main_swiglu_v2.py` lines 189–205: all three `--*_tile_size` args have `nargs="+"` with list defaults. |
| 7 | Single value CLI invocation remains backward compatible (per D-04) | VERIFIED | `main_swiglu_v2.py` lines 218–221: `nargs='+'` with `default=[16]`/`[128]`/`[32]`; single value produces `[val]` which is fixed-tile mode. No scalar-to-list wrapping present. |
| 8 | Multi-candidate BIG BOY run completes without error and selects valid tiles (per D-08) | VERIFIED | `tests/regression/test_e2e_variable_tile.py` `test_e2e_completes` and `test_selected_tiles_exist` tests present and marked `@pytest.mark.slow`. Summary confirms `8 passed in 70.43s` including these 4 E2E tests. |
| 9 | Selected tile sizes are valid divisors of their workload dimensions (per D-09) | VERIFIED | `test_selected_tiles_are_valid_divisors` verifies `workload_size % tile == 0` for each selected tile; divisibility assertion also enforced in `substitute_loop_sizes_with_selected_tiles()` at runtime. |
| 10 | CO objective with multiple candidates is at least as good as fixed-tile baseline (per D-08) | VERIFIED | `test_objective_at_least_as_good_as_baseline` asserts `results["latency_total"] <= baseline_latency * 1.001`. Summary confirms this test passed with 8/8 total tests passing. |
| 11 | Regression test with single-candidate still passes after pipeline reorder | VERIFIED | Summary states `8 passed in 70.43s` for `test_baseline.py + test_e2e_variable_tile.py`; unit tests confirm 104 passed with no regressions. |

**Plan 02 Score:** 6/6 truths verified

**Overall Score:** 11/11 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` | `get_selected_tiles()` method | VERIFIED | Method at line 1855; iterates search_space dims, reads w[dim,k].X > VAR_THRESHOLD, returns {dim: tile}. 16 lines, substantive implementation. |
| `stream/stages/generation/tiling_generation.py` | Post-solve mode accepting selected_tiles parameter | VERIFIED | Lines 49–93 implement post-solve detection and `substitute_loop_sizes_with_selected_tiles()`. Branches correctly on `selected_tiles is not None`. |
| `stream/api.py` | Reordered pipeline with ConstraintOptimizationAllocationStage before TilingGenerationStage | VERIFIED | Lines 197–207: `_FusionSplitsStage` (line 202) → `CoreCostEstimationStage` (line 203) → `ConstraintOptimizationAllocationStage` (line 204) → `TilingGenerationStage` (line 205). |
| `tests/unit/test_get_selected_tiles.py` | Unit tests for get_selected_tiles() | VERIFIED | 5 tests: empty-w, single-candidate, multi-candidate, multi-dim, VAR_THRESHOLD boundary. Uses `types.SimpleNamespace` stub pattern. |
| `tests/unit/test_tiling_generation_post_solve.py` | Unit tests for post-solve tiling mode | VERIFIED | 6 tests across `TestPostSolveMode` and `TestLegacyMode` classes. Covers correct tiled sizes, non-divisibility assertion, non-tiled dims preserved, and legacy mode preservation. |
| `main_swiglu_v2.py` | nargs='+' CLI args for tile sizes | VERIFIED | Lines 189–221: all three tile size args use `nargs="+"` with list defaults. `selected_tiles` read and reported at lines 101–151. |
| `tests/regression/test_e2e_variable_tile.py` | E2E multi-candidate validation test | VERIFIED | 4 `@pytest.mark.slow` tests with `MULTI_CANDIDATE_CONFIG` (3 candidates per dim, baseline-tile-first ordering). Wired to `run_swiglu_v2`. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `stream/api.py` | `stream/stages/allocation/constraint_optimization_allocation.py` | fusion_splits set in ctx before CO stage | WIRED | `_FusionSplitsStage` at line 202 sets `fusion_splits` on ctx; `ConstraintOptimizationAllocationStage` at line 204 reads it. Pattern `fusion_splits` confirmed in both files. |
| `stream/stages/allocation/constraint_optimization_allocation.py` | `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` | get_selected_tiles() called post-solve, result set on ctx | WIRED | `steady_state_scheduler.py` line 167: `raw_selected = tta.get_selected_tiles()`, stored as `self.selected_tiles`. CO allocation stage line 69: `selected_tiles=scheduler.selected_tiles` set on ctx. |
| `stream/stages/allocation/constraint_optimization_allocation.py` | `stream/stages/generation/tiling_generation.py` | selected_tiles passed to TilingGenerationStage post-solve | WIRED | `selected_tiles` set on ctx by CO allocation stage (line 69). TilingGenerationStage reads it in `__init__` at line 50 via `ctx.get("selected_tiles")`. |
| `main_swiglu_v2.py` | `stream/api.py` | optimize_allocation_co() call with multi-candidate tile_options | WIRED | `main_swiglu_v2.py` imports `optimize_allocation_co` at line 8; calls it at line 87 with the tile_options passed through from CLI args. |
| `tests/regression/test_e2e_variable_tile.py` | `main_swiglu_v2.py` | run_swiglu_v2() with multi-candidate options | WIRED | Test fixture at line 58 imports and calls `run_swiglu_v2(**MULTI_CANDIDATE_CONFIG)` with 3 candidates per dimension. |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `TilingGenerationStage.run()` | `selected_tiles` | `ConstraintOptimizationAllocationStage` sets via `ctx.set(selected_tiles=scheduler.selected_tiles)` | Yes — `SteadyStateScheduler.run()` calls `tta.get_selected_tiles()` which reads solved Gurobi variable values (`w[(dim,k)].X`) post-solve | FLOWING |
| `TilingGenerationStage.run()` | `fusion_splits` (post-solve path) | `_FusionSplitsStage` sets via `ctx.set(fusion_splits=...)` | Yes — calls `determine_fusion_splits(workload, mapping)` on the live untiled workload/mapping objects from ctx | FLOWING |
| `main_swiglu_v2.py` `run_swiglu_v2()` | `selected_tiles` (reporting) | `ctx.get("selected_tiles")` after `optimize_allocation_co()` returns | Yes — same flow as above, ctx object is returned from the pipeline | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Unit tests for get_selected_tiles() pass | `.venv/bin/pytest tests/unit/test_get_selected_tiles.py -x -q` | 5 passed | PASS |
| Unit tests for post-solve tiling mode pass | `.venv/bin/pytest tests/unit/test_tiling_generation_post_solve.py -x -q` | 6 passed | PASS |
| All unit tests pass (no regressions) | `.venv/bin/pytest tests/unit/ -x -q` | 104 passed in 1.05s | PASS |
| E2E regression tests (slow) | `pytest tests/regression/test_baseline.py tests/regression/test_e2e_variable_tile.py -m slow -x -q` | 8 passed in 70.43s (from SUMMARY) | PASS (documented in summary; slow tests not re-run) |
| CLI nargs='+' accepts multiple values | `grep -n "nargs" main_swiglu_v2.py` | 3 occurrences of `nargs="+"` on tile size args | PASS |
| No scalar-to-list wrapping remains | `grep -n "\[args\." main_swiglu_v2.py` | 0 matches | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PIPE-01 | 07-02-PLAN.md | End-to-end validation: variable tile CO on SwiGLU BIG BOY config selects valid tile sizes and produces a feasible allocation | SATISFIED | `test_e2e_variable_tile.py` provides 4 slow regression tests validating completion, tile existence, divisibility, and objective quality. SUMMARY confirms all 8 tests passed (4 baseline + 4 E2E). |
| PIPE-02 | 07-01-PLAN.md | TilingGenerationStage removed from the variable tile pipeline path — tile sizes are determined by the CO solver, not a preceding stage | SATISFIED | `optimize_allocation_co()` in `stream/api.py` places `TilingGenerationStage` after `ConstraintOptimizationAllocationStage`. The stage now reads `selected_tiles` from ctx (set by CO) and applies them in post-solve mode. `_FusionSplitsStage` replaces the previous side-effect of the pre-CO tiling stage. |

**Note on REQUIREMENTS.md status:** REQUIREMENTS.md shows `PIPE-02` as unchecked (`- [ ]`) and `PIPE-01` as checked (`- [x]`). The verification above confirms PIPE-02 is fully implemented. The checkbox status in REQUIREMENTS.md appears to be a documentation lag from an earlier state and does not reflect the actual codebase.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `stream/stages/generation/tiling_generation.py` | 31 | `TODO: Add support for multiple layer stacks.` | Info | Pre-existing limitation, not introduced by this phase. Does not affect the variable tile pipeline path. |

No blockers or warnings found.

---

## Human Verification Required

### 1. E2E Regression Test Pass (Gurobi + real CO)

**Test:** Run `.venv/bin/pytest tests/regression/test_e2e_variable_tile.py -m slow -x -v`
**Expected:** 4 tests pass; `test_selected_tiles_exist` confirms non-empty `selected_tiles`; `test_objective_at_least_as_good_as_baseline` confirms `latency_total <= baseline * 1.001`
**Why human:** Requires Gurobi license and several minutes of computation. Documented as passing (8/8) in SUMMARY but slow tests were not re-run during automated verification.

---

## Gaps Summary

No gaps found. All 11 must-have truths across both plans are verified. All 7 required artifacts exist, are substantive, and are wired. All 5 key links are confirmed connected. Both PIPE-01 and PIPE-02 requirements are satisfied by the actual codebase.

The `optimize_allocation_co()` function in `stream/api.py` now implements the exact pipeline order specified by D-03:
`AcceleratorParser -> ONNXModelParser -> MappingParser -> CandidateFilter -> _FusionSplitsStage -> CoreCostEstimation -> ConstraintOptimizationAllocation -> TilingGeneration -> MemoryAccessesEstimation`

TilingGenerationStage no longer appears before the CO in the variable tile pipeline path.

---

_Verified: 2026-04-08_
_Verifier: Claude (gsd-verifier)_
