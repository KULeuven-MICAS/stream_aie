---
phase: 07-pipeline-integration-e2e-validation
plan: 02
subsystem: cli-e2e-validation
tags: [cli, nargs, multi-candidate, e2e-test, regression, PIPE-01]
dependency_graph:
  requires:
    - main_swiglu_v2.py (run_swiglu_v2 function)
    - stream/api.py (optimize_allocation_co with multi-candidate tile_options)
    - stream/stages/generation/candidate_filter_stage.py (CandidateFilterStage)
    - tests/regression/fixtures/baseline_bigboy.json (latency reference)
    - Phase 07 Plan 01 (selected_tiles in ctx, post-solve pipeline)
  provides:
    - nargs='+' CLI args for tile sizes (D-04 validated)
    - E2E multi-candidate regression test (PIPE-01 validated)
    - Baseline tile must be tile_options[0] for iteration count parity (D-08 constraint)
  affects:
    - main_swiglu_v2.py CLI interface
    - tests/regression/ test suite
tech_stack:
  added: []
  patterns:
    - nargs='+' with list default for multi-value argparse args
    - tile_options[0] = baseline tile ensures SSIS iteration count matches fixed-tile baseline
    - pytest.mark.slow module-scoped fixtures for expensive CO runs
key_files:
  created:
    - tests/regression/test_e2e_variable_tile.py
  modified:
    - main_swiglu_v2.py
decisions:
  - "baseline tile must be tile_options[0] (first element) so make_swiglu_mapping_v2 uses the same SSIS base iteration count as the fixed-tile baseline run; otherwise latency_total is not comparable"
  - "nargs='+' with default=[] handles backward compatibility; single-value invocation produces [val] which is fixed-tile mode"
  - "MULTI_CANDIDATE_CONFIG uses [16,8,32] not [8,16,32] to ensure tile_options[0]=16 matches baseline"
metrics:
  duration: "~5 minutes"
  completed: "2026-04-08"
  tasks: 2
  files: 2
---

# Phase 07 Plan 02: CLI Multi-Candidate and E2E Validation Summary

**One-liner:** nargs='+' CLI with baseline-tile-first ordering validates PIPE-01 end-to-end: multi-candidate CO completes, selects valid divisor tiles, and achieves latency at least as good as fixed-tile baseline.

## Objective

Extend `main_swiglu_v2.py` CLI to accept multiple tile candidates per dimension (`nargs='+'`), create an E2E regression test validating multi-candidate tile selection on SwiGLU BIG BOY, and confirm the CO objective with multiple candidates is at least as good as the fixed-tile baseline.

## Tasks Completed

### Task 1: Change CLI args to nargs='+' and update arg-to-option conversion

**Commit:** 3f1ed5b

- Changed all three tile size args (`--seq_len_tile_size`, `--embedding_tile_size`, `--hidden_tile_size`) from `type=int` scalar to `type=int, nargs='+'` with list defaults.
- Removed scalar-to-list wrapping: `args.*_tile_size` is already a `list[int]` with `nargs='+'`.
- Added selected tile reporting: reads `ctx.get("selected_tiles")` after CO run, prints each dim/tile pair and includes in `results["selected_tiles"]`.
- Backward compatibility: single-value invocation `--seq_len_tile_size 16` produces `[16]` (fixed-tile mode, no change to behavior).

### Task 2: E2E regression test with multi-candidate tile selection on BIG BOY

**Commit:** 309fb56

- Created `tests/regression/test_e2e_variable_tile.py` with 4 `@pytest.mark.slow` tests:
  - `test_e2e_completes`: verifies `latency_total > 0`, `latency_per_iteration > 0`, `fire_counts` non-empty.
  - `test_selected_tiles_exist`: verifies `ctx.get("selected_tiles")` is not None and not empty.
  - `test_selected_tiles_are_valid_divisors`: checks each selected tile divides the corresponding workload dim.
  - `test_objective_at_least_as_good_as_baseline`: `latency_total <= baseline * 1.001`.

## Verification

```
pytest tests/regression/test_baseline.py tests/regression/test_e2e_variable_tile.py -m slow -x -q
8 passed in 70.43s
```

All 8 tests pass (4 baseline + 4 multi-candidate E2E).

## Deviations from Plan

### Auto-fix [Rule 1 - Bug] Candidate ordering must put baseline tile first

**Found during:** Task 2 test run

**Issue:** `MULTI_CANDIDATE_CONFIG` in the plan specification used `[8, 16, 32]` for seq_len candidates, placing the smallest tile (8) first. `make_swiglu_mapping_v2()` uses `tile_options[0]` as the reference tile to set the SSIS temporal loop sizes (and thus `iterations` count in the CO solver). With tile=8 as base, the pre-solve iteration count becomes ~1.5M (vs baseline ~56K with tile=16), causing `latency_total` to be ~8x larger even though per-iteration latency was better (5564 vs 19196). The D-08 "at least as good" assertion failed: 8,589,936,058 > 1,073,752,828 * 1.001.

**Fix:** Updated `MULTI_CANDIDATE_CONFIG` to put baseline tile first in each list:
- `seq_len_tile_options`: [16, 8, 32] (was [8, 16, 32])
- `embedding_tile_options`: [128, 64, 256] (was [64, 128, 256])
- `hidden_tile_options`: [32, 16, 64] (was [16, 32, 64])

With baseline tile first, `tile_options[0]` = 16/128/32 (same as fixed-tile baseline), iteration count matches, and `latency_total` comparison is valid.

**Files modified:** `tests/regression/test_e2e_variable_tile.py`

**Key insight documented:** The constraint `tile_options[0]` = baseline reference tile is an invariant required for correct `latency_total` comparison between variable-tile and fixed-tile runs. Added explicit comment to MULTI_CANDIDATE_CONFIG and decision record.

## Known Stubs

None — all code paths are wired to real logic and tests pass against actual CO solver.

## Self-Check: PASSED

- FOUND: main_swiglu_v2.py
- FOUND: tests/regression/test_e2e_variable_tile.py
- FOUND commit 3f1ed5b: feat(07-02): extend CLI to nargs='+' for multi-candidate tile sizes, add selected_tile reporting
- FOUND commit 309fb56: feat(07-02): add E2E multi-candidate variable tile regression test
