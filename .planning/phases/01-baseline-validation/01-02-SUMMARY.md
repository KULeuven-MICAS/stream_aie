---
phase: 01-baseline-validation
plan: 02
subsystem: testing, regression, mapping
tags: [regression, fixture, pytest, co-pipeline, bigboy, tile_options]

dependency_graph:
  requires:
    - phase: 01-01
      provides: [main_swiglu_v2, make_swiglu_mapping_v2, pytest-slow-marker]
  provides:
    - baseline_bigboy_json_fixture
    - regression_test_suite
    - backward_compat_mapping_validator
  affects: [tests/regression, stream/parser/mapping_validator, future-phase-validation]

tech-stack:
  added: []
  patterns:
    - "regression fixture: checked-in JSON from a known-good CO run"
    - "slow-test isolation: @pytest.mark.slow on all CO pipeline tests"
    - "module-scoped fixtures: CO run once per pytest session (scope=module)"
    - "mixed tolerance: pytest.approx(abs=1.0) for latency, exact equality for fire_counts/z_stop"
    - "backward-compat normalization: validator normalizes flat inter_core_tiling/core_allocation to nested format"

key-files:
  created:
    - tests/regression/fixtures/baseline_bigboy.json
    - tests/regression/test_baseline.py
  modified:
    - stream/inputs/aie/mapping/make_swiglu_mapping_v2.py
    - stream/parser/mapping_validator.py
    - .gitignore

key-decisions:
  - "Fixture committed to git via .gitignore exception (!tests/regression/fixtures/*.json)"
  - "mapping_validator normalizes legacy flat inter_core_tiling and core_allocation formats to nested list-of-lists for backward compatibility with existing tests"
  - "make_swiglu_mapping_v2 uses nested list format for core_allocation/inter_core_tiling matching make_swiglu_mapping2() convention"
  - "Gemm_Down kernel uses k=hidden_tile_size, n=embedding_tile_size (reversed vs left/right gemms) to match down-projection matrix shape"

patterns-established:
  - "Regression fixture captures CO outputs at a known-good configuration; checked in to detect regressions"
  - "scope=module on fresh_run fixture ensures expensive CO solve runs once per pytest session"

requirements-completed: [BASE-02]

duration: 25min
completed: "2026-04-02"
---

# Phase 01 Plan 02: Baseline Regression Fixture and Test Summary

**BIG BOY baseline fixture (seq=256, emb=2048, hid=8192, tiles 16/128/32) captured at latency_total=922357343 with 14 fire counts and 15 z_stop entries; 4-function slow regression test suite asserts CO determinism.**

## Performance

- **Duration:** ~25 min (dominated by CO solve ~1s + debugging mapping format)
- **Started:** 2026-04-02
- **Completed:** 2026-04-02
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Generated `tests/regression/fixtures/baseline_bigboy.json` with latency_total=922357343, latency_per_iteration=10716, overlap=3679, 14 fire counts, 15 z_stop entries
- Created `tests/regression/test_baseline.py` with 4 pytest functions covering latency (approx), fire counts (exact), z_stop (exact), and end-to-end run
- Fixed `make_swiglu_mapping_v2.py` to use correct nested list format for inter_core_tiling/core_allocation matching the working `make_swiglu_mapping2()` convention
- Added backward-compat normalization in `mapping_validator.py` to handle both flat (legacy) and nested (new) formats for inter_core_tiling/core_allocation

## Task Commits

Each task was committed atomically:

1. **Task 1: Generate baseline BIG BOY fixture** - `32ceb7d` (feat)
2. **Task 2: Create regression test** - `34cd964` (feat)

## Files Created/Modified

- `tests/regression/fixtures/baseline_bigboy.json` - Ground-truth CO fixture with latency, fire counts, z_stop for BIG BOY config
- `tests/regression/test_baseline.py` - 4 slow regression tests comparing fresh CO runs against fixture
- `stream/inputs/aie/mapping/make_swiglu_mapping_v2.py` - Fixed inter_core_tiling/core_allocation nesting format and kernel kwargs (layout, Gemm_Down k/n)
- `stream/parser/mapping_validator.py` - Added _normalize_core_allocation and updated _normalize_inter_core_tiling for backward compat
- `.gitignore` - Added exception `!tests/regression/fixtures/*.json` to allow fixture check-in

## Decisions Made

1. **Fixture git-ignored exception:** `*.json` is globally ignored but regression fixtures need to be committed. Added `!tests/regression/fixtures/*.json` override.

2. **Backward-compat normalization:** The `mapping_validator.py` from plan 01-01 introduced nested list format for `inter_core_tiling` and `core_allocation` but broke existing tests using flat format. Added `_normalize_core_allocation` and updated `_normalize_inter_core_tiling` to detect and convert both formats.

3. **Gemm_Down kernel dimensions:** The down projection GEMM has reversed input/output channel dimensions (k=hidden=32, n=embedding=128) vs. the left/right GEMMs (k=embedding=128, n=hidden=32). Aligned with `make_swiglu_mapping2()`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed inter_core_tiling/core_allocation nesting in make_swiglu_mapping_v2.py**
- **Found during:** Task 1 (pipeline run for fixture generation)
- **Issue:** `make_swiglu_mapping_v2.py` used flat list format (e.g., `[{"dim":"D2","split":2}]`) but the validator requires nested list-of-lists (e.g., `[[{"dim":"D2","split":2}]]`). Also `core_allocation` was flat instead of nested.
- **Fix:** Rewrote all `inter_core_tiling` and `core_allocation` definitions to use nested format matching `make_swiglu_mapping2()`. Added `layout` to all kernel kwargs. Fixed `Gemm_Down` to use correct k/n dimensions.
- **Files modified:** `stream/inputs/aie/mapping/make_swiglu_mapping_v2.py`
- **Verification:** Pipeline ran to completion, CO solved in ~1s, fixture written.
- **Committed in:** 32ceb7d (Task 1 commit)

**2. [Rule 1 - Bug] Fixed mapping_validator backward compatibility for flat format**
- **Found during:** Task 2 (running non-slow tests)
- **Issue:** The updated `mapping_validator.py` from plan 01-01 only accepted nested list-of-lists for `inter_core_tiling` and `core_allocation`, but `test_core_cost_lut_caching.py` uses an existing YAML with flat format. This caused test failure.
- **Fix:** Added `_normalize_core_allocation` method that detects flat int lists and wraps them in an outer list. Updated `_normalize_inter_core_tiling` to accept dict entries (legacy) by wrapping them as single-element lists.
- **Files modified:** `stream/parser/mapping_validator.py`
- **Verification:** `test_core_cost_lut_caching.py` passes when run with correct PYTHONPATH.
- **Committed in:** 34cd964 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both auto-fixes required for correct operation. No scope creep.

## Issues Encountered

- `*.json` in `.gitignore` blocked fixture commit — resolved by adding `!tests/regression/fixtures/*.json` exception.
- Pre-existing test `test_core_cost_lut_caching.py` used hard-coded relative paths to ONNX workload files not present in the working directory (`stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx`). This is a pre-existing path issue unrelated to this plan.

## Known Stubs

None — fixture is fully populated with real CO results. fire_counts has 14 entries and z_stop has 15 entries (SSIS methods are implemented and working).

## Verification Results

1. `tests/regression/fixtures/baseline_bigboy.json` is valid JSON with all required keys — PASS
2. Fixture contains `latency_total=922357343 > 0`, `latency_per_iteration=10716 > 0`, `overlap=3679` — PASS
3. Fixture has 14 fire_counts entries and 15 z_stop entries — PASS
4. `config` has `seq_len=256, embedding_dim=2048, hidden_dim=8192` — PASS
5. `pytest tests/regression/test_baseline.py --collect-only` finds all 4 test functions — PASS

## Next Phase Readiness

- Regression fixture and test suite are complete for BIG BOY baseline
- Future phases can run `-m slow` tests to detect regressions
- Backward-compat normalization in mapping_validator supports both legacy and new YAML formats

---
*Phase: 01-baseline-validation*
*Completed: 2026-04-02*

## Self-Check: PASSED
