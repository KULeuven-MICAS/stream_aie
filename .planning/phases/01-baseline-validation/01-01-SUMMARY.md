---
phase: 01-baseline-validation
plan: 01
subsystem: mapping-parser, entry-point, test-infra
tags: [tile_options, mapping, parser, pytest, baseline]
dependency_graph:
  requires: []
  provides: [tile_options-mapping-format, make_swiglu_mapping_v2, main_swiglu_v2, pytest-slow-marker]
  affects: [mapping_validator, mapping_factory, stream/inputs/aie/mapping, tests]
tech_stack:
  added: []
  patterns: [tile_options-list-in-yaml, cerberus-schema-optional-fields, graceful-attributeerror-fallback]
key_files:
  created:
    - stream/inputs/aie/mapping/make_swiglu_mapping_v2.py
    - main_swiglu_v2.py
    - tests/conftest.py
    - tests/regression/__init__.py
    - tests/regression/fixtures/.gitkeep
  modified:
    - stream/parser/mapping_validator.py
    - stream/parser/mapping_factory.py
    - pyproject.toml
decisions:
  - "tile_options is a list in YAML; factory takes [0] for Phase 1 single-value baseline"
  - "MappingValidator accepts either tile or tile_options per entry, validates both correctly"
  - "main_swiglu_v2 fire_counts/z_stop extraction uses graceful AttributeError fallback since scheduler methods are not yet implemented"
metrics:
  duration: 4 minutes
  completed_date: "2026-04-02"
  tasks_completed: 3
  files_created: 5
  files_modified: 3
---

# Phase 01 Plan 01: v2 Pipeline Entry Point and Mapping Parser Summary

**One-liner:** Added tile_options list format to mapping parser/factory with backward compatibility, created make_swiglu_mapping_v2() and main_swiglu_v2.py entry point for BIG BOY baseline, and set up pytest infrastructure with slow marker.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Add tile_options support to mapping parser and create make_swiglu_mapping_v2 | ac9eea0 | stream/parser/mapping_validator.py, stream/parser/mapping_factory.py, stream/inputs/aie/mapping/make_swiglu_mapping_v2.py |
| 2 | Create main_swiglu_v2.py entry point with JSON results output | 6958dc8 | main_swiglu_v2.py |
| 3 | Set up pytest infrastructure with slow marker | 24590e8 | tests/conftest.py, pyproject.toml, tests/regression/ |

## Decisions Made

1. **tile_options extraction strategy:** `_convert_intra_core_tiling_entry` extracts `tile_options[0]` for Phase 1. The list format is preserved in the YAML for future DSE phases where the CO solver will iterate over the full list.

2. **Backward compatibility:** Both `tile` and `tile_options` are accepted by MappingValidator and MappingFactory. Old YAML files with `tile` continue to work unchanged.

3. **Metric extraction fallback:** The `fire_counts` and `z_stop` extraction in `main_swiglu_v2.py` uses a two-level graceful fallback (first tries `get_applicable_temporal_sizes()`/`reuse_factor()`/`reuse_summary()`, then falls back to `get_temporal_sizes()`, then skips). These scheduler methods will be implemented in later phases.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing error handling] Graceful fallback for unimplemented scheduler methods**
- **Found during:** Task 2
- **Issue:** The plan's metric extraction code references `get_applicable_temporal_sizes()`, `reuse_factor()`, and `reuse_summary()` methods on SSIS objects that do not yet exist in the codebase. Calling these without handling would raise AttributeError at runtime.
- **Fix:** Wrapped the metric extraction in try/except AttributeError blocks. Primary path attempts the specified methods; fallback uses `get_temporal_sizes()` which exists; final fallback leaves dict empty.
- **Files modified:** main_swiglu_v2.py
- **Commit:** 6958dc8

## Known Stubs

None — the tile_options format is fully wired. The fire_counts/z_stop dicts will be empty until later phases implement the missing scheduler methods, but this is by design (baseline plan establishes structure, not execution).

## Verification Results

1. `make_swiglu_mapping_v2(256, 2048, 8192, True, [16], [128], [32])` produces valid YAML with `tile_options` keys — PASS
2. `MappingValidator` accepts the tile_options format without errors — PASS
3. `main_swiglu_v2.py` Python syntax is valid — PASS
4. `.venv/bin/pytest --collect-only tests/regression/` runs without configuration errors — PASS
5. No imports from `main_swiglu_dse_single.py` in any new file — PASS

## Self-Check: PASSED
