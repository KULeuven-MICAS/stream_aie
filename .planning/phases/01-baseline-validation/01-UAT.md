---
status: complete
phase: 01-baseline-validation
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md]
started: 2026-04-02T14:00:00Z
updated: 2026-04-02T14:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Run main_swiglu_v2.py End-to-End
expected: Running `python main_swiglu_v2.py` with BIG BOY config (256x2048x8192, tiles 16/128/32) completes without error, prints a CO objective value (latency_total), and produces JSON output. The objective value should be 922357343.
result: pass

### 2. Regression Tests Pass
expected: Running `pytest tests/regression/test_baseline.py -m slow` executes 4 test functions. All pass, confirming: latency_total matches fixture (approx abs=1.0), fire_counts match exactly (14 entries), z_stop assignments match exactly (15 entries), and end-to-end run completes.
result: pass

### 3. make_swiglu_mapping_v2 Produces Valid Mapping
expected: Calling `make_swiglu_mapping_v2(256, 2048, 8192, True, [16], [128], [32])` returns a mapping dict with `tile_options` keys (not `tile`) in each intra_core_tiling entry. The mapping passes MappingValidator without errors.
result: pass

### 4. Backward Compatibility — Existing Tests Pass
expected: Running existing (non-slow) tests that use the mapping validator and factory still pass. The mapping_validator accepts both legacy `tile` format and new `tile_options` format without errors.
result: pass

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
