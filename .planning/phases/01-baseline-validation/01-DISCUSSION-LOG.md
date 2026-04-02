# Phase 1: Baseline Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 01-baseline-validation
**Areas discussed:** Main file design, Regression fixture format, Test infrastructure, Mapping source, Mapping format design, Baseline comparison tolerance, Output directory structure

---

## Main File Design

### Q1: How should main_swiglu_v2.py relate to existing main_swiglu_dse_single.py?

| Option | Description | Selected |
|--------|-------------|----------|
| Fork and simplify | Copy main_swiglu_dse_single.py, strip out codegen/MLIR save, hardcode BIG BOY params | |
| Thin wrapper | Import and call run_main_aie_codegen_swiglu() with BIG BOY args | |
| New clean entry point | Build from scratch using optimize_allocation_co() directly | ✓ |

**User's choice:** New clean entry point
**Notes:** No dependency on main_swiglu_dse_single.py at all.

### Q2: CLI args or hardcoded BIG BOY?

| Option | Description | Selected |
|--------|-------------|----------|
| CLI args with BIG BOY defaults | argparse with defaults matching BIG BOY | ✓ |
| Hardcode BIG BOY only | No CLI args, simpler file | |

**User's choice:** CLI args with BIG BOY defaults

### Q3: What should main_swiglu_v2.py print/return?

| Option | Description | Selected |
|--------|-------------|----------|
| Print key metrics | Print CO objective, total latency, fire counts to stdout | |
| Print + save JSON | Print key metrics AND dump JSON results file to outputs/ | ✓ |
| You decide | Claude picks | |

**User's choice:** Print + save JSON

---

## Regression Fixture Format

### Q4: How should the regression fixture be managed?

| Option | Description | Selected |
|--------|-------------|----------|
| Generated + checked in | Run baseline once, check JSON into repo, test compares against it | ✓ |
| Inline expected values | Hardcode values directly in test file | |
| Generate on first run | Test generates fixture if missing, asserts on subsequent runs | |

**User's choice:** Generated + checked in (Recommended)

### Q5: What should the fixture capture?

| Option | Description | Selected |
|--------|-------------|----------|
| Just the required three | CO objective, z_stop, fire counts | |
| Add latency metrics | Also total_latency, latency_per_iteration, overlap | ✓ |
| You decide | Claude picks | |

**User's choice:** Add latency metrics

---

## Test Infrastructure

### Q6: pytest or standalone script?

| Option | Description | Selected |
|--------|-------------|----------|
| pytest | New tests/ directory, enables fixtures/parameterization | ✓ |
| Standalone script | Simple Python script with assert, similar to test_co.py | |

**User's choice:** pytest (Recommended)

### Q7: Where should the regression test live?

| Option | Description | Selected |
|--------|-------------|----------|
| tests/test_baseline.py | Standard pytest layout | |
| tests/regression/test_baseline.py | Nested to separate slow integration tests | ✓ |
| You decide | Claude picks | |

**User's choice:** tests/regression/test_baseline.py

---

## Mapping Source

### Q8: How should main_swiglu_v2.py get the BIG BOY mapping?

| Option | Description | Selected |
|--------|-------------|----------|
| make_swiglu_mapping2() | Generate programmatically, self-contained | |
| Check in the YAML | Copy mapping YAML into repo | |
| Keep hardcoded path | Reference existing outputs/ path | |

**User's choice:** (Other) New function in new file that generates mapping in updated format with tile_options for CO exploration.
**Notes:** User wants a new mapping generation function that supports multiple candidate tile sizes from day one.

### Q9: Single tile or multi-candidate for Phase 1?

| Option | Description | Selected |
|--------|-------------|----------|
| New format, single tile for now | Phase 1 outputs single tile, Phase 2 adds multi-candidate | |
| Multi-candidate from the start | Function accepts list of tile sizes, Phase 1 passes single BIG BOY config | ✓ |

**User's choice:** Multi-candidate from the start
**Notes:** "That's the entire reason we're making this change: to have the tile sizes be explored in the CO."

### Q10: Where should new mapping function live?

| Option | Description | Selected |
|--------|-------------|----------|
| New file in stream/inputs/aie/mapping/ | Alongside existing make_swiglu_mapping.py | ✓ |
| You decide | Claude picks | |

**User's choice:** New file in stream/inputs/aie/mapping/

---

## Mapping Format Design

### Q11: How to represent multiple candidate tile sizes?

| Option | Description | Selected |
|--------|-------------|----------|
| tile_options list per dim | Replace 'tile: 128' with 'tile_options: [64, 128, 256]' per dim entry | ✓ |
| Global tile_size_candidates | Top-level list on fused group, keep per-dim 'tile' as baseline | |

**User's choice:** tile_options list per dim

### Q12: Backward compatibility with old tile format?

| Option | Description | Selected |
|--------|-------------|----------|
| tile_options only | Clean break. Single tile = tile_options with one element. | ✓ |
| Support both | Parser accepts either 'tile' or 'tile_options' | |

**User's choice:** tile_options only

---

## Baseline Comparison Tolerance

### Q13: Exact equality or tolerance on regression values?

| Option | Description | Selected |
|--------|-------------|----------|
| Exact equality | MILP is deterministic, results should be bit-identical | |
| Small relative tolerance | ~1e-6 relative tolerance | |
| You decide | Claude picks | |

**User's choice:** (Other) ~1e0 tolerance (a couple of cycles)
**Notes:** With tile_options restricted to single candidate, results should match but solver path may differ slightly. Future multi-candidate runs will improve, not regress.

---

## Output Directory Structure

### Q14: Where to write outputs?

| Option | Description | Selected |
|--------|-------------|----------|
| Same outputs/ pattern | outputs/{experiment_id}/ with 'v2' in experiment_id | ✓ |
| Separate outputs/v2/ | Dedicated v2 output directory | |
| You decide | Claude picks | |

**User's choice:** Same outputs/ pattern

---

## Claude's Discretion

- JSON results file structure (key names, nesting)
- pytest configuration details (conftest.py, markers)
- Experiment ID format for v2 runs

## Deferred Ideas

None — discussion stayed within phase scope
