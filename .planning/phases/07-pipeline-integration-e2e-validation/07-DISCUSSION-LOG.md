# Phase 7: Pipeline Integration + E2E Validation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 07-pipeline-integration-e2e-validation
**Areas discussed:** TilingGenerationStage removal strategy, Multi-candidate CLI design, Post-solve tile application

---

## TilingGenerationStage Removal Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Skip conditionally | Skip stage in variable mode, keep for fixed | |
| Remove entirely | Delete from pipeline, both modes use CO | |
| Replace with no-op | Stage runs but does nothing in variable mode | |
| Move to post-solve | Keep stage but run it AFTER CO with solved tiles | Y |

**User's choice:** Move TilingGenerationStage to after the CO. CO runs on untiled workload, TilingGenerationStage applies chosen tiles post-solve for inspection.
**Notes:** Same flow for both fixed and variable modes. User explicitly said "generate a copy of the workload with the correct tiles to inspect."

---

## Multi-candidate CLI Design

| Option | Description | Selected |
|--------|-------------|----------|
| Extend existing args with nargs='+' | Same arg names, accept multiple values | Y |
| New --tile_size_options flag | Single flag, same list for all dims | |
| New CLI file | Separate main_swiglu_dse_v2.py | |

**User's choice:** Extend existing args to accept lists. Single value = fixed, multiple = variable.

---

## Post-Solve Tile Extraction

| Option | Description | Selected |
|--------|-------------|----------|
| Allocator method | Add get_selected_tiles() to TTA | Y |
| Pipeline extracts from model | Read w[dim,k].X directly | |

**User's choice:** Allocator method. Clean API, allocator owns the w variables.

---

## Claude's Discretion

- How TilingGenerationStage is adapted for post-solve usage
- Post-solve tiled workload storage (ctx vs separate return)
- E2E test structure
- Optional --report flag

## Deferred Ideas

None -- discussion stayed within phase scope.
