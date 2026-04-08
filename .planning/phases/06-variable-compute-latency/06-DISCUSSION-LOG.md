# Phase 6: Variable Compute Latency - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 06-variable-compute-latency
**Areas discussed:** Cost LUT vs inline computation, cost_lut removal scope, post-solve interface

---

## Cost LUT vs Inline Computation

| Option | Description | Selected |
|--------|-------------|----------|
| Option A: Extend cost_lut | Pre-compute per-candidate latencies in CoreCostEstimationStage | |
| Option B: Inline during CO build | Compute per-candidate latencies directly in the allocator | Y |
| Option C: Hybrid | cost_lut for fixed, inline for variable | |

**User's choice:** Option B, and explicitly: "don't keep cost_lut for backward compat, just get rid of its existence"
**Notes:** User wants full removal of cost_lut, not just CO-side changes.

---

## cost_lut Removal Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 6 = CO only, defer rest | Only change CO allocator, keep cost_lut for other consumers | |
| Phase 6 = full removal | Remove cost_lut from all 18 consumer files | Y |

**User's choice:** Full removal across all 18 files.
**Notes:** Despite the larger scope (14 tile-dependent consumers), user prefers clean removal over incremental migration.

---

## Post-Solve Interface for Non-CO Consumers

| Option | Description | Selected |
|--------|-------------|----------|
| Compute on-demand from solved tiles | Function call with node + resolved tile sizes, no LUT | |
| Rebuild cost_lut post-solve | Recompute LUT with selected tiles so downstream works unchanged | |
| Replace with tile-aware interface | New class: node + tile sizes -> latency on demand. All consumers call this. | Y |

**User's choice:** New tile-aware interface replacing cost_lut.
**Notes:** Clean API, same interface for both fixed-tile and variable-tile modes.

---

## Claude's Discretion

- Class name and module location for tile-aware interface
- Internal organization of the 18-file migration
- ZigZag estimator handling
- Visualization caching strategy

## Deferred Ideas

None -- discussion stayed within phase scope.
