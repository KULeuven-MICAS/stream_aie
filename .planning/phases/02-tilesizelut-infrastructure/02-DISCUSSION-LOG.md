# Phase 2: TileSizeLUT Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 02-tilesizelut-infrastructure
**Areas discussed:** LUT scope & quantities, Candidate filtering strategy, LUT integration point, Dimension identity model

---

## LUT Scope & Quantities

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal LUT | Only precompute what CO directly indexes (tensor sizes, transfer latencies) | |
| Full LUT | Precompute all five categories (tensor sizes, SSIS, reuse, latencies, buffer depths) | |
| On-demand functions | No LUT — utility functions compute values on demand, stored in local helper variables | ✓ |

**User's choice:** On-demand functions — no precomputed table
**Notes:** User raised the approach of utility functions instead of a LUT. Rationale: (1) combinatorial explosion with multiple tile options x dimensions x cores makes a full LUT unwieldy, (2) functions are cleaner and more extensible, (3) if values are needed multiple times, local helper variables at the call site provide natural memoization without a global cache. Functions are only called during model construction, not during solve, so no performance concern.

---

## Candidate Filtering Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| No memory pre-filter | Let CO handle all feasibility through constraints | |
| Single-tensor check | Filter tiles where even one tensor exceeds core memory | ✓ |
| Full allocation feasibility | Check total memory footprint per candidate tile | |

**User's choice:** Option 2 — single-tensor memory check as middle ground
**Notes:** Divisibility filtering is mandatory (hard constraint). Single-tensor check catches obvious outliers without duplicating CO logic.

---

## LUT Integration Point

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone module | Pure utility module imported by CO allocator, no pipeline changes | |
| New pipeline stage | Filtering stage passes results via StageContext using standardized search space class | ✓ |
| Inside CO allocator | Functions defined within transfer_and_tensor_allocation.py | |

**User's choice:** New pipeline stage with standardized search space class
**Notes:** User wants the pipeline stage as a logical separation. Emphasized that valid options should stem from a standardized class to make it clear and extensible. Vision: modular CO optimization system where optimization dimensions can be easily tuned/turned on and off.

---

## Dimension Identity Model

| Option | Description | Selected |
|--------|-------------|----------|
| Key by unique dimension group | One entry per unique workload dimension, matching determine_fusion_splits() grouping | ✓ |
| Key by LayerDim | One entry per intra_core_tiling entry, CO must enforce linked dims select same tile | |

**User's choice:** Option 1 — key by unique dimension group
**Notes:** Natural choice since the whole point is that one tile size applies to all layers sharing a dimension. Grouping already exists in the pipeline.

---

## Claude's Discretion

- Module path and naming conventions for utility functions and search space classes
- Internal structure of search space class
- Exact pipeline stage placement
- Unit test organization

## Deferred Ideas

None — discussion stayed within phase scope
