# Phase 8: Latency Computation Parity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 08-latency-computation-parity
**Areas discussed:** Double-counting diagnosis, MACs parity strategy, Regression test approach, Debugging workflow

---

## Double-counting diagnosis

| Option | Description | Selected |
|--------|-------------|----------|
| Fix inter_core_tiling source | Make _slot_latency_constraints compute inter_core_tiling correctly for the untiled workload. Keep fix localized to CO method. | ✓ |
| Fix TileAwareLatencyEstimator | Make the estimator aware of untiled dimensions and adjust MACs calculation. | |
| You decide | Claude determines fix location based on code analysis. | |

**User's choice:** Fix inter_core_tiling source (Recommended)
**Notes:** None

### Follow-up: Fusion splits handling

| Option | Description | Selected |
|--------|-------------|----------|
| Ignore fusion_splits for now | Focus purely on inter_core_tiling = workload_size / (tile * spatial). | |
| Account for fusion_splits | Include fusion_split factors in tiling computation. | |
| You decide | Claude investigates during research/planning. | |

**User's choice:** Other (free text)
**Notes:** The fusion_splits defined in a fusion group are actually the temporal T loop sizes (workload_size / (tile * spatial)). This should be correctly inferred as such. If currently the fusion splits are defined as fixed this is also wrong and somehow that info has to be clearly represented in the mapping which you can decide how.

---

## MACs parity strategy

| Option | Description | Selected |
|--------|-------------|----------|
| First principles | MACs = prod(node_dimension_sizes) / tiling_factor. Clean and verifiable. | ✓ |
| Replicate old path exactly | Trace old AIECostEstimator code path and replicate each step. | |
| You decide | Claude determines during research. | |

**User's choice:** First principles (Recommended)
**Notes:** None

---

## Regression test approach

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcoded baseline values | Assert latency_total=922357343, latency_per_iteration=10716 for single-candidate BIG BOY. | ✓ |
| Side-by-side comparison | Instantiate both old and new estimators, compare per-node. | |
| Both approaches | Hardcoded values + unit test comparing estimators. | |

**User's choice:** Hardcoded baseline values (Recommended)
**Notes:** None

---

## Debugging workflow

| Option | Description | Selected |
|--------|-------------|----------|
| Per-node MACs logging | Add temporary per-node MACs/latency logging. Run single-candidate case, compare node-by-node. Remove after fix. | ✓ |
| Minimal single-node test first | Create minimal test with one node, verify MACs, then scale up. | |
| You decide | Claude picks most efficient approach during execution. | |

**User's choice:** Per-node MACs logging (Recommended)
**Notes:** None

## Claude's Discretion

- How to represent variable fusion_splits in the mapping
- Whether to refactor get_unique_dims_inter_core_tiling or bypass it
- Internal organization of the fix
- Whether temporary logging is committed or kept as debug-only

## Deferred Ideas

None -- discussion stayed within phase scope
