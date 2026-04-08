# Roadmap: Stream AIE — Latency Parity & Iterations Fix (v2.1)

## Overview

v2.0 introduced variable tile sizes into the CO but the pipeline reorder (TilingGenerationStage moved to post-solve) caused a latency computation discrepancy: TileAwareLatencyEstimator computes different MACs than the old CoreCostLUT path for the same tile sizes. The root cause is temporal split double-counting in _slot_latency_constraints. Additionally, the iterations parameter needs to correctly reflect tile-dependent temporal loop counts. This milestone fixes both issues and validates parity with the original Phase 1 baseline.

## Milestones

- v2.0 Variable Tile Size Optimization — Phases 1-7 (shipped 2026-04-08)
- v2.1 Latency Parity & Iterations Fix — Phases 8-9 (in progress)

## Phases

- [ ] **Phase 8: Latency Computation Parity** - Fix TileAwareLatencyEstimator and _slot_latency_constraints to produce identical MACs/cycles as old path; restore baseline regression parity
- [ ] **Phase 9: Iterations Correctness + E2E Validation** - Fix iterations = prod(T) for tile-dependent temporal loops; validate with COAnalysis; all tests green

## Phase Details

### Phase 8: Latency Computation Parity
**Goal**: TileAwareLatencyEstimator and _slot_latency_constraints produce identical per-node latency as the old CoreCostLUT path for the same tile sizes, and the single-candidate regression test matches the original Phase 1 baseline
**Depends on**: Phase 7
**Requirements**: LAT-01, LAT-02, LAT-03
**Success Criteria** (what must be TRUE):
  1. For each compute node in the BIG BOY config, TileAwareLatencyEstimator.estimate() returns the same latency_total as the old AIECostEstimator would with the same tiling
  2. _slot_latency_constraints does not double-count temporal splits when building per-candidate tiling factors
  3. The single-candidate regression test produces latency_total=922357343 and latency_per_iteration=10716 (matching original Phase 1 baseline)
  4. The CO model remains feasible and optimal for the degenerate single-candidate case
**Plans**: TBD

### Phase 9: Iterations Correctness + E2E Validation
**Goal**: The iterations parameter correctly equals prod(T) over temporal dimensions with tile-dependent T values, and COAnalysis validates zero latency mismatches
**Depends on**: Phase 8
**Requirements**: ITER-01, ITER-02, VAL-01, VAL-02
**Success Criteria** (what must be TRUE):
  1. iterations = prod(T_d for d in temporal_dims) where T_d = workload_size_d / (K_d * S_d) and K_d is the selected tile
  2. For variable tiles, iterations is recomputed post-solve to reflect the actual selected tile sizes
  3. COAnalysis.compare_latency_with_estimator() shows zero mismatches on single-candidate AND multi-candidate BIG BOY runs
  4. All 108+ unit tests and regression tests pass
  5. Multi-candidate E2E test produces an objective at least as good as the single-candidate baseline
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 8 -> 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 8. Latency Computation Parity | 0/? | Not started | - |
| 9. Iterations Correctness + E2E Validation | 0/? | Not started | - |
