# Requirements: Stream AIE — Latency Parity & Iterations Fix

**Defined:** 2026-04-08
**Core Value:** Ensure the variable tile CO produces identical results to the original fixed-tile pipeline for the same tile sizes

## v2.1 Requirements

### Latency Computation Parity

- [ ] **LAT-01**: TileAwareLatencyEstimator.estimate() produces the same MACs and cycle counts as the old AIECostEstimator + CoreCostLUT for identical tile sizes and workload dimensions
- [ ] **LAT-02**: _slot_latency_constraints correctly accounts for temporal splits (fusion_splits) without double-counting when the CO runs on the untiled workload
- [ ] **LAT-03**: The single-candidate degenerate case reproduces the original Phase 1 baseline exactly: latency_total=922357343, latency_per_iteration=10716

### Iterations Correctness

- [ ] **ITER-01**: The iterations parameter equals prod(T) over all temporal dimensions, where T = workload_size / (K * S) for each dimension
- [ ] **ITER-02**: With variable tiles, iterations is consistent with the selected tile sizes post-solve (not computed from a fixed pre-solve tiling)

### Validation

- [ ] **VAL-01**: COAnalysis.compare_latency_with_estimator() shows zero mismatches for both single-candidate and multi-candidate runs
- [ ] **VAL-02**: All existing unit tests (108) and regression tests pass after fixes

## Out of Scope

| Feature | Reason |
|---------|--------|
| New workload types | SwiGLU is the validation target for this fix |
| Performance optimization | Focus is correctness, not solver speed |
| Multi-candidate E2E improvement | Fix parity first, then optimize |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| LAT-01 | Phase 8 | Pending |
| LAT-02 | Phase 8 | Pending |
| LAT-03 | Phase 8 | Pending |
| ITER-01 | Phase 9 | Pending |
| ITER-02 | Phase 9 | Pending |
| VAL-01 | Phase 9 | Pending |
| VAL-02 | Phase 9 | Pending |

**Coverage:**
- v2.1 requirements: 7 total
- Mapped to phases: 7
- Unmapped: 0

---
*Requirements defined: 2026-04-08*
