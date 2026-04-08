# Roadmap: Stream AIE — Variable Tile Size Optimization (v2.0)

## Overview

This milestone extends the existing Gurobi MILP constraint optimizer to treat intra-core tile sizes as decision variables. The build sequence is strictly dependency-ordered: a regression baseline is captured first so every subsequent phase has a correctness check; a pure-Python LUT module is built and validated before any MILP changes are touched; then tile selection variables, memory constraints, SSIS loop constraints, and transfer latency are introduced in that order, each reusing the pattern established by the previous phase. Pipeline integration and end-to-end validation close the milestone.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Baseline Validation** - New main entry point runs existing fixed-tile pipeline on BIG BOY config; regression test captures ground truth
- [ ] **Phase 2: TileSizeLUT Infrastructure** - Pure-Python utility functions for tensor-size computation per candidate; candidate pre-filtering and SearchSpace infrastructure
- [ ] **Phase 3: Tile Selection Variables + Memory Constraints** - Binary w[dim,k] variables introduced; memory capacity constraints linearized over tile selection
- [x] **Phase 4: Variable SSIS + FIFO Constraints** - SSIS loop sizes and object FIFO depth constraints propagate tile selection (completed 2026-04-07)
- [x] **Phase 5: Variable Transfer Latency** - Transfer latency constraints linearized over tile selection; all tile-dependent CO quantities complete (completed 2026-04-08)
- [ ] **Phase 6: Pipeline Integration + E2E Validation** - Variable mode wired through TilingGenerationStage; CLI entry point; BIG BOY end-to-end run

## Phase Details

### Phase 1: Baseline Validation
**Goal**: The existing fixed-tile pipeline runs end-to-end on BIG BOY config and its outputs are captured as a regression fixture
**Depends on**: Nothing (first phase)
**Requirements**: BASE-01, BASE-02
**Success Criteria** (what must be TRUE):
  1. Running main_swiglu_v2.py with BIG BOY args (256x2048x8192, tiles 16/128/32) completes without error and produces a CO objective value
  2. A regression test asserts the exact CO objective value, z_stop assignments, and per-transfer fire counts against the captured fixture
  3. The regression test passes when run against the unmodified fixed-tile pipeline
**Plans**: 2 plans
Plans:
- [x] 01-01-PLAN.md — tile_options schema support, make_swiglu_mapping_v2, main_swiglu_v2.py entry point, pytest setup
- [ ] 01-02-PLAN.md — Generate baseline fixture from BIG BOY CO run, create regression test suite

### Phase 2: TileSizeLUT Infrastructure
**Goal**: Pure-Python utility functions compute tensor sizes per candidate tile, candidate lists are pre-filtered for divisibility and memory feasibility, and a SearchSpace/CandidateFilterStage infrastructure is in place for downstream CO phases
**Depends on**: Phase 1
**Requirements**: TILE-01, TILE-02, TILE-04
**Success Criteria** (what must be TRUE):
  1. User can pass a list of candidate tile sizes as input; the list is accepted and threaded through the pipeline without error
  2. Tensor-size utility functions compute per-candidate tensor sizes on demand; SSIS loop sizes, reuse levels, and transfer size utilities are added incrementally in Phases 3-5 as each CO constraint type needs them
  3. Candidates that are not divisors of their workload dimension or that exceed core memory capacity are excluded before any Gurobi variable is created
  4. Unit tests for SearchSpace, utility functions, and CandidateFilterStage pass independently with no Gurobi dependency
**Plans**: 2 plans
Plans:
- [ ] 02-01-PLAN.md — SearchSpace/TileSizeOption data model, tile_size_utils.py utility functions, unit tests
- [ ] 02-02-PLAN.md — Thread tile_options_raw through MappingParserStage, CandidateFilterStage, wire into api.py

### Phase 3: Tile Selection Variables + Memory Constraints
**Goal**: The CO model contains binary w[dim,k] tile selection variables with one-hot constraints, and memory capacity constraints use LUT-derived linear expressions with tight per-constraint big-M bounds
**Depends on**: Phase 2
**Requirements**: TILE-03, TILE-05, CO-01, CO-05
**Success Criteria** (what must be TRUE):
  1. For each unique workload dimension, exactly one w[dim,k] binary variable is selected (one-hot constraint is enforced by the solver)
  2. Memory capacity constraints use sum_k(tensor_size_lut[t,k] * w[dim,k]) in place of a scalar tensor_size constant
  3. Per-constraint big-M bounds are derived from max(tensor_size_lut[t,k]) over k, not the legacy scalar heuristic
  4. The regression test passes with a single-candidate degenerate input (variable tile mode recovers the fixed-tile baseline result)
**UI hint**: no
**Plans**: 3 plans
Plans:
- [x] 03-01-PLAN.md — Thread SearchSpace through CO pipeline, add tensor_size_bits_for_candidate and max_tensor_size_bits utilities
- [x] 03-02-PLAN.md — Add w[dim,k] binary variables, tile_var[dim] INTEGER variables, joint candidate enumeration
- [ ] 03-03-PLAN.md — Rewrite _memory_capacity_constraints with continuous auxiliaries, tight big-M, regression verification

### Phase 4: Variable SSIS + FIFO Constraints
**Goal**: SSIS loop sizes (kernel and temporal), reuse levels, fire counts, and object FIFO depth constraints all use linear expressions over tile selection variables
**Depends on**: Phase 3
**Requirements**: CO-02, CO-04
**Success Criteria** (what must be TRUE):
  1. reuse_levels and tiles_needed_levels are tile-indexed linear expressions in the CO model, not scalar constants
  2. Object FIFO depth constraints use tile-dependent buffer sizes drawn from TileSizeLUT
  3. The regression test passes with a single-candidate degenerate input after SSIS changes
**Plans**: 2 plans
Plans:
- [x] 04-01-PLAN.md — SSIS utility functions, _ssis_coefficients_for_transfer helper, refactored _init_transfer_fire_helpers
- [x] 04-02-PLAN.md — Linearize fire rate, reuse factor, FIFO depth, and buffer descriptor constraints

### Phase 5: Variable Transfer Latency
**Goal**: Transfer latency in the CO model is a linear expression over tile selection variables; no tile-dependent quantity remains as a fixed scalar in the MILP
**Depends on**: Phase 4
**Requirements**: CO-03
**Success Criteria** (what must be TRUE):
  1. _active_transfer_latency uses latency_var[tr] = quicksum(latency_lut[tr,k] * w[dim,k] for k) rather than a scalar computed from a fixed tensor size
  2. The model remains a pure MILP (no nonlinear or bilinear terms); addGenConstrNL is not used
  3. The regression test passes with a single-candidate degenerate input after latency changes
**Plans**: 1 plan
Plans:
- [x] 05-01-PLAN.md — Refactor _active_transfer_latency to pure MILP enumeration, remove NL helpers, add latency unit tests

### Phase 6: Pipeline Integration + E2E Validation
**Goal**: Variable tile mode is wired through TilingGenerationStage and a CLI entry point; the CO selects a valid tile size and produces a feasible allocation on SwiGLU BIG BOY with multiple tile candidates
**Depends on**: Phase 5
**Requirements**: PIPE-01, PIPE-02
**Success Criteria** (what must be TRUE):
  1. Running main_swiglu_dse_v2.py --tile_size_options with BIG BOY config and multiple candidates completes without error and reports the selected tile size per dimension
  2. The selected tile sizes are valid divisors of their respective workload dimensions
  3. The CO objective on the multi-candidate run is at least as good as the fixed-tile baseline captured in Phase 1
  4. TilingGenerationStage can be toggled between fixed and variable tile modes without modifying the CO directly
**UI hint**: no
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Baseline Validation | 0/2 | Planning complete | - |
| 2. TileSizeLUT Infrastructure | 0/2 | Planning complete | - |
| 3. Tile Selection Variables + Memory Constraints | 2/3 | In Progress|  |
| 4. Variable SSIS + FIFO Constraints | 2/2 | Complete   | 2026-04-07 |
| 5. Variable Transfer Latency | 1/1 | Complete   | 2026-04-08 |
| 6. Pipeline Integration + E2E Validation | 0/? | Not started | - |
