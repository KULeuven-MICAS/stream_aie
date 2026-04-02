# Project Research Summary

**Project:** Variable Tile Size Selection in Gurobi MILP Constraint Optimizer (stream_aie)
**Domain:** MILP-based AIE accelerator allocation with discrete tile size optimization
**Researched:** 2026-04-02
**Confidence:** HIGH

## Executive Summary

This project extends an existing Gurobi MILP constraint optimizer (`TransferAndTensorAllocator`) to make intra-core tile sizes decision variables rather than fixed constants. The existing CO already performs joint optimization of tensor placement (`x`), transfer routing (`y`), and reuse levels (`z_stop`) using well-established binary one-hot patterns. The recommended approach adds tile size selection as a fourth decision dimension: one binary indicator per (unique workload dimension, candidate tile size), linked to all tile-dependent quantities via precomputed scalar lookup tables (LUTs). This keeps the formulation fully linear — no nonlinear or bilinear solver extensions are required — and is consistent with the patterns already present in the codebase.

The central architectural decision is to perform tile selection jointly inside `TransferAndTensorAllocator` rather than in an outer enumeration loop. Memory feasibility depends on the interaction of tile size, tensor placement, and reuse level simultaneously; separating these into sequential stages produces suboptimal results and misses constraint interactions. The recommended implementation path builds a `TileSizeLUT` module that precomputes all tile-dependent constants (tensor sizes, SSIS loop sizes, reuse levels, transfer latencies) for every candidate, then represents each of these inside the MILP as a linear combination of precomputed scalars weighted by tile-selection binaries. This converts what would be nonlinear constraint coefficients into straightforward linear expressions.

The highest-risk areas are: (1) SSIS data flow, where the existing pipeline computes SSIS from a single fixed tile before the CO runs and passing it unchanged will produce silently wrong reuse factors; (2) big-M bound quality, where the existing heuristic `len(workload.nodes()) + 5` is valid for placement logic but far too loose for tile-size-dependent memory coefficients that can reach kilobits; and (3) binary product explosion, where naively treating tile selection as a third binary dimension alongside placement and reuse level creates a three-way product requiring quadratic growth in auxiliary variables. All three risks have clear mitigations that must be designed before the first constraint is written.

---

## Key Findings

### Recommended Stack

No new packages are required. The entire implementation uses `gurobipy 13.0.0` (already installed), standard Python `math` and `collections` modules, and the existing `GRB.BINARY`, `GRB.INTEGER`, `quicksum`, `addConstr`, `addLConstr`, and `addGenConstrIndicator` APIs already imported in the codebase. The one-hot tile selection pattern (`quicksum(b[d][k] for k) == 1`) is structurally identical to the existing `z_stop` one-hot encoding, making the extension consistent with established code patterns.

Key Gurobi 13.0-specific recommendations: use `addLConstr` instead of `addConstr` for all-linear constraints (approximately 50% faster constraint addition); add solver parameters `MIPFocus=1` (prioritize finding feasible solutions quickly) and `Presolve=2` (aggressive variable elimination); and use `model.setParam()` per-model-instance (global `gp.setParam()` no longer affects already-created Model objects in Gurobi 13.0). Avoid `addGenConstrPoly/Exp/Log` (removed in 13.0), loose global big-M constants, and `GRB.INTEGER` variables for discrete non-contiguous tile size sets.

**Core technologies:**
- `gurobipy 13.0.0`: MILP solver interface — already installed, all required APIs available, no version change needed
- `math.ceil` (stdlib): rounding tile-dependent size computations — needed for LUT precomputation outside the MILP
- `collections.defaultdict` (stdlib): accumulating tile-size-dependent linear expressions — used in LUT construction

### Expected Features

The milestone v2.0 MVP is fully defined. All P1 features must be present for the variable tile CO to run end-to-end on SwiGLU BIG BOY (256×2048×8192).

**Must have (table stakes — P1):**
- Candidate tile size list input — single shared list for all unique workload dimensions; divisibility pre-filtering applied before variable creation
- Binary tile-selection indicators `w[dim, k]` — one binary per (unique dimension, candidate); exactly-one constraint per dimension
- Precomputed per-candidate auxiliary tables — tensor sizes, SSIS loop sizes, reuse levels, transfer latencies all computed as Python scalars before the MILP is built
- Linearized variable tensor sizes in memory capacity constraints — replaces scalar `tensor_size` with `sum_k(lut[t, k] * w[dim, k])`; re-linearizes the `tensor_size * z_stop` bilinear product
- Linearized variable SSIS loop sizes — `tiles_needed_levels` and `reuse_levels` become linear expressions over `w`
- Baseline fixed-tile validation — correctness reference before any structural changes
- End-to-end SwiGLU BIG BOY validation — CO must select a valid tile and produce allocation at least as good as fixed-tile baseline

**Should have (P2, post-core-validation):**
- Memory-capacity-aware candidate pruning — eliminate candidates that exceed core capacity before building `w` variables
- Warm-start from fixed-tile baseline — pass fixed-tile allocation as `VarHintVal` hints to accelerate search
- Tile efficiency objective term — penalize under-utilized tiles using kernel cost LUT lookup per candidate

**Defer (v2+):**
- Per-layer independent tile size selection — requires relaxing `_ensure_same_ssis_for_all_transfers`; significant SSIS invariant refactoring
- Joint tile size and inter-core dataflow optimization — requires rebuilding the workload graph topology inside the optimization loop; fundamentally different architecture
- Continuous tile size variables — not expressible in MILP without MIQP/piecewise-linear reformulation

### Architecture Approach

The recommended architecture introduces two new modules (`TileSizeLUT` and `TileSizeSelector`) alongside targeted modifications to `TransferAndTensorAllocator` and `TilingGenerationStage`. The core design principle is strict separation of concerns: `Tensor` objects remain pure data with fixed shapes; all tile-variable quantities live exclusively in `TileSizeLUT`, which is built before the MILP is constructed and consumed only inside `TransferAndTensorAllocator`. The MILP itself uses three interlocking patterns: SOS1 one-hot tile selection, precomputed LUT linear combinations, and big-M integer-binary product linearization. The `tile_coeff_var` abstraction — an INTEGER variable constrained to equal the tile-selected scalar coefficient — is the key pattern that prevents three-way binary product explosion.

**Major components:**
1. `TileSizeLUT` (new module) — precomputes all tile-dependent scalar quantities for every candidate; pure Python, no Gurobi dependency; unit-testable independently
2. `TileSizeSelector` (new class inside CO) — encapsulates `w[dim, k]` binary variables, one-hot constraints, and `tile_size_expr` linear expressions; analogous to `NamespaceConstraints`
3. Modified `TransferAndTensorAllocator` — consumes `TileSizeLUT`; replaces scalar `req_size`, `reuse_levels`, and latency constants with LUT-indexed linear expressions
4. Modified `TilingGenerationStage` — adds variable mode that skips fixed tile substitution and passes `tile_size_options` through `StageContext`
5. `main_swiglu_dse_v2.py` (new entry point) — CLI with `--tile_size_options` flag wiring the variable mode end-to-end

### Critical Pitfalls

1. **Loose big-M bounds from tile-dependent coefficient range** — The existing `big_m = len(workload.nodes()) + 5` is valid for placement logic but completely wrong for tile-size-dependent memory coefficients. Compute per-constraint tight bounds from `max(tensor_size_lut[t, k])` over all k. Failure symptom: LP relaxation gap >50% after presolve; solve time scales super-linearly with number of tile options.

2. **SSIS data flow mismatch** — The SSIS is constructed from a single fixed tile before the CO runs. With variable tile sizes the CO selects a tile that changes SSIS loop sizes, but the SSIS passed in reflects a different (or arbitrary) tile. Fix: generate one SSIS per candidate tile upstream and index them by tile option. This must be resolved before any CO variable for tile selection is introduced.

3. **Three-way binary product explosion (`u × z_stop × s_k`)** — Adding tile selection as a third binary gating dimension produces `N × C × L × K` auxiliary variables, growing cubically. Fix: introduce `tile_coeff_var[t, stop]` (INTEGER variable constrained to the tile-selected coefficient) so the memory constraint only ever has a two-way binary-integer product. Design this abstraction before writing any constraint.

4. **Variable transfer latency becomes nonlinear without the LUT approach** — `_transfer_latency_for_path` computes `tensor.size_bits() / min_bw` as a scalar constant. With variable tile sizes the numerator changes with tile selection. The safe approach: precompute `latency_lut[tr, k]` for every (transfer, tile option) pair and represent `latency_var[tr]` as a linear combination. Do not introduce `addGenConstrNL` or the model becomes nonlinear (MINLP).

5. **Missing regression baseline before any structural changes** — MILP bugs are silent: the model remains feasible and returns an objective, just wrong. Capture the exact baseline (objective value, `z_stop` assignments, `fires` per transfer, memory allocations) on the BIG BOY fixed-tile config before the first line of code is changed. Assert exact match after every phase using the degenerate single-tile-option case.

---

## Implications for Roadmap

Based on research, the dependency structure is unambiguous and dictates the phase order. The SSIS data flow refactor is a hard prerequisite for all CO changes. The `TileSizeLUT` module is a prerequisite for all variable-coefficient constraints. The memory capacity constraint is the right first CO integration point (highest impact, most complexity). Transfer latency and SSIS loop size constraints follow. P2 enhancements (warm-start, pruning, objective term) are independent add-ons after core validation.

### Phase 0: Baseline Capture and Regression Infrastructure
**Rationale:** PITFALLS.md identifies missing regression baseline as a prerequisite for all phases. Without captured ground truth, every subsequent phase has no correctness check. Zero code changes to CO in this phase.
**Delivers:** Captured baseline outputs (objective, `z_stop`, `fires`, memory allocation) for BIG BOY fixed-tile config; regression test in `test_co.py` that asserts exact match; documented baseline numbers in `.planning/` or test fixtures.
**Addresses:** Pitfall 7 (regression baseline)
**Avoids:** Silent correctness regressions throughout all subsequent phases

### Phase 1: TileSizeLUT Precomputation Module
**Rationale:** ARCHITECTURE.md establishes the LUT pattern as the foundation for all subsequent CO changes. This module has no Gurobi dependency, can be unit-tested independently, and unblocks all downstream phases. Building and validating it first eliminates the risk that incorrect precomputed constants corrupt the MILP.
**Delivers:** `stream/opt/allocation/constraint_optimization/tile_size_lut.py` with `TileSizeLUT.build(workload, tile_size_options)` producing verified scalar tables for tensor sizes, SSIS loop sizes, reuse levels, and transfer latencies per candidate tile.
**Uses:** `workload.with_modified_dimension_sizes()`, `SteadyStateScheduler.generate_ssis()`, standard Python
**Implements:** TileSizeLUT component; SSIS-per-candidate-tile data flow (also resolves Pitfall 2 / 6 at the data layer)
**Avoids:** Pitfall 2 (SSIS mismatch), Pitfall 6 (reuse dict desync)

### Phase 2: Tile Selection Variables and Memory Capacity Constraint
**Rationale:** FEATURES.md rates variable tensor sizes in memory capacity as the highest-complexity P1 feature. ARCHITECTURE.md identifies this constraint as the primary integration point. PITFALLS.md confirms Pitfalls 1 and 3 manifest here first. Building the `tile_coeff_var` abstraction and tight big-M pattern in this phase establishes the correct pattern for all subsequent constraint groups.
**Delivers:** `TileSizeSelector` class with `w[dim, k]` binary variables, one-hot constraints, and `tile_size_expr`; modified `_memory_capacity_constraints` using `tensor_size_var` linear expressions; `tile_coeff_var` abstraction; tight per-constraint big-M bounds from LUT; regression passing at single-tile-option degenerate case.
**Uses:** `gurobipy 13.0.0` (`addLConstr`, `quicksum`, `GRB.BINARY`, `GRB.INTEGER`); `TileSizeLUT` from Phase 1
**Implements:** TileSizeSelector component; Pattern 1 (LUT + linear combination); Pattern 3 (big-M for integer-binary product)
**Avoids:** Pitfall 1 (loose big-M), Pitfall 3 (binary product explosion), Pitfall 5 (mixed-magnitude coefficients — normalize memory units in this phase)

### Phase 3: Variable SSIS Loop Sizes (Reuse and Fire Constraints)
**Rationale:** FEATURES.md rates variable SSIS loop sizes as the second highest-complexity P1 feature. This phase propagates tile selection into `_init_transfer_fire_helpers`, making `reuse_levels`, `tiles_needed_levels`, and `bds_needed_levels` tile-indexed. PITFALLS.md confirms Pitfall 6 (reuse dict desync) is tightly coupled with Pitfall 2 and must be resolved at the constraint level here.
**Delivers:** Updated `_init_transfer_fire_helpers` consuming `reuse_levels_lut[(t, stop, k)]` from `TileSizeLUT`; `fires_var` and `size_factor_var` as linear expressions over `w`; modified `z_stop`-dependent constraint groups (object FIFO depth, buffer descriptor depth); regression passing at single-tile-option case.
**Uses:** `TileSizeLUT` reuse level tables; existing `_add_binary_product` helper extended via `tile_coeff_var` pattern
**Avoids:** Pitfall 6 (z_stop / reuse dict desync), Pitfall 3 (binary explosion via `tile_coeff_var`)

### Phase 4: Variable Transfer Latency
**Rationale:** FEATURES.md includes variable transfer sizes as a P1 table-stakes feature. PITFALLS.md Pitfall 4 specifically warns that `_transfer_latency_for_path` will silently compute latency from the wrong tensor size if not updated. This phase replaces the scalar latency constant with the LUT-based linear combination, completing the linearization of all tile-dependent quantities in the CO.
**Delivers:** Modified `_active_transfer_latency` using `latency_var[tr] = quicksum(latency_lut[tr, k] * w[dim, k] for k)` as the numerator linear expression; updated `_add_binary_times_const_over_linexpr` or replacement approach; regression passing at single-tile-option case.
**Uses:** `TileSizeLUT` latency tables; piecewise latency pattern from ARCHITECTURE.md
**Avoids:** Pitfall 4 (variable transfer latency becoming nonlinear)

### Phase 5: Pipeline Integration and End-to-End Validation
**Rationale:** ARCHITECTURE.md identifies `TilingGenerationStage` modification and `main_swiglu_dse_v2.py` as the final integration step. FEATURES.md requires end-to-end SwiGLU BIG BOY validation as a P1 table-stakes feature. This phase wires the variable tile mode through `StageContext`, adds the CLI entry point, and validates the full pipeline.
**Delivers:** `TilingGenerationStage` variable mode (skips fixed tile substitution; passes `tile_size_options` through context); `main_swiglu_dse_v2.py` with `--tile_size_options` CLI flag; full end-to-end run with BIG BOY config selecting among multiple tile candidates; objective compared against fixed-tile baseline.
**Addresses:** End-to-end SwiGLU BIG BOY validation (final P1 requirement)
**Uses:** All components from Phases 1-4; Gurobi solver parameters (`MIPFocus=1`, `Presolve=2`, `NumericFocus=1`)

### Phase 6: P2 Enhancements (Post-Core-Validation)
**Rationale:** FEATURES.md explicitly defers warm-start, memory-capacity-aware pruning, and tile efficiency objective term to P2 after core is validated. These are independent of each other and of the core phases, so they can be implemented in any order or in parallel.
**Delivers:** Memory-capacity-aware candidate pre-pruning (reduces `w` variable count); `VarHintVal` warm-start from fixed-tile baseline; tile utilization objective term in CO.
**Addresses:** P2 features from FEATURES.md
**Uses:** `VarHintVal`/`VarHintPri` Gurobi 13.0 variable attributes (NoRel heuristic now respects these)

### Phase Ordering Rationale

- Phase 0 before everything: MILP bugs are silent; no correctness claim is possible without a baseline.
- Phase 1 before CO changes: `TileSizeLUT` is a pure-Python module; validating it independently eliminates a major source of corruption risk for the MILP formulation.
- Phase 2 before Phases 3 and 4: Memory capacity is the constraint that introduces the new binary variables and the `tile_coeff_var` abstraction. The pattern established here is reused verbatim in Phases 3 and 4.
- Phase 3 before Phase 4: SSIS-derived quantities (reuse levels, fire counts) feed into the latency constraint indirectly through `reuse_factor_var`. The latency formulation in Phase 4 is simpler once the reuse factor is already a well-formed linear expression.
- Phase 5 last among core phases: pipeline integration is the integration test for Phases 1-4.
- Phase 6 after Phase 5: P2 enhancements are premature until correctness is confirmed end-to-end.

### Research Flags

Phases needing deeper research during planning:
- **Phase 2 (Memory Capacity Constraint):** The exact shape of the integer-binary linearization (`tile_coeff_var × uz`) needs careful formulation review before implementation. STACK.md and ARCHITECTURE.md provide the patterns but the exact boundary between using `tile_coeff_var` as INTEGER vs. re-expanding to binary products per tile option may need a small formulation experiment.
- **Phase 3 (Variable SSIS Loop Sizes):** The number of distinct `(t, stop, k)` triples in the BIG BOY config is not quantified in research. If `tiles_needed_levels` has many entries, the `tile_coeff_var` population may still be large. Recommend quantifying variable counts before implementation.

Phases with standard patterns (skip research-phase):
- **Phase 0:** Capturing test outputs and writing assertions is standard pytest work; no research needed.
- **Phase 1:** `TileSizeLUT` is pure Python calling existing workload APIs; the API surface is already understood from codebase analysis.
- **Phase 4 (Transfer Latency):** The LUT-based linear combination pattern is identical to Phase 2; research is not needed once Phase 2 is validated.
- **Phase 5 (Pipeline Integration):** `StageContext` changes and CLI argument addition follow existing patterns in the codebase.
- **Phase 6 (P2 Enhancements):** All three P2 features are additive; `VarHintVal` usage is documented in STACK.md; pruning is pre-filtering logic only.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Gurobi 13.0 official docs verified; all API calls confirmed against reference; existing codebase cross-checked for already-imported symbols |
| Features | HIGH | Primary source is the codebase itself (1717-line CO file); feature boundaries derived from actual code structure, not speculation |
| Architecture | HIGH | Based on direct analysis of all relevant source files; LUT + linear combination pattern is a standard MILP technique with clear precedent |
| Pitfalls | HIGH | Each pitfall grounded in Gurobi official documentation and direct code line references; warning signs are observable and specific |

**Overall confidence:** HIGH

### Gaps to Address

- **Exact variable counts for BIG BOY config:** The number of unique tensors, transfer nodes, reuse stop levels, and their product with K tile options is not computed. This affects Phase 2 and 3 implementation decisions (whether SOS1 registration is needed; whether `tile_coeff_var` count is acceptable). Resolve by adding a diagnostic print of model statistics after Phase 2 baseline construction.
- **`ceil()` incompatibility with `LinExpr`:** STACK.md flags that `req_size = ceil(size_factor * tensor_size)` cannot be applied when `tensor_size` is a `LinExpr`. The research recommends absorbing rounding into the LUT constants or using a conservative upper bound, but the exact handling of fractional tile-derived sizes needs a concrete decision before Phase 2 implementation.
- **`SteadyStateScheduler` API for multi-tile SSIS generation:** Phase 1 requires calling `generate_ssis()` for each candidate tile. Whether this can be called in isolation per tile without side effects on shared state needs verification against the scheduler source code before Phase 1 implementation begins.

---

## Sources

### Primary (HIGH confidence)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` (1717 lines) — primary authority on existing CO structure, constraint groups, binary variable patterns
- [Gurobi 13.0 Python Model Reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html) — `addVar`, `addConstr`, `addLConstr`, `addGenConstrIndicator`, `addSOS` signatures
- [Gurobi 13.0 Constraint Types Reference](https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/constraints.html) — SOS, indicator, big-M guidance
- [Gurobi 13.0 Release Notes: Changes](https://docs.gurobi.com/projects/optimizer/en/current/reference/releasenotes/changes.html) — `setParam` global behavior change, deprecated APIs
- [Gurobi Tolerances and User-Scaling](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html) — coefficient range `[10^-3, 10^6]` recommendation
- [Gurobi Dealing with Big-M Constraints](https://www.gurobi.com/documentation/9.1/refman/dealing_with_big_m_constra.html) — tight big-M guidance

### Secondary (MEDIUM confidence)
- [Gurobi Community: Select variable from discrete set](https://support.gurobi.com/hc/en-us/community/posts/13302447819537-Select-a-Gurobi-variable-from-a-set-of-items) — confirmed one-hot encoding is the prescribed workaround for discrete non-contiguous sets
- [Gurobi MIP Starts vs Variable Hints](https://support.gurobi.com/hc/en-us/articles/20410834783377-What-are-the-differences-between-MIP-Starts-and-Variable-Hints) — `VarHintVal`/`VarHintPri` warm-start semantics

### Domain Literature (HIGH confidence)
- [Mixed-Integer Linear Programming Formulation Techniques (Vielma, 2015)](https://juan-pablo-vielma.github.io/publications/Mixed-Integer-Linear-Programming-Formulation-Techniques.pdf) — SOS1 encoding, binary product linearization
- [Solving Mixed Integer Bilinear Problems Using MILP Formulations (SIAM 2013)](https://epubs.siam.org/doi/10.1137/110836183) — bilinear product linearization via McCormick bounds
- [From Loop Nests to Silicon: AMD NPUs with MLIR-AIR (2025)](https://arxiv.org/abs/2510.14871) — domain context for AIE tiling and mapping

---
*Research completed: 2026-04-02*
*Ready for roadmap: yes*
