# Stream AIE — Variable Tile Size Optimization

## What This Is

A design space exploration (DSE) and constraint optimization framework for AMD AIE accelerators. It takes ONNX workloads (e.g., SwiGLU), maps them onto a multi-core AIE array, and uses Gurobi-based constraint optimization to find optimal tensor/transfer allocations across timeslots and cores.

## Core Value

Enable the constraint optimizer to explore variable tile sizes across workload dimensions, finding better allocation solutions than fixed tiling allows.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- Existing pipeline: ONNX parsing, workload creation, mapping generation, tiling generation, CO-based allocation, cost model evaluation, steady-state scheduling
- Per-dimension tile size specification via CLI args (seq_len_tile_size, embedding_tile_size, hidden_tile_size)
- SwiGLU workload support with multi-node graph (gate_proj, up_proj, silu, eltwise_mul, down_proj)
- AIE2 hardware backend with whole-array Strix target
- Steady-state iteration space computation with kernel/temporal loop decomposition

### Active

<!-- Current scope. Building toward these. -->

- [ ] Baseline validation with per-dimension tile sizes (BIG BOY: 256x2048x8192, tiles 16/128/32)
- [ ] User-defined list of possible tile sizes as CO input
- [x] Variable tensor sizes in CO (dependent on selected tile) — Validated in Phase 3
- [ ] Variable ComputationNode sizes in CO
- [ ] Variable transfer sizes in CO
- [ ] Variable SSIS loop sizes (kernel and temporal) in CO
- [x] CO selects optimal tile size per unique workload dimension — Validated in Phase 3 (w[dim,k] binary selection variables)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Per-dimension different tile size lists — single list applied to all dimensions for simplicity
- Codegen changes — focus on CO optimization, codegen adapts downstream
- New workload types — SwiGLU is the validation target
- Genetic algorithm allocation — CO path only

## Context

- `tiling_generation.py` computes fixed tile sizes upstream and substitutes them into the mapping
- `transfer_and_tensor_allocation.py` (1717 lines) consumes fixed sizes via SteadyStateIterationSpace
- `z_stop` binary variables in CO select reuse levels but don't vary tile sizes
- `make_swiglu_mapping2()` already accepts per-dimension tile sizes
- `main_swiglu.py` has CLI args for tile sizes; `main_swiglu_dse_single.py` has hardcoded mapping path
- BIG BOY config: seq_len=256, embedding_dim=2048, hidden_dim=8192, tiles 16/128/32, 4 rows x 8 cols, NPU2

## Current Milestone: v2.0 Variable Tile Size Optimization

**Goal:** Integrate variable tile sizes into the constraint optimization, allowing the CO solver to select optimal tile sizes from a user-defined list across all unique workload dimensions.

**Target features:**
- Baseline validation with existing per-dimension tile sizes
- New main file for v2.0 variable tile DSE
- CO variables for tile size selection
- Dynamic tensor/computation/transfer/SSIS sizes dependent on tile selection
- End-to-end validation with SwiGLU BIG BOY config

## Constraints

- **Solver**: Gurobi MILP — tile size selection adds integer variables and big-M constraints
- **Memory**: AIE core local memory is limited — tile sizes must respect memory bounds
- **Branch**: Develop on new branch from current arne/swiglu-dse
- **Compatibility**: Must not break existing fixed-tile pipeline

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Single tile size list for all dimensions | Simplicity; per-dimension lists explode the search space | -- Pending |
| Gurobi MILP for tile selection | Already using Gurobi; natural extension with integer variables | -- Pending |
| SwiGLU as validation target | Current active workload with known-good baseline | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-07 after Phase 3 completion — tile selection variables + memory constraints*
