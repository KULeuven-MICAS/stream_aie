# Feature Research

**Domain:** Variable tile size optimization in Gurobi MILP constraint optimization for AIE accelerator allocation
**Researched:** 2026-04-02
**Confidence:** HIGH (codebase is the primary source; supplemented with domain literature)

---

## Context

This milestone adds variable tile size selection to an existing Gurobi MILP that already handles:
- Fixed-size tensor/transfer allocation with `z_stop` binary reuse variables
- Steady-state iteration space (SSIS) with kernel/temporal loop decomposition
- Per-dimension CLI tile sizes fed into `tiling_generation.py` before CO runs

The CO currently receives tensor sizes and SSIS loop trip-counts as **constants**. The goal is to make tile sizes **decision variables** that the solver selects from a user-supplied list, with all downstream quantities (tensor sizes, SSIS loop sizes, memory loads, fire counts) becoming dependent expressions.

---

## Feature Landscape

### Table Stakes (Users Expect These)

These features must exist for the v2.0 milestone to be considered complete. Missing any of them means the variable tile DSE cannot run end-to-end.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| User-supplied candidate tile size list | Entry point for all variable tile logic; without it the CO has nothing to select from | LOW | CLI arg or config key; single list applied to all unique dimensions per PROJECT.md decision |
| Binary tile-selection indicator variables | Core CO primitive; one binary per (dimension, candidate) enabling SOS1-style exactly-one selection | MEDIUM | `w[dim, k] in {0,1}` for each candidate `k`; exactly-one constraint per dimension; Gurobi natively supports SOS1 which tightens LP relaxation vs raw big-M |
| Variable tensor sizes dependent on selected tile | `tensor_size` in `_memory_capacity_constraints` is currently a constant; must become a linear expression parameterized by `w` | HIGH | Size = sum over candidates of `(size_for_candidate_k * w[dim, k])`; requires auxiliary bilinear products `tensor_size_var * z_stop` to be re-linearized; this is the hardest constraint to rework |
| Variable SSIS temporal loop sizes | SSIS `IterationVariable.size` values are Python ints baked at construction; they feed `tiles_needed_levels`, `reuse_levels`, `bds_needed_levels` — all must become Gurobi linear expressions | HIGH | Requires either symbolic SSIS construction or re-deriving loop trip-counts inside the model as linear combinations of `w` variables |
| Variable ComputationNode sizes (tiled workload) | `tiling_generation.py` calls `workload.with_modified_dimension_sizes()` with fixed sizes; for variable tile CO this substitution must be deferred or parameterized | MEDIUM | Can be handled by constructing one workload per candidate and selecting inside the model, or by parameterizing the existing workload construction path |
| Variable transfer sizes | Transfer latency in `_transfer_latency_cache` uses tensor bit-width; must become a variable expression when tensor size is variable | MEDIUM | Transfer latency = `size_bits / bandwidth`; if `size_bits` is a linear expression in `w`, latency becomes a linear expression too |
| Divisibility / validity constraints | Not all tile sizes divide all dimension sizes; the solver must only select valid candidates | LOW | Pre-filter candidate list before constructing `w` variables: `candidates[dim] = [k for k in all_candidates if dim_size % k == 0]` |
| End-to-end validation with SwiGLU BIG BOY | Confirms the full pipeline works with variable tile selection: 256x2048x8192, tiles drawn from a candidate list | MEDIUM | Baseline fixed-tile run first, then variable run, results should bracket or improve on the fixed baseline |

### Differentiators (Competitive Advantage)

Features that go beyond minimum correctness and provide meaningful optimization value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-unique-dimension tile selection | Allows the solver to pick different tile sizes for seq_len vs embedding_dim vs hidden_dim, rather than one global tile size | MEDIUM | PROJECT.md already establishes that unique dimensions are tracked; the `w[dim, k]` formulation extends naturally per-dimension; avoids collapsing to a single tile that suboptimizes some dimensions |
| Memory-capacity-aware tile elimination | Pre-prune candidate list per core/dimension based on memory capacity before adding to the model, reducing binary variable count | LOW | Run a quick feasibility check: `size_for_candidate * max_size_factor <= core_mem_cap`; infeasible candidates are removed before `w` variables are created |
| Objective term for tile efficiency | Add a secondary objective term penalizing small tiles (underutilized PEs) or favoring tiles aligned to vector width | MEDIUM | Kernel utilization already in cost LUT; extend it to include a utilization lookup indexed by tile choice; blended into the existing latency/energy objective |
| Warm-start from fixed-tile baseline | Run the fixed-tile allocation first, extract its allocations as initial hints for Gurobi in the variable-tile model | LOW | Gurobi supports `VarHintVal`; the fixed baseline corresponds to forcing all `w[dim, k*]` = 1 for the fixed size k* |
| Tile size symmetry breaking | When multiple candidates produce equivalent allocations, add symmetry-breaking constraints to speed up search | MEDIUM | Applicable when two candidates differ only in a dimension irrelevant to a given tensor's memory footprint |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Per-dimension independent candidate lists | Seems like it should give more freedom | Multiplies the number of `w` variable combinations combinatorially; PROJECT.md explicitly puts this out of scope because it explodes search space | Use a single shared candidate list; all dimensions draw from the same set of values |
| Continuous tile size variable (not from a list) | Appears more general | Makes tensor sizes non-linear (product of continuous variables); cannot be handled by MILP without a major reformulation to MIQP or piecewise-linear approximation | Enumerate a discrete candidate list; 8-16 candidates is tractable for Gurobi |
| Variable spatial unrolling / inter-core tiling jointly with tile size | Seems like a natural joint optimization | The inter-core tiling structure determines the number of active cores and the workload graph topology; changing it requires rebuilding the graph, not just adjusting constraints | Fix inter-core tiling at mapping generation time; only vary the intra-tile (temporal loop) sizes in the CO |
| Automatic candidate list generation (e.g., all powers of 2 up to dim_size) | Convenient | Unpruned lists create many infeasible or dominated candidates; large candidate sets increase model size and solve time quadratically | Require explicit user-supplied list; document divisibility requirement; provide a helper function that validates and filters the list |
| Tile size variation across layers (gate_proj vs down_proj) | Maximum flexibility | SwiGLU layers share dimensions (D0=seq_len, D1=embedding, D2=hidden); if layers use different tile sizes for the same logical dimension, the SSIS shared-total-size invariant (`_ensure_same_ssis_for_all_transfers`) is violated | Enforce one tile size per unique workload dimension, shared across all layers that share that dimension |

---

## Feature Dependencies

```
[User-supplied candidate list]
    └──required-by──> [Binary tile-selection indicators w[dim, k]]
                          └──required-by──> [Variable tensor sizes]
                          |                     └──required-by──> [Memory capacity constraints (reworked)]
                          |
                          └──required-by──> [Variable SSIS loop sizes]
                                                └──required-by──> [reuse_levels / tiles_needed_levels (reworked)]
                                                └──required-by──> [Variable transfer sizes]
                                                └──required-by──> [Variable ComputationNode sizes]

[Divisibility/validity constraints]
    └──required-by──> [Binary tile-selection indicators w[dim, k]]

[Baseline fixed-tile validation]
    └──must-precede──> [Variable tile CO run]
    └──enables──> [Warm-start from fixed-tile baseline]

[Variable tensor sizes] ──bilinear-product-linearization──> [Memory capacity constraints]
    (tensor_size_var * z_stop product must be linearized with auxiliary variable + big-M)

[Memory-capacity-aware tile elimination]
    └──enhances──> [Binary tile-selection indicators w[dim, k]]
    (reduces variable count before model is built)
```

### Dependency Notes

- **Variable tensor sizes requires bilinear linearization:** The existing `_memory_capacity_constraints` multiplies `tensor_size * z_stop`. With variable `tensor_size` becoming a linear expression, the product becomes bilinear in `(w[dim,k], z_stop)`. Standard approach: introduce auxiliary binary `v[dim, k, stop] = w[dim,k] * z_stop[tensor, stop]` and add big-M constraints. Gurobi's `addGenConstrAnd` or manual big-M product linearization (`_add_binary_product` already exists in the codebase) handles this.

- **Variable SSIS loop sizes requires deep refactoring:** `_init_transfer_fire_helpers` bakes Python int values from `IterationVariable.size` into `tiles_needed_levels` and `reuse_levels`. These dicts are then used as constants in constraint expressions. With variable tile sizes, these must become Gurobi `LinExpr` objects. The cleanest approach: keep the SSIS structure, but after selecting a tile size (i.e., fixing `w`), recompute the SSIS-derived constants. For a MILP formulation, this means expressing each `tiles_needed_levels[(t, stop)]` as `sum_k(tiles_needed_for_k * w[dim, k])` where `tiles_needed_for_k` is precomputed for each candidate.

- **Variable ComputationNode sizes must respect existing workload immutability:** `tiling_generation.py` creates a new `tiled_workload` for each fixed tile size. For variable tile CO, one option is to instantiate one workload per candidate tile combination, keep them all in memory, and use `w` to select which workload's tensors are active. A simpler option: construct the workload with a symbolic tile size, computing tensor shapes as expressions over dimension sizes.

- **SSIS shared-total-size invariant restricts tile size independence per layer:** `_ensure_same_ssis_for_all_transfers` asserts all transfer nodes have the same total SSIS iteration count. This is satisfied when all layers share the same tile size for each unique dimension — which is exactly the constraint the single-list-per-dimension design enforces.

---

## MVP Definition

### Launch With (v1 — Milestone v2.0)

Minimum needed to claim variable tile size CO is working.

- [ ] **Baseline validation run** — Run existing CO with fixed tile sizes from CLI, confirm correct solution before touching variable tile code. Creates the correctness reference.
- [ ] **Candidate list input path** — Accept a list of candidate tile sizes (e.g., `[16, 32, 64, 128]`) as a single shared list for all unique dimensions.
- [ ] **Binary selection variables `w[dim, k]`** — One binary per (unique dimension, candidate); exactly-one constraint per dimension.
- [ ] **Precomputed per-candidate auxiliary tables** — For each candidate tile size, precompute tensor sizes, SSIS loop sizes, tiles_needed, reuse factors. Store as Python dicts indexed by candidate.
- [ ] **Linearized variable tensor sizes in memory capacity constraints** — Replace constant `tensor_size` with `sum_k(tensor_size_k * w[dim, k])`; re-linearize the `tensor_size * z_stop` bilinear product.
- [ ] **Linearized variable SSIS loop sizes** — Replace constant `tiles_needed_levels` and `reuse_levels` lookups with linear expressions over `w`.
- [ ] **Divisibility pre-filtering** — Before constructing `w`, filter out candidates that do not divide each dimension size.
- [ ] **End-to-end SwiGLU BIG BOY validation** — Confirm CO selects a valid tile size and produces an allocation with lower or equal latency than the fixed baseline.

### Add After Validation (v1.x)

- [ ] **Per-dimension candidate filtering based on memory capacity** — Prune infeasible candidates before building `w` variables; reduces model size.
- [ ] **Warm-start from fixed-tile baseline** — Pass fixed-tile allocation as Gurobi variable hints to speed up search.
- [ ] **Tile efficiency term in objective** — Penalize tiles that under-utilize the kernel (utilization below threshold); use kernel cost LUT lookup per candidate.

### Future Consideration (v2+)

- [ ] **Per-layer tile size selection** — Requires relaxing the SSIS shared-total-size invariant; significant refactoring of `_ensure_same_ssis_for_all_transfers`.
- [ ] **Joint tile size and dataflow (inter-core tiling) optimization** — Requires rebuilding the workload graph as part of the optimization loop; fundamentally different architecture.
- [ ] **Learned tile size heuristics** — Use prior CO solutions to pre-rank candidates, feeding a smaller shortlist to the MILP; reduces solve time at the cost of some optimality.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Candidate list input path | HIGH | LOW | P1 |
| Binary selection variables w[dim, k] | HIGH | LOW-MEDIUM | P1 |
| Precomputed per-candidate auxiliary tables | HIGH | MEDIUM | P1 |
| Linearized variable tensor sizes (memory capacity) | HIGH | HIGH | P1 |
| Linearized variable SSIS loop sizes | HIGH | HIGH | P1 |
| Divisibility pre-filtering | HIGH | LOW | P1 |
| Baseline fixed-tile validation | HIGH | LOW | P1 |
| End-to-end SwiGLU BIG BOY validation | HIGH | MEDIUM | P1 |
| Memory-capacity-aware candidate pruning | MEDIUM | LOW | P2 |
| Warm-start from fixed-tile baseline | MEDIUM | LOW | P2 |
| Tile efficiency objective term | MEDIUM | MEDIUM | P2 |
| Per-layer tile size selection | LOW | HIGH | P3 |
| Joint tile+dataflow optimization | LOW | VERY HIGH | P3 |

**Priority key:**
- P1: Must have for milestone v2.0
- P2: Should have, add when core is validated
- P3: Nice to have, future milestone

---

## Interaction Notes: Tile Sizes, Memory, Data Reuse, and Loop Bounds

These are not standard "competitor" patterns but rather the domain constraints that define what variable tile selection must respect in this codebase.

### Tile size governs tensor size (not just loop count)

In this framework, the tiled dimension size feeds into `get_tensor_shape_with_tiling`, which computes the actual tensor shape (and hence `size_bits()`) for memory constraint evaluation. Smaller tiles → smaller tensors → more can fit on a core → more data reuse is feasible. The relationship is linear in tile size for rectangular tilings of independent dimensions.

### Memory capacity constraint is bilinear under variable tile sizes

The constraint `sum_tensors(size_factor * tensor_size * uses_core * z_stop) <= mem_cap` has two binary variables multiplied by a size that is now a variable. The product `(tensor_size_var) * (z_stop binary)` requires auxiliary variables. The existing `_add_binary_product` helper in `transfer_and_tensor_allocation.py` handles binary × binary products with big-M; extending it to handle (linear expression) × binary is a natural generalization using the same big-M pattern.

### SSIS loop sizes determine fire counts and tiles_needed

`tiles_needed_levels[(tensor, stop)] = product of relevant temporal loop sizes up to stop`. If temporal loop sizes change with tile selection, so do the fire counts (`reuse_levels`) and buffer counts (`tiles_needed`). For a fixed set of candidates, each candidate produces a concrete set of these constants, so the variable version is a SOS1-style selection among precomputed constant tables.

### The "same SSIS total size" invariant constrains independent per-layer tile variation

`_ensure_same_ssis_for_all_transfers` requires all transfers to have the same product of temporal loop sizes. When tile sizes change, `total_iterations = dim_size / tile_size` changes too. As long as all layers sharing a given logical dimension use the same tile size (the single-list design), the total iteration count changes uniformly across all transfers and the invariant holds. This is the main reason per-layer tile independence is deferred.

### Object FIFO depth and buffer descriptor counts scale with tiles_needed

AIE object FIFOs have a hardware depth limit. `_object_fifo_depth_constraints` accumulates `tiles_needed * uses_core * z_stop` per core. With variable `tiles_needed`, this becomes a linear expression in `w` before further multiplication by `z_stop`, requiring the same auxiliary variable linearization as the memory capacity constraints.

---

## Sources

- Codebase: `/home/micas/stream_aie/stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` (1717 lines, primary authority)
- Codebase: `/home/micas/stream_aie/stream/stages/generation/tiling_generation.py`
- Codebase: `/home/micas/stream_aie/stream/workload/steady_state/iteration_space.py`
- Codebase: `/home/micas/stream_aie/stream/inputs/aie/mapping/make_swiglu_mapping.py`
- Domain: [Tile Size Selection for Optimized Memory Reuse in High-Level Synthesis](https://johnwickerson.github.io/papers/tilesize_FPL17.pdf) (FPL 2017)
- Domain: [Energy-Aware Tile Size Selection for Affine Programs on GPUs](https://malithjayaweera.com/wp-content/uploads/2024/01/CGO24_eatss_PREPRINT.pdf) (CGO 2024)
- Domain: [Special Ordered Sets in MILP](https://en.wikipedia.org/wiki/Special_ordered_set)
- Domain: [Solving Mixed Integer Bilinear Problems Using MILP Formulations](https://epubs.siam.org/doi/10.1137/110836183) (SIAM 2013)
- Domain: [From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR](https://arxiv.org/abs/2510.14871) (2025)
- Project context: `/home/micas/stream_aie/.planning/PROJECT.md`

---
*Feature research for: Variable tile size optimization in AIE MILP constraint optimizer*
*Researched: 2026-04-02*
