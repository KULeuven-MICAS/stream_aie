# Architecture Research

**Domain:** MILP constraint optimization with variable tile size selection for AIE accelerator allocation
**Researched:** 2026-04-02
**Confidence:** HIGH (based on direct codebase analysis)

## Standard Architecture

### System Overview

```
CLI / main_swiglu_dse.py
        |
        v
  StageContext (carries all pipeline state)
        |
        v
[parsing] --> [tiling_generation] --> [mapping_generation] --> [CO allocation] --> [cost estimation]
                     |                                                |
                     |                                               v
             TilingGenerationStage                  ConstraintOptimizationAllocationStage
             - compute fusion_splits                      - builds SteadyStateScheduler
             - substitute_loop_sizes_with_tiled_sizes()   - invokes TransferAndTensorAllocator
             - tiled_workload + tiled_mapping ->ctx        - Gurobi MILP solves:
                     |                                        * tensor placement (x_tensor_choice)
                     |                                        * transfer routing (y_path_choice)
                     v                                        * reuse levels (z_stop)
        Workload with fixed dim sizes                         * slot latencies
        (ComputationNode tensors have                   - outputs: TensorAlloc, TransferAlloc,
         concrete shape/size_bits values)                 TensorReuseLevels, total_latency

Key constant derived at tiling time and consumed by CO:
  tensor.size_bits()  <-- used directly in _memory_capacity_constraints()
  SSIS loop sizes     <-- used in _init_transfer_fire_helpers()
```

### Component Responsibilities

| Component | Responsibility | Key Interface |
|-----------|----------------|---------------|
| `TilingGenerationStage` | Converts fusion_splits to fixed dim sizes; produces tiled_workload | `substitute_loop_sizes_with_tiled_sizes()` -> `dict[LayerDim, int]` |
| `SteadyStateIterationSpace` | Ordered intra-core loop descriptor; each IterationVariable has a fixed `.size` | `.get_applicable_temporal_sizes()` -> `list[int]` |
| `TransferAndTensorAllocator` | Gurobi MILP; consumes tiled workload with fixed tensor sizes | `solve()` -> `(TensorAlloc, TransferAlloc, ...)` |
| `_memory_capacity_constraints` | Enforces core memory limits using `tensor.size_bits()` as a scalar constant | Hardcoded constant `req_size = ceil(size_factor * tensor_size)` |
| `_init_transfer_fire_helpers` | Pre-computes `reuse_levels[(t, stop)]` from fixed SSIS sizes | Used by `fires_def` and `reuse_factor_def` constraints |
| `z_stop[(t, stop)]` | Binary variable selecting reuse level for tensor t at loop stop | Drives `fires`, `reuse_factor`, and `core_load` expressions |

## Current Data Flow (Fixed Tile Path)

```
mapping.intra_core_tiling (tile sizes as constants)
        |
        v
TilingGenerationStage.substitute_loop_sizes_with_tiled_sizes()
        |
        v
Workload.with_modified_dimension_sizes(new_sizes)
 -> all Tensor.shape and Tensor.size_bits() become concrete scalars
        |
        v
SteadyStateScheduler.generate_ssis()
 -> SteadyStateIterationSpace per (node, operand)
 -> each IterationVariable.size is a concrete int
        |
        v
TransferAndTensorAllocator.__init__()
 -> _init_transfer_fire_helpers()
    reuse_levels[(t, stop)] = (fires: int, size_factor: int)   <-- ALL SCALARS
        |
        v
_memory_capacity_constraints():
  tensor_size = workload.get_tensor_of_transfer_to_single_core(t, tr, mapping).size_bits()  <-- scalar
  req_size = ceil(size_factor * tensor_size)   <-- scalar coefficient in LinExpr
  core_load[c] += req_size * uz
```

## Dependency Chain for Variable Tile Integration

The critical dependency chain that must become variable in the CO:

```
tile_size (discrete choice from list)
    |
    v
dim_size = workload_dim_size / fusion_split_factor
    |
    v
tensor_shape = f(dim_sizes, operand mapping)       [via get_tensor_shape_with_dimension_sizes]
    |
    v
tensor_size_bits = prod(tensor_shape) * dtype_bits
    |
    v
req_size = size_factor * tensor_size_bits           [size_factor from reuse_levels]
    |
    v
core_load[c] += req_size * uz                       [memory capacity constraint]
    |
    v
core_load[c] <= core.get_memory_capacity()          [feasibility bound]
```

Each step in this chain is currently a scalar constant computed before the MILP. Making tile size variable requires this entire chain to become part of the MILP formulation.

## Recommended Architecture for Variable Tile Integration

### Decision: Joint Selection (Tile + Allocation in One MILP)

Tile size selection should happen **jointly** inside `TransferAndTensorAllocator`, not before or after. Rationale:

- Memory feasibility (core_load <= capacity) depends on both tile size AND tensor placement (z_stop reuse level). These constraints are non-separable — a tile size valid alone may be infeasible at a given reuse level on a given core.
- The latency objective depends on transfer sizes which depend on tile sizes. Optimizing latency requires seeing both simultaneously.
- Pre-solving tile size selection independently (outer loop) is valid but produces a two-level DSE that degrades to enumeration. The Gurobi MILP can explore the joint space more efficiently.
- The existing `_add_binary_scaled_continuous` and `_add_binary_product` linearization helpers already demonstrate the pattern for handling nonlinear products inside the MILP.

### New MILP Variables

```
# One tile selection indicator per unique workload dimension
t_sel[dim, k]  : BINARY
  = 1 iff dimension `dim` uses tile_size_options[k]

# Selected tile size (integer linear combination)
tile_size[dim] : INTEGER
  tile_size[dim] = sum_k( t_sel[dim, k] * tile_size_options[k] )

# Selected tensor size in bits per (tensor, tile_config_index)
# Precomputed as a look-up table OUTSIDE the MILP, indexed by tile choice
tensor_size_lut[tensor, k] : scalar (precomputed, not a variable)
  = tensor.size_bits() when dim uses tile_size_options[k]

# Active tensor size (integer variable, selected by tile choice)
tensor_size_var[tensor] : INTEGER
  tensor_size_var[t] = sum_k( t_sel[dim(t), k] * tensor_size_lut[t, k] )
```

### MILP Formulation Pattern: SOS1 / Big-M for Tile Selection

The cleanest formulation uses SOS1 (Special Ordered Set type 1) constraints for tile selection:

```python
# For each unique workload dimension dim:
t_sel[dim, :] forms an SOS1 set — exactly one is 1
model.addConstr(sum_k(t_sel[dim, k]) == 1)  # one-hot

# Tile size as linear combination (valid because SOS1 enforces one-hot)
tile_size[dim] = sum_k(t_sel[dim, k] * tile_size_options[k])

# Tensor size as linear combination of precomputed constants
# This is key: tensor_size_lut[t, k] is a SCALAR, so the expression stays linear
tensor_size_var[t] = sum_k(t_sel[dim(t), k] * tensor_size_lut[t, k])
```

This avoids big-M entirely for the tensor size computation. Big-M is only needed when linking tile selection to the nonlinear product `req_size * uz` in memory constraints.

### Memory Constraint with Variable Tile Size

The current constraint:
```
req_size * uz  (scalar * binary)
```

becomes with variable tile size:
```
tensor_size_var[t] * size_factor * uz  (integer * scalar * binary)
```

This is a product of an integer variable and a binary variable. Standard linearization:

```
# Introduce auxiliary: w[t, c, stop] = tensor_size_var[t] * uz[t, c, stop]
# (integer-binary product, linearized as follows)
w[t, c, stop] >= 0
w[t, c, stop] >= tensor_size_var[t] - M * (1 - uz[t, c, stop])
w[t, c, stop] <= tensor_size_var[t]
w[t, c, stop] <= M * uz[t, c, stop]

# Then memory constraint:
sum_{t,c,stop}( size_factor[t,stop] * w[t, c, stop] ) <= capacity[c]
```

where `M = max(tensor_size_lut[t, k])` over all k (tight big-M from the look-up table).

Gurobi 10+ also supports `addGenConstrNL` and indicator constraints that can express this more directly, but the big-M linearization above is both reliable and already consistent with how `_add_binary_scaled_continuous` works in the existing code.

### SSIS Loop Sizes with Variable Tile

The SSIS `IterationVariable.size` values are also derived from tile sizes. These are consumed by `_init_transfer_fire_helpers()` to compute `reuse_levels[(t, stop)]`, which are currently scalar look-up table entries.

With variable tile sizes, `reuse_levels[(t, stop)]` also becomes variable. However:
- `reuse_levels[(t, stop)][0]` (fires count) = product of SSIS loop sizes above stop level
- `reuse_levels[(t, stop)][1]` (size_factor) = product of relevant SSIS loop sizes up to stop level

These are products of tile-size-dependent integers. The same precomputed LUT approach applies: precompute `reuse_levels_lut[(t, stop, k)]` for each tile option `k`, then use:

```
fires_var[(t, stop)] = sum_k( t_sel[dim(t), k] * reuse_levels_lut[(t, stop, k)][0] )
size_factor_var[(t, stop)] = sum_k( t_sel[dim(t), k] * reuse_levels_lut[(t, stop, k)][1] )
```

This keeps the MILP linear.

### Transfer Latency with Variable Tensor Size

Current:
```python
latency_constant = tensor.size_bits() / min_bw  # scalar
active_latency = latency_constant / reuse_factor  # constant / variable
```

With variable tile sizes:
```
latency_var[tr] = tensor_size_var[tr.input_tensor] / (min_bw * reuse_factor_var[tr])
```

This is a ratio of two integer variables, which is non-linear. The existing `_add_const_over_linexpr` helper handles constant-over-variable but not variable-over-variable. Options:

1. **Piecewise-linear approximation**: Precompute `latency_lut[tr, k]` for each tile option k, then `latency_var[tr] = sum_k(t_sel[k] * latency_lut[tr, k])`. This is exact (not approximate) because tile sizes are discrete — the latency at each tile option is computable exactly before solve time. This is the recommended approach.

2. **Gurobi addGenConstrNL**: Handle the nonlinear ratio directly. Higher complexity, less portable.

The piecewise approach (option 1) is preferable: it turns a nonlinear expression into a linear combination of precomputed constants, consistent with the LUT-based pattern used throughout the rest of the MILP.

## Component Integration Map

### New vs Modified Components

| Component | Status | Change Description |
|-----------|--------|--------------------|
| `TilingGenerationStage` | Modified | Accept `tile_size_options: list[int]` instead of (or alongside) fixed per-dim tiles; skip substitution when CO is doing variable tile selection |
| `TransferAndTensorAllocator.__init__` | Modified | Accept `tile_size_options: list[int]`; pass through to new tile selection subsystem |
| `TransferAndTensorAllocator._create_vars` | Modified | Add `__create_tile_selection_vars()` call |
| `TransferAndTensorAllocator._create_constraints` | Modified | Add `_tile_selection_constraints()`, `_tensor_size_constraints()` calls; update `_memory_capacity_constraints()` to use `tensor_size_var` |
| `TransferAndTensorAllocator._init_transfer_fire_helpers` | Modified | Build LUT over all tile options instead of single fixed values; populate `reuse_levels_lut` |
| `TileSizeSelector` (new class) | New | Encapsulates tile selection variables, LUT precomputation, and linking constraints; analogous to how `NamespaceConstraints` encapsulates hardware-specific constraints |
| `TileSizeLUT` (new dataclass) | New | Precomputed mapping from `(tensor/transfer, tile_option_index)` to tensor_size_bits, SSIS sizes, reuse_levels, latencies |
| `main_swiglu_dse_v2.py` | New | Entry point that passes `tile_size_options` list to pipeline |

### Existing Helpers Already Available

- `_add_binary_product(a, b)`: Binary AND linearization — reusable for `t_sel[k] * uz` products
- `_add_binary_scaled_continuous(binary_var, continuous_var, continuous_ub)`: Binary-times-integer — reusable for `t_sel[k] * tensor_size_var` if formulated differently
- `_add_binary_times_const_over_linexpr(binary_var, numerator, denominator_expr)`: Used for transfer latency — the piecewise LUT approach replaces the need for this with a simpler linear expression

## Recommended Project Structure Changes

```
stream/
├── opt/
│   └── allocation/
│       └── constraint_optimization/
│           ├── transfer_and_tensor_allocation.py   # Modified: integrates tile selection
│           ├── tile_size_lut.py                    # New: TileSizeLUT precomputation
│           ├── tile_size_selector.py               # New: TileSizeSelector MILP component
│           ├── context.py                          # Minimal modification: pass tile_size_options
│           └── config.py                           # Add tile_size_options to config
├── stages/
│   └── generation/
│       └── tiling_generation.py                    # Modified: skip fixed substitution in variable mode
main_swiglu_dse_v2.py                               # New: variable tile DSE entry point
```

## Architectural Patterns

### Pattern 1: Precomputed LUT + Linear Combination

**What:** All tile-size-dependent quantities (tensor sizes, SSIS loop sizes, reuse levels, transfer latencies) are precomputed as scalar LUTs indexed by tile option, then expressed as linear combinations inside the MILP using `t_sel` binary indicators.

**When to use:** Always, for any quantity that is a deterministic function of the tile size. This keeps the MILP linear and avoids introducing nonlinear constraints.

**Trade-offs:** Requires LUT precomputation before solve time (fast, pure Python). Scales linearly in the number of tile options. No nonlinear solver features needed.

**Example:**
```python
# Pre-solve:
tensor_size_lut: dict[tuple[Tensor, int], int] = {}
for k, tile_size in enumerate(tile_size_options):
    dim_sizes = compute_dim_sizes_for_tile(tile_size)
    for tensor in all_tensors:
        tensor_size_lut[(tensor, k)] = compute_size_bits(tensor, dim_sizes)

# In-MILP:
tensor_size_var[t] = quicksum(
    t_sel[dim, k] * tensor_size_lut[(t, k)]
    for k in range(len(tile_size_options))
)
```

### Pattern 2: SOS1 One-Hot Tile Selection

**What:** Use a single binary indicator per (dimension, tile_option) with an `addConstr(sum == 1)` one-hot constraint. Gurobi can optionally treat these as SOS1 sets for stronger branching.

**When to use:** When the number of tile options is small (< ~20). For larger option sets, consider tighter bound propagation by registering explicit SOS1 sets.

**Trade-offs:** Simple, readable, and already consistent with how `z_stop` one-hot constraints are done in the existing code. Adding `model.addSOS(GRB.SOS_TYPE1, vars)` is optional but helps solver heuristics.

### Pattern 3: Big-M Linearization for Integer-Binary Products

**What:** When a Gurobi integer variable `v` must be multiplied by a binary variable `b`, introduce auxiliary `w = v * b` via:
```
w >= 0
w >= v - M * (1 - b)
w <= v
w <= M * b
```
where `M` is the tightest known upper bound on `v`.

**When to use:** Memory capacity constraint after variable tile selection. `M = max(tensor_size_lut[t, k])` over all k gives a tight bound.

**Trade-offs:** Linear formulation, no nonlinear solver needed. Bound tightness matters for LP relaxation quality — use the LUT maximum, not a global constant.

## Data Flow Changes (Variable Tile Path)

```
StageContext carries tile_size_options: list[int]
        |
        v
TilingGenerationStage (variable mode):
  - does NOT substitute fixed tiles into workload
  - passes original workload + tile_size_options to context
        |
        v
SteadyStateScheduler:
  - generates SSIS for EACH tile option k
  - or generates a parameterized SSIS template
        |
        v
TileSizeLUT.build(workload, tile_size_options):
  - for each k: compute dim_sizes_k, tensor shapes, tensor sizes, SSIS sizes,
    reuse_levels, transfer latencies
  - returns TileSizeLUT (all scalars, pure Python)
        |
        v
TransferAndTensorAllocator(workload, ..., tile_size_options, tile_size_lut):
  - creates t_sel[dim, k] binary variables
  - creates tensor_size_var[t] = sum_k(t_sel[k] * lut[t, k])
  - modifies _memory_capacity_constraints to use tensor_size_var
  - modifies _transfer_fire_rate_constraints to use variable fires/size_factor
  - modifies _slot_latency_constraints to use variable transfer latency
        |
        v
Solver result includes t_sel values -> resolved tile sizes
```

## Build Order Considering Dependencies

Build in this order to validate incrementally and avoid blocking:

1. **`TileSizeLUT` precomputation** (new standalone module, no Gurobi dependency)
   - Input: original workload, tile_size_options list
   - Output: dict of precomputed scalars per tile option
   - Can be unit-tested independently

2. **Variable tile selection in `TransferAndTensorAllocator` — memory constraint only**
   - Add `t_sel` variables and one-hot constraints
   - Replace scalar `tensor_size` in `_memory_capacity_constraints` with `tensor_size_var`
   - Validate: fixed-tile mode still works (degenerate case: single tile option)

3. **Variable SSIS loop sizes** (fires and size_factor become variable)
   - Update `_init_transfer_fire_helpers` to use LUT per tile option
   - Add `fires_var` and `size_factor_var` constraints
   - Validate against fixed-tile baseline

4. **Variable transfer latency**
   - Replace `_transfer_latency_for_path` scalar with LUT-based linear combination
   - Update `_active_transfer_latency` to use piecewise expression
   - Validate latency objective still matches baseline on single-tile runs

5. **`TilingGenerationStage` variable mode**
   - Add mode flag: when `tile_size_options` is given, skip fixed substitution
   - Pass tile_size_options through StageContext
   - Validate pipeline still runs end-to-end

6. **`main_swiglu_dse_v2.py` new entry point**
   - CLI with `--tile_size_options` flag (comma-separated list)
   - Wires up variable mode end-to-end

## Integration Points

### Internal Boundaries

| Boundary | Current | After Variable Tile |
|----------|---------|---------------------|
| `TilingGenerationStage` -> `StageContext` | sets `workload` with fixed dim sizes | additionally sets `tile_size_options` and `tile_size_lut` |
| `StageContext` -> `TransferAndTensorAllocator` | workload tensors have concrete sizes | workload tensors have original (pre-tile) sizes; LUT carries per-option values |
| `_memory_capacity_constraints` | `req_size` is a Python `int` constant | `req_size` is a Gurobi `LinExpr` (sum of LUT constants * `t_sel`) |
| `_init_transfer_fire_helpers` | `reuse_levels[(t, stop)]` is a `(int, int)` tuple | `reuse_levels_lut[(t, stop, k)]` is a `(int, int)` per tile option; MILP expresses the active value via linear combination |
| `_active_transfer_latency` | `latency_constant` is a float | `latency_var` is a `LinExpr` (sum of per-option constants * `t_sel`) |

### External Services

| Service | Integration | Notes |
|---------|-------------|-------|
| Gurobi | No new solver features required | SOS1 is optional (performance hint only); all linearizations use standard MILP constraints |
| `workload.with_modified_dimension_sizes` | Not called in variable mode | LUT precomputation uses this internally for each tile option |

## Anti-Patterns

### Anti-Pattern 1: Pre-Solving Tile Sizes in an Outer Loop

**What people do:** Enumerate `tile_size_options`, run the full CO for each, pick the best result.

**Why it's wrong:** Misses interactions between tile size and reuse level selection. A tile size that looks bad in isolation (large tiles -> fewer reuses feasible) may be optimal when the CO chooses different z_stop levels. Outer-loop enumeration is also O(N) in solver calls instead of a single solve.

**Do this instead:** Joint selection inside a single MILP invocation.

### Anti-Pattern 2: Using a Non-Tight Big-M Bound

**What people do:** Use a global constant (e.g., `M = 2**32`) for big-M in the integer-binary product linearization.

**Why it's wrong:** A loose big-M weakens the LP relaxation, drastically increasing branch-and-bound nodes and solve time.

**Do this instead:** Set `M = max(tensor_size_lut[t, k] for k in range(len(options)))` — the actual maximum possible tensor size for that tensor. This is computable from the LUT before the solve.

### Anti-Pattern 3: Parameterizing Tensor Objects Instead of Using a LUT

**What people do:** Modify `Tensor.size_bits()` to return a Gurobi variable by making `shape` variable.

**Why it's wrong:** The `Tensor` dataclass is used throughout the pipeline outside the MILP (graph traversal, SSIS construction, codegen). Embedding Gurobi variables inside it creates a hard dependency on the solver state, breaks serialization, and violates separation of concerns.

**Do this instead:** Keep `Tensor` objects as pure data (fixed shape). Carry all variable-tile information in a separate `TileSizeLUT` structure that is only used inside `TransferAndTensorAllocator`.

### Anti-Pattern 4: Building a Separate MILP Stage for Tile Selection

**What people do:** Add a new pipeline stage before `TransferAndTensorAllocator` that solves a separate MILP for tile sizes, then passes the result as fixed input.

**Why it's wrong:** This decomposes a joint optimization into a sequential one. The first-stage MILP would need to predict memory feasibility without knowing the z_stop choices, which requires approximations that undermine the whole-system optimality.

**Do this instead:** Extend `TransferAndTensorAllocator` with tile selection variables. The existing MILP already has all the necessary structure.

## Scaling Considerations

| Scale | Architecture Consideration |
|-------|---------------------------|
| 1-4 tile options | LUT is small; SOS1 not needed; solve time increase is modest |
| 5-10 tile options | Add `model.addSOS(GRB.SOS_TYPE1, t_sel[dim, :])` for all dims; aids branching |
| 10+ tile options | Consider pruning infeasible options from LUT before solve (any tile option that exceeds core capacity for any tensor at stop=-1 is always infeasible, regardless of placement) |
| Multiple unique dimensions each with independent options | Combinatorial search space grows multiplicatively; start with a single shared list (as scoped in PROJECT.md) before extending to per-dim lists |

## Sources

- Direct codebase analysis: `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` (1717 lines)
- Direct codebase analysis: `stream/workload/steady_state/iteration_space.py`
- Direct codebase analysis: `stream/stages/generation/tiling_generation.py`
- Direct codebase analysis: `stream/workload/workload.py` (`with_modified_dimension_sizes`, `get_tensor_shape_with_dimension_sizes`)
- Direct codebase analysis: `stream/opt/allocation/constraint_optimization/context.py` (NamespaceConstraints pattern)
- Gurobi MILP linearization patterns: standard SOS1, big-M, binary-times-integer product

---
*Architecture research for: Variable tile size integration into Gurobi-based MILP CO for AIE accelerator allocation*
*Researched: 2026-04-02*
