# Stack Research

**Domain:** Variable tile size selection in MILP-based AIE accelerator constraint optimization
**Researched:** 2026-04-02
**Confidence:** HIGH (Gurobi 13.0 docs verified; code analysis of existing CO confirmed)

## Context: What Already Exists

This is a subsequent-milestone research document. The following are already in place and not re-researched:

- `gurobipy` 13.0.0 installed, Gurobi license active
- `TransferAndTensorAllocator` in `transfer_and_tensor_allocation.py` (1700+ lines)
- `z_stop` binary variables selecting reuse levels (one-hot via `quicksum == 1`)
- `_add_binary_product()` linearizing products of two BINARY variables
- `big_m` already threaded through the allocator (`self.big_m = big_m or len(workload.nodes()) + 5`)
- `addConstr`, `quicksum`, `GRB.BINARY`, `GRB.INTEGER` already imported and used

The new work adds tile size selection as a MILP decision. The research below covers exactly what Gurobi API features are needed for that extension.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| gurobipy | 13.0.0 (pinned) | MILP solver interface | Already installed; 13.0 delivers ~16% faster solves on difficult MIP models; no version change needed |
| Python | 3.10+ (existing) | Modeling language | Existing codebase constraint |

No new packages are needed. All required Gurobi modeling primitives are already available in the installed version.

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| gurobipy (existing) | 13.0.0 | One-hot tile selection variables, big-M conditional sizing | All tile selection modeling |
| math.ceil (stdlib) | — | Round up tile-dependent sizes | When computing memory footprints from selected tile sizes |
| collections.defaultdict (stdlib) | — | Accumulate tile-size-dependent linear expressions | Building `core_load` / `bd_depth` sums over tile choices |

---

## Gurobi API: Specific Features for Variable Tile Sizes

### 1. One-Hot Tile Selection Variables

Gurobi does not support integer variables over an arbitrary discrete set (e.g., `{16, 32, 64}`) natively. The standard MILP pattern is a binary one-hot encoding:

```python
# For each unique workload dimension d and each candidate tile size t_k in tile_list:
b[d][k] = model.addVar(vtype=GRB.BINARY, name=f"tile_sel_{d}_{t_k}")

# Exactly-one constraint (same pattern as existing z_stop):
model.addConstr(quicksum(b[d][k] for k in range(len(tile_list))) == 1,
                name=f"tile_sel_sos_{d}")

# Recover the selected tile size as a linear expression:
tile_size_expr[d] = quicksum(tile_list[k] * b[d][k] for k in range(len(tile_list)))
```

This is identical in structure to the existing `z_stop` one-hot pattern. `quicksum` is already imported.

### 2. Tile-Dependent Sizes: Big-M Conditional Constraints

When tensor/transfer/SSIS sizes depend on the selected tile, sizes become products of a binary variable and a constant. The existing `_add_binary_product()` handles products of two binary variables. For binary-times-constant (simpler case):

```python
# If tile k is selected, then size_var == tile_list[k] * base_factor:
# Pattern: size_var = sum_k (tile_list[k] * base_factor * b[d][k])
# This is a linear expression -- no auxiliary variable needed.
size_expr = quicksum(tile_list[k] * base_factor * b[d][k]
                     for k in range(len(tile_list)))
```

This is a direct linear expression, not a big-M constraint, because it is a weighted sum of binary variables. No auxiliary variable or big-M is required for this case.

Big-M is needed only when a continuous variable must equal a tile-dependent expression conditional on another binary (e.g., `u * size_expr` where `u` is a placement binary). This follows the existing `_add_binary_product` pattern extended to a linear expression:

```python
# Product of BINARY u and linear tile expression (big-M linearization):
# w = u * sum_k(tile_list[k] * b[d][k])
# Linearize per tile choice:
w_k = model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{...}_{k}")
# If b[d][k] == 0: w_k == 0 (big-M upper bound on w_k)
model.addConstr(w_k <= big_m * b[d][k])
model.addConstr(w_k <= u * tile_list[k])          # or use addGenConstrIndicator
model.addConstr(w_k >= u * tile_list[k] - big_m * (1 - b[d][k]))
size_times_u = quicksum(w_k for k in range(len(tile_list)))
```

Alternatively, use `addGenConstrIndicator` for cleaner formulation when big-M bounds are hard to tighten:

```python
for k, ts in enumerate(tile_list):
    model.addGenConstrIndicator(b[d][k], True,
                                size_var == ts * base_factor,
                                name=f"tile_size_link_{d}_{ts}")
```

### 3. When to Use Indicator Constraints vs. Big-M

| Situation | Recommendation | Rationale |
|-----------|---------------|-----------|
| Tile size drives a constant multiplier on an existing linear expression | Big-M (explicit) | Bounds are known exactly (tile sizes are constants); tight big-M is numerically safe and faster |
| Tile-dependent size in a memory capacity constraint (sum <= cap) | Direct linear expression | Tile sizes are constants; `sum_k(tile_list[k] * factor * b[k])` is already linear -- no big-M needed |
| Tile selection must activate/deactivate a block of constraints | `addGenConstrIndicator` | Cleaner; Gurobi 13.0 converts internally to SOS1 then big-M if bounds are tight |
| Big-M value is unknown or hard to bound tightly | `addGenConstrIndicator` | Avoids numerical issues from loose big-M; Gurobi chooses internally |

Key Gurobi guidance (official docs, verified): indicator constraints are internally translated to SOS1 and then to big-M only when the solver can deduce tight bounds. When explicit big-M is used, the bound must be as tight as possible. For tile sizes (bounded, known constants), explicit big-M is preferred and faster.

### 4. SOS Constraints: When to Use

SOS1/SOS2 constraints are an alternative encoding for the one-hot tile selection. They are not recommended here because:

- The existing `z_stop` pattern uses explicit `quicksum == 1` one-hot encoding successfully
- SOS1 reformulation is controlled by `PreSOS1Encoding` parameter and may introduce unneeded variables
- One-hot binary encoding gives Gurobi explicit bounds (0/1) that presolve can exploit directly
- SOS1 adds overhead for small discrete sets (typical tile lists have 3-8 elements)

Use SOS constraints only if tile list grows beyond ~20 elements, where the logarithmic-size reformulation (`PreSOS1Encoding=2` or `3`) becomes beneficial.

### 5. Tight Big-M Values

The current `self.big_m` is set to `len(workload.nodes()) + 5` -- a count-based heuristic appropriate for binary indicator logic. For tile-size-dependent constraints, a tighter big-M is:

```python
# For memory capacity constraints:
big_m_mem = max(tile_list) * max_size_factor * max_tiles_needed

# For latency constraints:
big_m_lat = max(tile_list) * max_latency_per_tile
```

Tight big-M reduces LP relaxation looseness, which directly reduces branch-and-bound tree size.

---

## Supporting Gurobi Model Methods

| Method | Signature | Use Case |
|--------|-----------|----------|
| `model.addVar` | `addVar(vtype=GRB.BINARY, name=...)` | Tile selection binary variables `b[d][k]` |
| `model.addVars` | `addVars(*indices, vtype=GRB.BINARY)` | Vectorized creation of tile selection variables |
| `model.addConstr` | `addConstr(quicksum(...) == 1)` | One-hot exactly-one constraint |
| `model.addGenConstrIndicator` | `addGenConstrIndicator(bin, True, lhs == rhs)` | Conditional size assignment when big-M is loose |
| `quicksum` | `quicksum(c_k * b_k for k in ...)` | Constructing tile-size linear expressions |
| `model.addLConstr` | `addLConstr(lhs, GRB.EQUAL, rhs)` | ~50% faster than `addConstr` for pure linear constraints |

`addLConstr` is a 13.0-available optimization: replacing `addConstr` with `addLConstr` for all linear (non-indicator, non-quadratic) constraints in the hot path is a low-effort speedup worth applying during this milestone.

---

## Performance Considerations for Adding Integer Variables

### Model Size Impact

Adding one tile selection variable per unique workload dimension. For SwiGLU BIG BOY (3 unique dims: seq_len, embedding, hidden), this adds 3 binary variables per candidate tile size. With a tile list of 8 candidates, that is 24 new binary variables. This is negligible relative to the existing model.

The larger impact is that tensor sizes, memory loads, and latencies become linear expressions over these new binaries instead of constants. Every constraint that currently has a constant `tensor_size` will now have a `quicksum` expression. This increases constraint density, which affects LP relaxation solve time.

### Recommended Solver Parameters

Add these parameters to `TransferAndTensorAllocator.__init__` or a new `configure_solver()` method:

```python
# Existing:
self.model.setParam("OutputFlag", gurobi_verbosity)

# Add for variable tile milestone:
self.model.setParam("MIPFocus", 1)       # find feasible solutions quickly first
self.model.setParam("Presolve", 2)       # aggressive presolve to eliminate redundant binaries
self.model.setParam("Symmetry", -1)      # auto symmetry detection (default; leave unless solving degrades)
self.model.setParam("Cuts", -1)          # auto cuts (default; only change if root relaxation is weak)
self.model.setParam("NumericFocus", 1)   # mild extra care for numerics from new big-M constraints
```

`MIPFocus=1` is the most impactful: DSE runs benefit from finding a good feasible solution fast rather than proving optimality from the start.

Do not set `TimeLimit` during development -- it can mask convergence issues. Add it for production DSE sweeps.

### Symmetry Considerations

If tile sizes are applied per-dimension but the tile list is the same for all dimensions, the solver may see symmetric assignments (different dimensions, same tile size). Gurobi's automatic symmetry detection (`Symmetry=-1`) handles this. No manual symmetry breaking is needed for the 3-dimension SwiGLU case.

### Warm Starting Between DSE Iterations

For DSE sweeps over multiple tile configurations, the existing fixed-tile allocations can seed variable hints:

```python
# After solving a fixed-tile baseline, set hints for the nearest tile size:
b[d][nearest_k].setAttr("VarHintVal", 1.0)
b[d][nearest_k].setAttr("VarHintPri", 1)
```

`VarHintVal`/`VarHintPri` are variable attributes (not parameters); set via `.setAttr()`. In Gurobi 13.0, the NoRel heuristic now respects `VarHintVal`, making warm starts more effective than in prior versions.

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `GRB.INTEGER` variable for tile size | Gurobi integers are over all integers in [lb, ub], not discrete sets; cannot constrain to {16, 32, 64} directly without additional constraints that are more complex than one-hot | Binary one-hot `GRB.BINARY` with `quicksum == 1` |
| SOS2 constraints for tile selection | SOS2 requires contiguity of non-zeros; overkill for pure discrete selection; adds PreSOS2Encoding complexity | SOS1 equivalent via one-hot binary |
| `addGenConstrPoly/Exp/Log` | Deprecated in Gurobi 13.0; removed API | `addGenConstrNL` if nonlinear is needed (not needed here -- tile sizes are discrete constants) |
| `Xn` / `PoolObjVal` attributes | Deprecated in Gurobi 13.0 | `PoolNX` / `PoolNObjVal` (if solution pool is used) |
| Global `gp.setParam()` | Gurobi 13.0 changed behavior: global `setParam` no longer affects already-created Model objects | `model.setParam()` per model instance (already the pattern in existing code) |
| Loose big-M (e.g., `1e6`) | Degrades LP relaxation quality, increases branch-and-bound tree, risks numerical issues | Compute tight big-M from actual tile size bounds |

---

## Integration with Existing Code

### Where to Add Tile Selection Variables

The `_build_model()` method (or equivalent initialization flow) in `TransferAndTensorAllocator` should gain a new `_init_tile_selection_vars()` method:

```python
def _init_tile_selection_vars(self, tile_list: list[int]) -> None:
    """Create binary one-hot variables for tile size selection per unique dimension."""
    self.tile_sel: dict[str, dict[int, gp.Var]] = {}
    self.tile_size_expr: dict[str, gp.LinExpr] = {}
    for dim_name in self.unique_dims:
        self.tile_sel[dim_name] = {}
        for ts in tile_list:
            v = self.model.addVar(vtype=GRB.BINARY, name=f"tile_{dim_name}_{ts}")
            self.tile_sel[dim_name][ts] = v
        self.model.addConstr(
            quicksum(self.tile_sel[dim_name].values()) == 1,
            name=f"tile_sos1_{dim_name}"
        )
        self.tile_size_expr[dim_name] = quicksum(
            ts * v for ts, v in self.tile_sel[dim_name].items()
        )
```

### Where Sizes Must Become Expressions

Currently `tensor_size = self.workload.get_tensor_of_transfer_to_single_core(t, tr, self.mapping).size_bits()` returns a Python `int`. For variable tile sizes, this must return a `gp.LinExpr` dependent on `self.tile_size_expr`. The key change point is in `_memory_capacity_constraints()` at line 726.

The `req_size = ceil(size_factor * tensor_size)` pattern will need to change: `ceil()` cannot be applied to a `LinExpr`. The rounding must be absorbed into the size formula or handled via a conservative upper bound.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Binary one-hot per candidate tile size | `GRB.INTEGER` with auxiliary constraints | Never for this problem -- discrete non-contiguous set |
| Explicit big-M (tight bounds from tile constants) | `addGenConstrIndicator` | Use indicator constraints if tile-derived big-M is still loose (>1000x problem scale), which is unlikely here |
| One-hot per workload dimension | One-hot per tensor / per transfer node | More flexibility but explodes variable count; single per-dimension is sufficient given PROJECT.md scope |

---

## Version Compatibility

| Package | Version | Notes |
|---------|---------|-------|
| gurobipy | 13.0.0 | Current install; all APIs listed here are available; deprecated function constraints (`addGenConstrExp` etc.) must not be used |
| Python | 3.10+ | `match`/structural pattern matching available if needed for tile dispatch logic |

---

## Sources

- [Gurobi 13.0 Constraint Types Reference](https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/constraints.html) — SOS, indicator, general constraint API (HIGH confidence)
- [Gurobi 13.0 Python Model Class Reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html) — `addVar`, `addConstr`, `addLConstr`, `addGenConstrIndicator`, `addSOS` signatures (HIGH confidence)
- [Gurobi 13.0 Release Notes: Changes](https://docs.gurobi.com/projects/optimizer/en/current/reference/releasenotes/changes.html) — `setParam` global behavior change, deprecated attributes, new parameters (HIGH confidence)
- [Gurobi Parameter Guidelines](https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html) — `MIPFocus`, `Presolve`, `Cuts`, `ImproveStartTime` recommendations (HIGH confidence)
- [Gurobi Numeric Parameters](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/numeric_parameters.html) — `NumericFocus`, `ScaleFlag`, tolerance guidance (HIGH confidence)
- [Gurobi Community: Select variable from discrete set](https://support.gurobi.com/hc/en-us/community/posts/13302447819537-Select-a-Gurobi-variable-from-a-set-of-items) — Confirmed no native discrete-set integer type; one-hot encoding is the prescribed workaround (MEDIUM confidence)
- [Gurobi: Dealing with big-M constraints](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintgeneral.html) — Tight big-M guidance (HIGH confidence, redirected from legacy URL)
- [Gurobi: MIP Starts vs Variable Hints](https://support.gurobi.com/hc/en-us/articles/20410834783377-What-are-the-differences-between-MIP-Starts-and-Variable-Hints) — `VarHintVal`/`VarHintPri` for warm start (MEDIUM confidence)
- Code analysis: `transfer_and_tensor_allocation.py` lines 70-95, 446-572, 726-793, 1373-1385 — existing `z_stop` one-hot pattern, `_add_binary_product`, `big_m` usage (HIGH confidence, direct source)

---
*Stack research for: Variable tile size MILP optimization in stream_aie CO*
*Researched: 2026-04-02*
