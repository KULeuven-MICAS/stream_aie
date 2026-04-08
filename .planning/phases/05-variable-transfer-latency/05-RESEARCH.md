# Phase 05: Variable Transfer Latency - Research

**Researched:** 2026-04-08
**Domain:** Gurobi MILP — linearizing a nonlinear (division) constraint into pure binary/continuous products
**Confidence:** HIGH

## Summary

Transfer latency is currently computed as `ceil(tensor_size_bits / min_bw) / reuse_factor` where `tensor_size_bits` is a fixed scalar and `reuse_factor` is a Gurobi INTEGER variable. The division is implemented via `addGenConstrNL` — the only nonlinear constraint in the model. Phase 5 eliminates this by pre-computing `amortized_latency[k, s] = ceil(size_bits[k] / min_bw) / sf_coeff[k, s]` as a scalar for each (tile-candidate, stop-level) pair and selecting the active coefficient using binary products — a pattern already established in Phases 3-4.

The core insight is that `tensor_size_bits[k]` varies per tile candidate (already enumerated by `_joint_candidates_for_tensor`) and `sf_coeff[k, s]` (the reuse size-factor coefficient for candidate `k` at stop level `s`) is also a pre-computed scalar stored in `reuse_levels[(t, s)]`. Both are scalars at model-build time; only the active combination is unknown at solve time. This means the quotient is a pre-computable scalar and the only "variable" part is which `(k, s)` pair is active — gated by `joint_w[k] * z_stop[s]`, then further gated by path choice `y`.

**Primary recommendation:** Refactor `_active_transfer_latency` to enumerate `(k, s)` pairs using data already in `_joint_candidates_for_tensor` and `reuse_levels`, compute `amortized_latency[k,s]` as a scalar coefficient, then use `_add_binary_product` + `_add_binary_scaled_continuous` to produce a pure MILP latency variable. Remove or tombstone `_add_const_over_linexpr` and `_add_binary_times_const_over_linexpr` since latency was their only consumer.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Pre-compute per-candidate latency coefficients `latency_lut[k] = ceil(tensor_size_bits[k] / min_bw)` for each joint candidate combination, reusing the same `_joint_candidates_for_tensor` enumeration pattern from Phase 3. Bandwidth (`min_bw`) is path-specific but tile-independent, so only `tensor_size_bits` varies per candidate.
- **D-02:** For multi-dimensional tensors, joint candidate enumeration across tiled dimensions produces the linearization (same as Phase 3 memory constraints). Single-dimensional tensors use direct per-candidate coefficients.
- **D-03:** The existing `_add_const_over_linexpr` uses `addGenConstrNL` (line 1707) for `constant / linexpr` division. This is the only nonlinear constraint in the entire model. It must be eliminated and replaced with a pure MILP formulation.
- **D-04:** Replace the `constant / reuse_factor` ratio with enumeration over (tile_candidate, stop_level) pairs. For each pair `(k, s)`, pre-compute `amortized_latency[k,s] = ceil(tensor_size[k] / min_bw) / reuse_factor_coeff[k,s]`. The active combination is selected by `joint_w[k] AND z_stop[s]` binary products, gated by path choice `y`. All scalar coefficients x binary variables — pure MILP.
- **D-05:** The (tile_candidate, stop_level) enumeration produces binary products `joint_w[k] * z_stop[s]` via `_add_binary_product`. The result is further gated by path choice `y` via `_add_binary_scaled_continuous`. This follows established patterns from Phases 3-4.
- **D-06:** `_add_const_over_linexpr` and `_add_binary_times_const_over_linexpr` are eliminated or refactored. The new latency formulation does not need a ratio helper — it's all pre-computed coefficients x binary products. If these helpers are not used elsewhere, they can be removed entirely.
- **D-07:** `_transfer_latency_for_path` (line 243) stays as-is for computing per-candidate latency scalars (called once per candidate in the enumeration loop). It just won't be called with the default tensor shape anymore.
- **D-08:** Transfer latency is the ONLY remaining tile-dependent scalar in the CO model. All other quantities were linearized in Phases 3-4.
- **D-09:** DMA constraints, force_nonconstant_reuse_levels, variable bounds, and computation node latencies are tile-independent and require no changes.

### Claude's Discretion

- Whether to keep `_add_const_over_linexpr`/`_add_binary_times_const_over_linexpr` as dead code or remove them
- How to structure the (k, s) pair enumeration loop (inline in `_active_transfer_latency` vs separate helper)
- Whether the latency cache `_transfer_latency_cache` needs restructuring for per-candidate values
- Unit test design for the new pure MILP latency formulation

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CO-03 | Transfer sizes and latencies become linear expressions over tile selection variables | D-01 through D-07 document the full linearization path; `_joint_candidates_for_tensor` + `reuse_levels` variable-mode data provides all needed scalars; `_add_binary_product` + `_add_binary_scaled_continuous` implement the MILP gating |
</phase_requirements>

## Standard Stack

No new packages. Phase 5 uses only what is already imported.

### Core (Already Present)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gurobipy | project version | MILP model building | Project solver |
| math.ceil | stdlib | Per-candidate latency coefficient computation | Same as `_transfer_latency_for_path` |
| itertools.product | stdlib | Joint candidate enumeration | Already used in `_joint_candidates_for_tensor` and `_ssis_coefficients_for_transfer` |

### Installation
No new installation required.

## Architecture Patterns

### Established Patterns Already in Codebase

#### Pattern 1: Per-candidate coefficient enumeration with joint binary variable
Used in `_joint_candidates_for_tensor` (Phase 3 memory) and `_ssis_coefficients_for_transfer` (Phase 4 SSIS). The Phase 5 latency LUT follows the same shape.

**What:** For each joint tile-candidate combination `k`, compute a scalar coefficient and pair it with the joint binary variable `joint_w[k]` (product of `w[dim, idx]` for each tiled dimension in the combination).

**When to use:** Whenever a tile-dependent quantity needs to become a linear expression in a MILP. The scalar coefficients are pre-computed at model-build time; `joint_w[k]` selects the active one.

#### Pattern 2: Big-M continuous linearization of `z_stop * expression`
Used in `_reuse_factor_rate_constraints` (Phase 4). Template:
```python
# Source: transfer_and_tensor_allocation.py lines 840-849
lc = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=M, name=f"lat_lc_{tr.name}_L{s}")
model.addConstr(lc <= expr,           name=f"lat_lc_ub_expr_{tr.name}_L{s}")
model.addConstr(lc <= M * z,          name=f"lat_lc_ub_m_{tr.name}_L{s}")
model.addConstr(lc >= expr - M*(1-z), name=f"lat_lc_lb_{tr.name}_L{s}")
```
For latency: `expr = quicksum(amortized_coeff[k,s] * joint_w[k] for k)`, `z = z_stop[(t, s)]`.

#### Pattern 3: _add_binary_scaled_continuous for path gating
Used throughout the model for `y * continuous`. Produces a new `gp.Var` with bounds `[0, continuous_ub]` and three Big-M constraints. Signature:
```python
# Source: transfer_and_tensor_allocation.py lines 1715-1754
result = self._add_binary_scaled_continuous(
    binary_var=y,
    continuous_var=lc_sum_var,   # intermediate continuous variable
    continuous_ub=max_amortized_latency,
    base_name=f"transfer_latency_{tr}_{hash(choice)}",
)
```

### Proposed Structure for `_active_transfer_latency` (Variable Mode)

```
for each stop level s:
    for each joint candidate k:
        amortized[k, s] = ceil(size_bits[k] / min_bw) / sf_coeff[k, s]
    
    expr_s = quicksum(amortized[k,s] * joint_w[k] for k)
    M_s    = max(amortized[k,s] for k)
    lc_s   = new CONTINUOUS var [0, M_s]
    constrain lc_s == z_stop[s] * expr_s  (big-M)

lat_sum = quicksum(lc_s for s)
lat_var = new CONTINUOUS var [0, sum(M_s)]
constrain lat_var == lat_sum

active_latency = _add_binary_scaled_continuous(y, lat_var, ub=sum(M_s))
```

In scalar mode (no tiled dims or empty search space): fall back to the existing `_add_binary_times_const_over_linexpr` path, OR reimplement equivalently using the same enumeration with a single candidate. Both are correct; using the same code path for both modes simplifies the implementation.

### Key Data Available at Call Time

`_active_transfer_latency(tr, choice, y)` already has:
- `choice`: `MulticastPathPlan` — `min_bw = min(link.bandwidth for link in choice.links_used)` (path-specific, tile-independent)
- `tr`: `TransferNode` — `tr.inputs[0]` gives the input tensor
- `self._joint_candidates_for_tensor(tensor, tr)` → `list[(size_bits, joint_w)]` — per-candidate size and binary variable
- `self.reuse_levels[(t, s)]` in variable mode → `list[(fires_c, sf_c, jw)]` — per (candidate, stop) sf coefficients

The mapping between the `k`-th candidate in `_joint_candidates_for_tensor` and the `k`-th triple in `reuse_levels[(t, s)]` holds because both enumerate candidates in the same `iproduct` order (both call `_joint_binary_for_combo` on the same `per_dim_options`). However, there is an important distinction: `_joint_candidates_for_tensor` uses **tensor tiled dims** (`_tiled_dims_for_tensor`) while `_ssis_coefficients_for_transfer` uses **SSIS temporal tiled dims** (`_ssis_tiled_dims_for_transfer`). If these two dim sets differ for a transfer, the candidate lists are not co-indexed. Research finding: verify at plan time whether this can occur in practice and if the two candidate lists are always length-compatible.

### Anti-Patterns to Avoid

- **Calling `_transfer_latency_for_path(tr, choice)` with `tr.inputs[0]` (default tensor shape):** This gives a single scalar ignoring variable tile candidates. Instead call it with a custom tensor mock or inline `ceil(size_bits[k] / min_bw)` using the size already computed in `_joint_candidates_for_tensor`.
- **Using `addGenConstrNL` for any new latency computation:** Violates CO-03 and the pure MILP invariant.
- **Sharing `_transfer_latency_cache` between variable-mode and scalar-mode calls:** The cache key `(tr, choice)` is sufficient as long as the variable-mode path is fully self-contained. The cache stores the final `active_latency` var; no restructuring needed since the output type remains `gp.Var`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Binary product z = a AND b | Custom constraints | `_add_binary_product(a, b, base_name)` | Already in codebase, tested |
| Binary × continuous linearization | Custom Big-M | `_add_binary_scaled_continuous(binary_var, continuous_var, continuous_ub, base_name)` | Already in codebase, tested |
| Joint binary variable for tile candidate combo | New variable creation | `_joint_binary_for_combo(per_dim_options, combo)` | Already in codebase, reused by Phases 3-4 |
| Per-candidate tensor size enumeration | Recomputing from scratch | `_joint_candidates_for_tensor(tensor, tr)` | Already in codebase, cached |
| Per-candidate reuse size_factor coefficients | Recomputing `reuse_coefficients_for_sizes` | `reuse_levels[(t, s)]` entries (index 1 = sf_coeff) | Already computed and stored by `_init_transfer_fire_helpers` |

**Key insight:** All per-candidate scalars needed for Phase 5 are already computed and stored. Phase 5 is primarily a refactor of `_active_transfer_latency` to use them instead of calling the nonlinear helper.

## Common Pitfalls

### Pitfall 1: Mismatched candidate indexing between tensor-tiled-dims and SSIS-tiled-dims
**What goes wrong:** `_joint_candidates_for_tensor` enumerates using dims from `_tiled_dims_for_tensor` (based on `inter_core_tiling`). `reuse_levels` is populated using dims from `_ssis_tiled_dims_for_transfer` (based on SSIS temporal variables). If these sets differ, the `k`-th candidate in the joint-candidates list does not correspond to the `k`-th triple in `reuse_levels[(t, s)]`.

**Why it happens:** Different code paths compute the two sets independently. In the current BIG BOY config they may coincide, but this is a structural fragility.

**How to avoid:** Either (a) always compute amortized latency from the size-bits already present in `_joint_candidates_for_tensor` (using `ceil(size_bits[k] / min_bw)`) and separately iterate over `reuse_levels[(t, s)]` with its own `joint_w`, not assuming co-indexing; or (b) verify at plan time that these dims are always the same set for the SwiGLU use case and document the assumption.

**Warning signs:** `len(_joint_candidates_for_tensor(...))` != `len(reuse_levels[(t, s)])` at runtime.

### Pitfall 2: Incorrect `amortized_latency` when `sf_coeff == 0`
**What goes wrong:** `reuse_levels[(t, -1)]` at stop level `-1` (no reuse) has `sf_coeff = 1` in the scalar case (base reuse factor = 1). In variable mode the sf_coeff may also be 1 for stop=-1. However for higher stop levels, sf_coeff can be large integers (e.g., number of temporal iterations). Division by zero is impossible since sf_coeff >= 1 by construction, but large sf_coeff produces small amortized latency coefficients — ensure `math.ceil` is used consistently, or use `float` division with explicit ceil.

**How to avoid:** Use `ceil(size_bits[k] / min_bw) / sf_c` with Python's `math.ceil` on the numerator first (matching `_transfer_latency_for_path`), then divide by `sf_c`. Do not use `ceil(...)` on the ratio — amortized latency need not be integer; use float division.

### Pitfall 3: Scalar fallback path breaks when `_joint_candidates_for_tensor` returns `[]`
**What goes wrong:** When there are no tiled dims (scalar mode), `_joint_candidates_for_tensor` returns `[]` and `reuse_levels[(t, s)]` is a `(fires_c, sf_c)` int-tuple (not a list). The new `_active_transfer_latency` must detect this and use the scalar path.

**How to avoid:** Use the same `isinstance(rl_check, list)` guard already used in `_reuse_factor_rate_constraints` (line 831) to detect variable vs scalar mode. In scalar mode, either call the existing `_add_binary_times_const_over_linexpr` (which uses `addGenConstrNL`) — but only if `addGenConstrNL` is being kept — or implement an equivalent pure-MILP scalar path. Given D-03 (eliminate `addGenConstrNL`), the scalar path also needs re-implementation without the NL helper, or the single-candidate variable path handles it gracefully when there is exactly one joint candidate.

**Warning signs:** `isinstance(self.reuse_levels[(t, -1)], list)` is `False` in scalar mode — must not attempt to iterate over it as a list.

### Pitfall 4: Big-M bound too loose for the latency sum
**What goes wrong:** `_add_binary_scaled_continuous` requires an accurate `continuous_ub`. Using a loose upper bound is valid (the constraint remains correct) but produces weaker LP relaxation, potentially slowing the solver.

**How to avoid:** Set `continuous_ub = sum(max(amortized[k,s] for k) for s in stop_levels)` — the exact per-stop max summed over all stops. This is tight because at most one `z_stop[s]` is active.

### Pitfall 5: Cache key collision between old NL var and new MILP var
**What goes wrong:** `_transfer_latency_cache` stores `gp.Var` keyed by `(tr, choice)`. If the cache is populated during model build by one code path and consumed by another, stale entries return the wrong variable type.

**How to avoid:** The cache is only populated inside `_active_transfer_latency` itself (lines 1929-1941). The refactored method writes to the same cache key, so no collision. No restructuring needed.

## Code Examples

### Computing per-candidate latency numerator (inline in loop)
```python
# Source: transfer_and_tensor_allocation.py lines 243-249 (_transfer_latency_for_path)
# Instead of calling _transfer_latency_for_path(tr, choice) once,
# compute per candidate using size_bits already in _joint_candidates_for_tensor:
min_bw = min(link.bandwidth for link in choice.links_used)
for size_bits, joint_w in self._joint_candidates_for_tensor(tr.inputs[0], tr):
    latency_numerator_k = ceil(size_bits / min_bw)   # same formula, per-candidate
```

### Accessing sf_coeff from reuse_levels in variable mode
```python
# Source: transfer_and_tensor_allocation.py lines 838, 424-426
t = tr.inputs[0]
rl_check = self.reuse_levels[(t, -1)]
if isinstance(rl_check, list):
    # Variable tile mode
    for s in stop_levels:
        for fires_c, sf_c, jw in self.reuse_levels[(t, s)]:
            amortized = ceil(size_bits_for_this_jw / min_bw) / sf_c
            # ... accumulate coeff * jw
```

### Big-M gating of stop-level latency (mirrors _reuse_factor_rate_constraints)
```python
# Source: transfer_and_tensor_allocation.py lines 840-849
z = self.z_stop[(t, s)]
expr_s = quicksum(amortized[k, s] * joint_w_k for k, joint_w_k in ...)
M_s = max_amortized_for_stop_s
lc_s = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=M_s,
                          name=f"lat_lc_{tr.name}_L{s}")
self.model.addConstr(lc_s <= expr_s,            name=f"lat_lc_ub_expr_{tr.name}_L{s}")
self.model.addConstr(lc_s <= M_s * z,           name=f"lat_lc_ub_m_{tr.name}_L{s}")
self.model.addConstr(lc_s >= expr_s - M_s*(1-z),name=f"lat_lc_lb_{tr.name}_L{s}")
```

### Confirming _add_const_over_linexpr callers before removal
```python
# Verify no other callers exist before deleting:
# Grep for _add_const_over_linexpr and _add_binary_times_const_over_linexpr
# Expected: only _active_transfer_latency (line ~1934) and _add_binary_times_const_over_linexpr
```

### Degenerate single-candidate test pattern (from existing test)
```python
# Source: tests/unit/test_co_tile_variables.py lines 1395-1507
# Pattern: one jw constrained to 1, one z_stop constrained to 1.
# With amortized_latency[0, -1] = ceil(size_bits / min_bw) / sf_coeff,
# model should solve optimally and return that scalar value.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `addGenConstrNL` for `c / linexpr` | Pre-computed scalar coefficients × binary variables (MILP) | Phase 5 | Eliminates last nonlinear constraint; model becomes pure MILP |
| Single scalar `_transfer_latency_for_path(tr, choice)` | Per-candidate LUT `ceil(size_bits[k] / min_bw)` | Phase 5 | Tile-dependent latency; CO-03 satisfied |

**Deprecated/outdated after Phase 5:**
- `_add_const_over_linexpr`: Used only for latency division. Safe to delete if no other caller.
- `_add_binary_times_const_over_linexpr`: Wrapper that calls `_add_const_over_linexpr`. Also deletable.
- The incorrect type annotation `reuse_factor: gp.LinExpr` on line 1933 (it is a `gp.Var`) can be cleaned up as a byproduct.

## Open Questions

1. **Are tensor-tiled-dims and SSIS-tiled-dims always the same set for SwiGLU transfers?**
   - What we know: `_tiled_dims_for_tensor` uses `get_unique_dims_inter_core_tiling`; `_ssis_tiled_dims_for_transfer` uses `get_applicable_temporal_dimensions`. These can differ if a dim appears in inter-core tiling but not in SSIS temporal vars, or vice versa.
   - What's unclear: Whether any SwiGLU transfer nodes exhibit this difference.
   - Recommendation: At plan/task time, add an assertion or explicit check and document the assumption. If they differ, the plan must use separate enumeration loops for size-bits (from tensor candidates) and sf-coefficients (from reuse_levels), not assume co-indexing.

2. **Should the scalar fallback path also eliminate `addGenConstrNL`?**
   - What we know: D-03 says the NL constraint must be eliminated. In scalar mode (no tiled dims), the current code path still calls `_add_binary_times_const_over_linexpr` → `addGenConstrNL`. The question is whether this path is reached in the final configuration.
   - What's unclear: Whether the scalar path is exercised at all once variable tile mode is active everywhere (it would only be reached for transfers with no tiled dims).
   - Recommendation: Eliminate `addGenConstrNL` from all code paths, including scalar fallback. Implement scalar fallback as a single-candidate degenerate case of the variable path (one candidate, one joint_w that is effectively `1`), or implement direct scalar amortized latency as `amortized = ceil(size_bits / min_bw) / sf_coeff` with no linearization needed.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies — phase is pure code refactor of existing Gurobi model-building code).

## Validation Architecture

`workflow.nyquist_validation` is not set to false in `.planning/config.json` — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pytest.ini or pyproject.toml (project root) |
| Quick run command | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x -q` |
| Full suite command | `.venv/bin/pytest tests/unit/test_co_tile_variables.py tests/regression/test_baseline.py -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CO-03 | `_active_transfer_latency` returns a linear expression (no `addGenConstrNL`) | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k latency -x` | ❌ Wave 0 |
| CO-03 | Degenerate single-candidate: latency variable equals `ceil(size_bits / min_bw) / sf_coeff` | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k degenerate_latency -x` | ❌ Wave 0 |
| CO-03 | Multi-candidate: only the selected candidate's latency is active | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k latency_multi_candidate -x` | ❌ Wave 0 |
| CO-03 | Model does not contain `addGenConstrNL` after build | unit (inspect model) | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k no_nl_constraint -x` | ❌ Wave 0 |
| CO-03 | Regression: degenerate single-candidate model solves optimally | integration | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k degenerate_ssis -x` | ✅ existing (extend) |

### Sampling Rate
- **Per task commit:** `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x -q`
- **Per wave merge:** `.venv/bin/pytest tests/unit/test_co_tile_variables.py tests/regression/test_baseline.py -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_co_tile_variables.py` — add `test_active_transfer_latency_variable_mode`, `test_active_transfer_latency_degenerate`, `test_active_transfer_latency_no_nl`, `test_active_transfer_latency_multi_candidate` (extend existing file; file already exists)
- [ ] No new config or framework install required

## Sources

### Primary (HIGH confidence)
- Direct read of `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `_active_transfer_latency` (lines 1923-1943), `_add_const_over_linexpr` (lines 1665-1713), `_add_binary_times_const_over_linexpr` (lines 1756-1785), `_add_binary_scaled_continuous` (lines 1715-1754), `_add_binary_product` (lines 1787-1799), `_joint_candidates_for_tensor` (lines 1819-1885), `_ssis_coefficients_for_transfer` (lines 287-378), `_init_transfer_fire_helpers` (lines 392-438), `_reuse_factor_rate_constraints` (lines 819-860), `_transfer_latency_for_path` (lines 243-249), `_slot_latency_constraints` (lines 1254-1257)
- Direct read of `tests/unit/test_co_tile_variables.py` — stub pattern, degenerate test `test_degenerate_ssis_single_candidate_feasible` (lines 1395-1507)
- Direct read of `.planning/phases/05-variable-transfer-latency/05-CONTEXT.md` — all decisions D-01 through D-09

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md` — CO-03 requirement definition and traceability
- `.planning/STATE.md` — accumulated decisions from Phases 3-4

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all helpers already verified in codebase
- Architecture: HIGH — enumeration pattern directly readable from Phase 3-4 code; new formulation is a direct parallel
- Pitfalls: HIGH — derived from reading actual code and data structures; not inferred
- Open questions: MEDIUM — Pitfall 1 (dim-set alignment) requires runtime verification but the mitigation strategy is clear

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable codebase; only relevant change would be Phase 4 refactors not yet applied)
