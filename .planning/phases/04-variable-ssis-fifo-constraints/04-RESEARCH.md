# Phase 4: Variable SSIS + FIFO Constraints - Research

**Researched:** 2026-04-07
**Domain:** MILP linearization of tile-dependent loop sizes; SSIS class refactor; Gurobi big-M patterns
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Replace (not duplicate) the existing `SteadyStateIterationSpace` class with a tile-aware version. When a dimension has one candidate, K and T are scalar constants (backward compatible). When a dimension has multiple candidates, K and T are candidate-indexed coefficients used to build linear expressions in the CO model.
- **D-02:** Loop decomposition: K × S × T = workload_size per dimension. S is fixed. K = candidate tile size. T = workload_size / (S × K).
- **D-03:** Refactor `generate_steady_state_iteration_spaces` to produce tile-aware objects. Same entry point, richer output.
- **D-04:** `reuse_levels`, `tiles_needed_levels`, `bds_needed_levels`, and fire counts become linear expressions: `sum_k(coeff[k] * joint_binary_var[k])` with coefficients pre-computed per candidate combination.
- **D-05:** Interaction with z_stop uses continuous auxiliary variables with big-M activation — same three-constraint pattern as Phase 3's `_memory_capacity_constraints`. No triple product with `u` needed (simpler than memory case).
- **D-06:** Tight per-constraint big-M bounds: max over candidate combinations, computed as a byproduct of coefficient enumeration.
- **D-07:** `_ensure_same_ssis_for_all_transfers` moves to post-solve verification. Structural property holds by construction.
- **D-08:** Phase 4 adds SSIS loop-size utility functions (K, T, reuse_levels, fire counts, tiles_needed per candidate tile combination) — the TILE-05 utilities deferred from Phase 3.
- **D-09:** These utilities work with the tile-aware SSIS class to produce candidate-indexed coefficients used in linearization.

### Claude's Discretion

- Internal structure of the refactored SSIS class (method signatures, properties)
- How `_init_transfer_fire_helpers` is restructured to produce linear expressions
- Unit test design for tile-aware SSIS and linearized FIFO constraints
- Whether `_object_fifo_depth_constraints` and `_buffer_descriptor_constraints` share linearization code or remain separate

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CO-02 | SSIS loop sizes (kernel and temporal) become linear expressions over tile selection variables, updating reuse_levels, fire counts, and buffer depth constraints | Covered by D-04/D-05 linearization pattern; SSIS refactor (D-01/D-02/D-03); tile-size utility additions (D-08/D-09) |
| CO-04 | Object FIFO depth constraints use variable tile-dependent sizes | Covered by extending `_object_fifo_depth_constraints` and `_buffer_descriptor_constraints` to consume tile-indexed `tiles_needed_levels` and `bds_needed_levels` |
</phase_requirements>

---

## Summary

Phase 4 makes all SSIS-derived quantities — kernel loop count K, temporal loop count T, reuse levels, fire counts, tiles_needed, and bds_needed — tile-dependent linear expressions in the Gurobi CO model. The central insight is that the current SSIS encodes T in its temporal `IterationVariable.size` fields (via `fusion_splits`) and encodes K implicitly as workload_size/S in the kernel variable, with neither being a CO decision variable. Under variable tile sizes, both K and T must be expressed as weighted sums over the one-hot w[dim,k] binary variables.

The phase has three coordinated work areas: (1) extend `SteadyStateIterationSpace` to carry per-candidate K/T coefficient tables rather than fixed sizes; (2) refactor `_init_transfer_fire_helpers` in `TransferAndTensorAllocator` to produce candidate-indexed coefficient dictionaries rather than scalars; (3) extend `_object_fifo_depth_constraints` and `_buffer_descriptor_constraints` to build linear expressions from those coefficients, gated through z_stop with the big-M activation pattern already established in Phase 3. The degenerate single-candidate path must produce the same scalar results as today, ensuring regression compatibility without branching in callers.

The linearization is simpler than Phase 3's memory capacity constraint because there is no triple product u × z_stop × tile_expr — only z_stop × tile_expr, so no continuous auxiliary variable is needed for that gating in the FIFO depth case. However, `reuse_levels[fires]` and `reuse_levels[size_factor]` may vary across candidates and are multiplied by z_stop, so those still require the two-term linearization pattern (scalar coefficient times binary variable for each candidate).

**Primary recommendation:** Implement the tile-aware coefficient tables in a new helper method `_ssis_coefficients_for_transfer(tr)` that returns `{(tensor, stop): {"fires": [(coeff, jw)], "tiles_needed": [(coeff, jw)], "bds_needed": [(coeff, jw)]}}`, then consume these tables in `_init_transfer_fire_helpers`, `_object_fifo_depth_constraints`, and `_buffer_descriptor_constraints`. This keeps the three consumers decoupled from the SSIS class and concentrates the combinatorial enumeration in one place.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gurobipy | project-installed | MILP variable/constraint creation | Only solver in the project; Phase 3 pattern established |
| stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation | project | Primary modification target | All CO constraint methods live here |
| stream.workload.steady_state.iteration_space | project | SSIS class and IterationVariable — tile-aware extension target | Primary data structure |
| stream.workload.utils | project | `generate_steady_state_iteration_spaces`, `determine_fusion_splits`, `collect_spatial_unrollings` | SSIS construction pipeline |
| stream.opt.tile_size_utils | project | Phase 4 adds K/T utility functions here alongside existing tensor-size utilities | Convention established in Phase 3 |

### No External Dependencies

Phase 4 is a pure code/MILP refactor. No new packages. No external services.

---

## Architecture Patterns

### Current SSIS Loop Structure (Before Phase 4)

```
SteadyStateIterationSpace.variables = [
  IterationVariable(dim, size=workload_size/S, type=KERNEL),   # per unique_dim
  IterationVariable(dim, size=S, type=SPATIAL),                 # per spatial unrolling
  IterationVariable(dim, size=T=workload_size/(S*K), type=TEMPORAL), # per fusion_split dim
]
```

- **Kernel var size**: `workload_size / S` — this is K*T (NOT just K). The kernel variable does NOT equal the candidate tile size.
- **Temporal var size**: `T = workload_size / (S * tile_size)` — stored as a fixed int in `determine_fusion_splits`.
- Under variable tiles, **T varies per candidate**: `T_k = workload_size / (S * tile_k)`.

### Target: Tile-Aware Coefficient Tables (Phase 4)

Instead of storing fixed `size` ints in `IterationVariable`, Phase 4 introduces per-dimension candidate coefficient vectors. Two design options:

**Option A (Recommended): External coefficient table, minimal SSIS change.**
`SteadyStateIterationSpace` gains a method `candidate_loop_sizes(dim)` that returns `{k: (K_k, T_k)}` where `K_k = tile_k`, `T_k = workload_size / (S * tile_k)`. The SSIS `variables` keep their current scalar `size` (matching the selected/only candidate for the degenerate case). A new helper on `TransferAndTensorAllocator` pre-computes `reuse_level_coeffs[(t, stop, k)] = (fires_k, size_factor_k)` and `tiles_needed_coeffs[(t, stop, k)] = tiles_needed_k` for all candidates.

**Option B: Embed coefficient lists in IterationVariable.**
`IterationVariable.size` becomes a `list[int]` when multiple candidates exist. Risky: the existing `size: int` contract is used by many callsites (`slices_per_full`, `reuse_factor`, `shape_mem`, etc.).

Recommendation: Option A. Zero breakage outside the CO allocator.

### Pattern 1: Reuse Level Linearization (D-04)

For each `(t, stop)` pair, the current scalar `(fires, size_factor)` becomes a linear expression:

```python
# Current (scalar):
fires_scalar = prod(sizes)
...
# Phase 4 (variable tiles):
# Pre-compute per candidate k:
fires_coeffs = {}
for k, tile_k in enumerate(candidates[dim]):
    T_k = workload_size // (S * tile_k)
    # Rebuild the sizes list with T_k substituted for the tiled dim
    sizes_k = [...updated sizes...]
    fires_k = prod(sizes_k)  # integer
    size_factor_k = ...     # integer
    fires_coeffs[(t, stop, k)] = fires_k
    size_factor_coeffs[(t, stop, k)] = size_factor_k

# In CO model:
fires_expr = quicksum(fires_coeffs[(t, stop, k)] * w[(dim, k)] for k in range(n_candidates))
```

For multiple tiled dimensions: use `_joint_candidates_for_tensor`-style enumeration with `joint_binary_var` (product of w[dim,k] for each dim).

### Pattern 2: FIFO Depth Linearization (D-05, simpler than memory)

```python
# Current (scalar):
self.object_fifo_depth[c] += tiles_needed * uz  # uz = u AND z_stop binary

# Phase 4 (variable tiles):
# tiles_needed is now a linear expression over joint binaries:
tiles_needed_expr = quicksum(
    tiles_needed_k * jw_k
    for tiles_needed_k, jw_k in tiles_needed_joint_candidates[(t, stop)]
)
# uz is still a plain binary (u AND z_stop)
# tiles_needed_expr * uz is a product of LinExpr × binary — requires auxiliary

# Linearize via big-M activation (lc pattern):
M = max_tiles_needed + (1 if force_double_buffering else 0)
lc = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=M, name=f"fifo_lc_{t.name}_{c}_{stop}")
model.addConstr(lc <= tiles_needed_expr + int(force_double_buffering))
model.addConstr(lc <= M * uz)
model.addConstr(lc >= tiles_needed_expr + int(force_double_buffering) - M * (1 - uz))
self.object_fifo_depth[c] += lc
```

Note: `force_double_buffering` adds a fixed integer to the expression, not a variable — it does not affect the linearization structure.

### Pattern 3: Big-M Bound Computation (D-06)

```python
# Tight M = max tiles_needed over all joint candidate combinations
M_tiles_needed = max(tiles_needed_k for tiles_needed_k, _ in tiles_needed_joint_candidates[(t, stop)])
M_tiles_needed += 1 if force_double_buffering else 0
```

This mirrors `self._tensor_max_size[t]` used in `_memory_capacity_constraints`.

### Pattern 4: Degenerate Single-Candidate Path (Regression Safety)

When `len(candidates[dim]) == 1` for all tiled dims, `joint_candidates` has exactly one element `(coeff, w[(dim,0)])`. Since `w[(dim,0)] == 1` by the one-hot constraint, `quicksum(coeff * jw) == coeff * 1 == coeff`. The linear expression evaluates to the same scalar as the current code. No special branch needed — the math handles it.

This is verified by the existing `test_single_candidate_regression_compat` test in `tests/unit/test_co_tile_variables.py`.

### Pattern 5: Determining Tiled Dimensions for SSIS

Not every SSIS dimension is variable. Only dimensions present in `search_space.dims()` need candidate-indexed coefficients. Other dimensions contribute fixed size factors. The enumeration must:

1. Identify which SSIS temporal dimensions correspond to search_space dims.
2. Enumerate joint candidates across tiled dims using `iproduct`.
3. For non-tiled dims, use the fixed temporal size (from the single-candidate SSIS).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Binary AND of multiple variables | Custom constraint chain | `_add_binary_product(a, b)` + fold | Already implemented; tested |
| Joint candidate enumeration | Custom nested loops | `_joint_candidates_for_tensor` pattern with `iproduct` | Handles arbitrary dimensionality |
| Linear product of binary × LinExpr | NonlinearConstr | big-M three-constraint pattern (lc pattern) | Gurobi MIP handles LPs efficiently; NL constraints degrade to MIP anyway |
| Spatial unrolling lookup | Scan workload manually | `collect_spatial_unrollings(workload, mapping)` | Returns both per-node and unique unrollings |

---

## Common Pitfalls

### Pitfall 1: Kernel Variable Size Is NOT K

**What goes wrong:** Developer reads `IterationVariable(type=KERNEL).size` expecting the tile size K, but finds `workload_size / S` instead (the total kernel extent = K*T).

**Why it happens:** In `_insert_kernel_iteration_variables`, `size = workload_size / spatial_unrolling`. This represents the per-core kernel range, not the candidate tile K.

**How to avoid:** Compute K from `TileSizeOption.tile` in the search space; compute T = `workload_size / (S * K)`. The existing KERNEL variable in SSIS should be treated as the "kernel extent" column for the loop structure, not as the decision variable K.

**Warning signs:** Test shows T values that don't match `workload_size / (S * tile_k)`.

### Pitfall 2: reuse_levels Dict Type Changes

**What goes wrong:** `self.reuse_levels[(t, stop)]` currently returns `(int, int)`. After Phase 4, for variable-tile transfers it returns per-candidate coefficient lists. Any code that does `fires, size_factor = self.reuse_levels[(t, stop)]` and treats them as plain ints will break.

**Why it happens:** `_transfer_fire_rate_constraints` and `_reuse_factor_rate_constraints` both index `reuse_levels[(t, s)][0]` and `[1]` as ints.

**How to avoid:** Either (a) change the dict type to hold linear expressions and update both consumers, or (b) maintain two separate dicts — scalar and variable-tile — and select based on `use_variable_tiles`. The scalar path must remain for non-tiled transfers.

**Warning signs:** `TypeError: 'LinExpr' object cannot be interpreted as an integer` at constraint construction time.

### Pitfall 3: _ensure_same_ssis_for_all_transfers Called at Init Time

**What goes wrong:** Line 148 of the allocator `__init__` calls `self._ensure_same_ssis_for_all_transfers()` before the CO model is built. Under variable tiles, SSIS objects carry candidate coefficients, not resolved sizes — so the size comparison fails spuriously.

**Why it happens:** The check compares `prod(temporal_sizes)` across transfers. With candidate-indexed sizes, there is no single scalar product.

**How to avoid:** Per D-07, remove the `__init__`-time call and move the validation to a post-solve method that compares concrete sizes after `self.model.optimize()` resolves w[dim,k]. The structural invariant (K × S × T = workload_size for all k) is enforced by construction, so the pre-solve check is redundant.

**Warning signs:** Assertion error at allocator construction time referencing SSIS size mismatch.

### Pitfall 4: z_stop Variable Count Depends on get_applicable_temporal_variables

**What goes wrong:** `__create_reuse_vars` iterates `range(-1, len(sizes))` where `sizes = ssis[tr].get_applicable_temporal_sizes()`. If Phase 4 changes the number of temporal variables in SSIS, the z_stop count changes and all downstream dicts `reuse_levels`, `tiles_needed_levels`, etc. must match.

**Why it happens:** The stop-level range is computed once at variable creation and must stay consistent throughout model construction.

**How to avoid:** Keep the temporal variable count invariant under Phase 4. The SSIS should produce the same number of temporal variables regardless of tile selection — only the `size` values (or coefficient tables) change per candidate.

**Warning signs:** KeyError on `(t, stop)` lookups in `reuse_levels` or `z_stop`.

### Pitfall 5: Double-Buffering Offset in FIFO Constraints

**What goes wrong:** `tiles_needed += 1 if self.force_double_buffering else 0` (line 855) applies an integer increment to a scalar. Under variable tiles, if tiles_needed becomes a linear expression, the increment must be applied to the expression, not the coefficient.

**Why it happens:** Scalar int + 1 vs. LinExpr + 1. The latter is valid in Gurobi (`LinExpr + constant`), but the big-M bound must account for it.

**How to avoid:** Apply the increment after building `tiles_needed_expr`: `tiles_expr_with_db = tiles_needed_expr + (1 if self.force_double_buffering else 0)`. Update M to include the same offset: `M += 1 if self.force_double_buffering else 0`.

### Pitfall 6: Joint Candidates for SSIS May Differ from Joint Candidates for Tensor Size

**What goes wrong:** `_joint_candidates_for_tensor` filters to dims that appear in the successor's inter-core tiling. SSIS coefficients are indexed by ALL search_space dims (since T depends on the tile regardless of whether the tensor shape changes). If the same enumeration helper is reused for SSIS coefficients, it may silently exclude a dimension.

**Why it happens:** The two computations answer different questions: tensor size depends on inter-core tiling structure; SSIS loop counts depend on which dims have variable tile sizes in `intra_core_tiling`.

**How to avoid:** For SSIS coefficient enumeration, iterate over all dims in `search_space.dims()` that correspond to temporal loop dimensions in the SSIS, independent of whether they appear in the successor's tiling. Build a separate helper `_ssis_tiled_dims_for_transfer(tr)` that uses `ssis[tr].get_applicable_temporal_dimensions()` intersected with `search_space.dims()`.

---

## Code Examples

### Computing K and T per Candidate

```python
# Source: stream/workload/utils.py — determine_fusion_splits logic
# S = spatial unrolling (from collect_spatial_unrollings)
# K_k = candidate tile size
# T_k = workload_size / (S * K_k)

def ssis_loop_sizes_for_candidate(
    dim: LayerDim,
    candidate_tile: int,
    workload: Workload,
    mapping: Mapping,
) -> tuple[int, int]:
    """Return (K, T) for candidate tile on dim.

    K = candidate_tile
    T = workload_size / (S * K)
    """
    _, unique_spatial_unrollings = collect_spatial_unrollings(workload, mapping)
    S = dict(unique_spatial_unrollings).get(dim, 1)
    workload_size = workload.get_dimension_size(dim)
    K = candidate_tile
    T, rem = divmod(workload_size, S * K)
    assert rem == 0, f"workload_size {workload_size} not divisible by S*K = {S*K}"
    return K, T
```

### Computing reuse_levels Coefficients per Candidate

```python
# Mirrors _init_transfer_fire_helpers but parameterized by candidate T_k values

def _compute_reuse_coefficients(
    sizes: list[int],          # per-stop sizes from SSIS (candidate-resolved)
    relevancies: list[bool],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return fires_per_stop and size_factor_per_stop dicts for one candidate."""
    fires_out = {}
    size_factor_out = {}
    fires = math.prod(sizes)
    size_factor = 1
    tiles_needed = 1
    fires_out[-1] = fires
    size_factor_out[-1] = size_factor
    for i, (Nl, relevancy) in enumerate(zip(sizes, relevancies)):
        size_factor *= Nl if relevancy else 1
        tiles_needed *= Nl if relevancy else 1
        fires //= Nl
        fires_out[i] = fires
        size_factor_out[i] = size_factor
    return fires_out, size_factor_out
```

### big-M Activation for tiles_needed × z_stop (FIFO Depth)

```python
# Source: Adapted from _memory_capacity_constraints big-M pattern (lines 801-831)
# Simpler: no u (tensor-uses-core) binary — only z_stop gating

for tiles_needed_k, jw_k in tiles_needed_joint_candidates[(t, stop)]:
    tiles_needed_expr = quicksum(tnk * jw for tnk, jw in tiles_needed_joint_candidates[(t, stop)])
    db_offset = 1 if self.force_double_buffering else 0
    M = max(tnk for tnk, _ in tiles_needed_joint_candidates[(t, stop)]) + db_offset

    lc = self.model.addVar(
        vtype=GRB.CONTINUOUS, lb=0, ub=M,
        name=f"fifo_lc_{t.name}_{_resource_key(c)}_L{stop}"
    )
    z = self.z_stop[(t, stop)]
    self.model.addConstr(
        lc <= tiles_needed_expr + db_offset,
        name=f"fifo_lc_ub_expr_{t.name}_{_resource_key(c)}_L{stop}"
    )
    self.model.addConstr(
        lc <= M * z,
        name=f"fifo_lc_ub_m_{t.name}_{_resource_key(c)}_L{stop}"
    )
    self.model.addConstr(
        lc >= tiles_needed_expr + db_offset - M * (1 - z),
        name=f"fifo_lc_lb_{t.name}_{_resource_key(c)}_L{stop}"
    )
    self.object_fifo_depth[c] += lc
```

### Test Stub Pattern for New Methods

```python
# Follow the established pattern from test_co_tile_variables.py:
# types.SimpleNamespace + __get__ binding, no real allocator construction

stub._init_transfer_fire_helpers = (
    TransferAndTensorAllocator._init_transfer_fire_helpers.__get__(stub)
)
# Provide: stub.transfer_nodes, stub.ssis, stub.search_space, stub.w,
#          stub.model, stub.transfer_nodes_to_optimize_firings_for,
#          stub.reuse_levels, stub.tiles_needed_levels, stub.bds_needed_levels
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Scalar `size` in IterationVariable | Candidate-indexed coefficient tables for tiled dims | Phase 4 | `_init_transfer_fire_helpers` produces linear expressions instead of scalars |
| `_ensure_same_ssis_for_all_transfers` called at init | Moved to post-solve verification | Phase 4 | Allocator construction no longer fails when tile sizes are unresolved |
| `tiles_needed * uz` scalar multiply in FIFO constraints | `lc` continuous auxiliary variable gated by `uz` binary | Phase 4 | Correct linearization of tile-dependent FIFO depth |

---

## Open Questions

1. **Does `_force_nonconstant_reuse_levels` need adaptation?**
   - What we know: It reads `ssis[tr].get_applicable_temporal_relevancies()` (booleans, not sizes). Under Phase 4, relevancies don't change — only sizes do.
   - What's unclear: Whether `z_stop` stop indices are affected if SSIS temporal variable count changes.
   - Recommendation: Keep unchanged if SSIS temporal variable count is invariant. Verify in Wave 0 test.

2. **Are bds_needed coefficients always integers?**
   - What we know: `bds_needed` currently accumulates `Nl` (integer temporal loop sizes). With variable T_k, these remain integers per candidate.
   - What's unclear: Whether the Gurobi model needs INTEGER variables for lc vars or CONTINUOUS is sufficient.
   - Recommendation: Use CONTINUOUS — the big-M pattern with integer coefficients produces integer solutions when the binary variables are resolved.

3. **Variable count in BIG BOY config.**
   - What we know: STATE.md flags "Number of distinct (t, stop, k) triples in BIG BOY config not quantified — variable count may be large."
   - What's unclear: How many additional continuous variables (lc per FIFO constraint) will be created.
   - Recommendation: Enumerate combinatorially in a pre-planning script or Wave 0 test. The degenerate-input regression test (single candidate) is fast; BIG BOY impact is unknown until Phase 6.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 4 is a pure code refactor with no external tool dependencies beyond Gurobi (already installed and verified in Phase 3).

---

## Validation Architecture

`workflow.nyquist_validation` is absent from `.planning/config.json` — treated as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (confirmed in `.venv`) |
| Config file | `tests/unit/conftest.py` (shared fixtures) |
| Quick run command | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -q` |
| Full suite command | `.venv/bin/pytest tests/unit/ -q` |

Existing baseline: 18 tests pass in `test_co_tile_variables.py`.

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CO-02 | `_init_transfer_fire_helpers` produces candidate-indexed linear expressions for `reuse_levels` and `tiles_needed_levels` when search_space is set | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k "ssis" -x` | Wave 0 gap |
| CO-02 | Degenerate single-candidate path produces same scalar values as current code | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k "degenerate" -x` | Partial (CO-01 degenerate test exists; new SSIS test needed) |
| CO-02 | `SteadyStateIterationSpace` tile-aware extension: `ssis_loop_sizes_for_candidate` returns correct K, T | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py -k "ssis" -x` | Wave 0 gap |
| CO-04 | `_object_fifo_depth_constraints` uses lc auxiliary vars when tiles are variable | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k "fifo" -x` | Wave 0 gap |
| CO-04 | `_buffer_descriptor_constraints` uses lc auxiliary vars when tiles are variable | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k "bddepth" -x` | Wave 0 gap |
| CO-02/CO-04 | Regression: single-candidate degenerate input passes end-to-end | unit | `.venv/bin/pytest tests/unit/ -q` | Partial (unit suite passes; integration test is Wave 0 gap) |

### Sampling Rate

- **Per task commit:** `.venv/bin/pytest tests/unit/test_co_tile_variables.py -q`
- **Per wave merge:** `.venv/bin/pytest tests/unit/ -q`
- **Phase gate:** Full unit suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] Tests for `ssis_loop_sizes_for_candidate(dim, tile, workload, mapping)` in `tests/unit/test_tile_size_utils.py`
- [ ] Tests for tile-aware `_init_transfer_fire_helpers` producing linear expressions in `tests/unit/test_co_tile_variables.py`
- [ ] Tests for `_object_fifo_depth_constraints` lc variable creation in `tests/unit/test_co_tile_variables.py`
- [ ] Tests for `_buffer_descriptor_constraints` lc variable creation in `tests/unit/test_co_tile_variables.py`
- [ ] Test verifying degenerate (1-candidate) scalar equivalence for SSIS-derived quantities

---

## Sources

### Primary (HIGH confidence)

- Direct code read: `stream/workload/steady_state/iteration_space.py` — full `SteadyStateIterationSpace` class
- Direct code read: `stream/workload/utils.py` — `generate_steady_state_iteration_spaces`, `_insert_kernel_iteration_variables`, `_add_temporal_iteration_variables`, `determine_fusion_splits`
- Direct code read: `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `_init_transfer_fire_helpers` (line 259), `_memory_capacity_constraints` (line 758), `_object_fifo_depth_constraints` (line 842), `_buffer_descriptor_constraints` (line 865), `_joint_candidates_for_tensor` (line 1502), `_joint_binary_for_combo` (line 1578), `_add_binary_product` (line 1470), `__create_tile_selection_vars` (line 567)
- Direct code read: `stream/opt/tile_size_utils.py` — existing utilities
- Direct code read: `stream/opt/search_space.py` — `SearchSpace`, `TileSizeOption`
- Direct code read: `tests/unit/test_co_tile_variables.py` — stub pattern, existing 18 passing tests
- Test run: 18 unit tests pass (2026-04-07)

### Secondary (MEDIUM confidence)

- `.planning/phases/04-variable-ssis-fifo-constraints/04-CONTEXT.md` — locked decisions D-01 through D-09
- `.planning/REQUIREMENTS.md` — CO-02, CO-04 requirement definitions
- `.planning/STATE.md` — known concerns about variable count in BIG BOY config

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all files read directly; no external dependencies
- Architecture patterns: HIGH — derived from reading actual implementation of Phase 3's big-M pattern and SSIS construction pipeline
- Pitfalls: HIGH — derived from concrete code analysis (line numbers cited); one pitfall (variable count) remains LOW — flagged in Open Questions

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable internal codebase; no external libraries change)
