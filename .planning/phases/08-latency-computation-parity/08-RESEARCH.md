# Phase 8: Latency Computation Parity - Research

**Researched:** 2026-04-08
**Domain:** MILP constraint optimization — slot latency linearization, MACs formula correctness, iteration scaling
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** The fix is localized to `_slot_latency_constraints` in `transfer_and_tensor_allocation.py`. The method must compute inter_core_tiling factors correctly for the untiled workload (since Phase 7 moved TilingGenerationStage post-solve). The tiling factor should be derived as `workload_size / (tile * spatial)`, not read from a pre-tiled mapping that no longer exists at CO time.
- **D-02:** `fusion_splits` in a fusion group are actually the temporal T loop sizes (`workload_size / (tile * spatial)`). They must be derived correctly from tile candidates, not stored as fixed values. If currently defined as fixed, the mapping representation needs updating so the CO can compute them from variable tile sizes.
- **D-03:** MACs are computed from first principles: `MACs = prod(node_dimension_sizes) / tiling_factor`, where `tiling_factor = prod(spatial * tile)` per tiled dimension. Any mismatch with the old path indicates a bug in understanding, not in code. No attempt to replicate old-path quirks.
- **D-04:** Validate using hardcoded baseline values: assert `latency_total=922357343` and `latency_per_iteration=10716` for the single-candidate BIG BOY config. Matches success criteria directly.
- **D-05:** Add temporary per-node MACs/latency logging to both the estimator and `_slot_latency_constraints`. Run the single-candidate case and compare node-by-node. Remove logging after fix is confirmed.

### Claude's Discretion

- How to represent variable fusion_splits in the mapping (D-02 implementation detail)
- Whether to refactor `get_unique_dims_inter_core_tiling` or bypass it with a local computation
- Internal organization of the fix (single plan or multiple)
- Whether temporary logging is committed or kept as debug-only

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LAT-01 | TileAwareLatencyEstimator.estimate() produces the same MACs and cycle counts as the old AIECostEstimator + CoreCostLUT for identical tile sizes and workload dimensions | Root cause identified: missing iteration-scale correction in `_slot_latency_constraints`; TileAwareLatencyEstimator formula itself is correct |
| LAT-02 | _slot_latency_constraints correctly accounts for temporal splits (fusion_splits) without double-counting when the CO runs on the untiled workload | Root cause identified: variable tile path substitutes `workload_size // tile` but does not account for fixed `self.iterations` being tied to `tile_options[0]`; per-candidate coefficients need scaling by `base_tile / candidate_tile` |
| LAT-03 | The single-candidate degenerate case reproduces the original Phase 1 baseline exactly: latency_total=922357343, latency_per_iteration=10716 | Degenerate case (single candidate): `base_tile / candidate_tile = 1`, so scaling is identity — no regression expected once LAT-02 is fixed |
</phase_requirements>

---

## Summary

Phase 8 fixes one localized bug in `_slot_latency_constraints` and verifies end-to-end parity with the Phase 1 baseline. The bug was introduced by the Phase 7 pipeline reorder: TilingGenerationStage now runs AFTER the CO, so `self.iterations` is fixed at model construction time using `tile_options[0]` as the reference tile. When `_slot_latency_constraints` enumerates candidate tiles (e.g., tile=32 vs baseline=16), it correctly computes `raw_latency` for each candidate via `TileAwareLatencyEstimator.estimate()`. However, the CO objective is `total_latency = iterations * sum(slot_latency) - (iterations-1) * overlap`, where `iterations` is the fixed baseline count. If tile=32 is selected, the true number of iterations would be halved — but `self.iterations` stays at the baseline value. This inflates `total_latency` for larger tiles.

The fix: when building the `lat_coeffs` quicksum for each candidate tile combo, scale each coefficient by `prod(base_tile[dim] / candidate_tile[dim])` over all tiled dimensions. This correction factor converts `raw_latency_per_iteration[candidate]` into `effective_latency_when_using_fixed_iteration_count`. For the single-candidate degenerate case, `base_tile == candidate_tile` so scale = 1, leaving the baseline untouched.

The test stub at line 1931–1943 in `test_co_tile_variables.py` already scaffolds the needed attributes (`_base_orig_dim_sizes`, `_iter_scale_by_jw`), and the test at line 2003–2007 already asserts the scaled values `[4, 10]`. The production `_slot_latency_constraints` simply needs to compute and apply the scale factor — the gap between stub and production is the only code change required for LAT-01 and LAT-02.

**Primary recommendation:** Add iteration-scale correction to `_slot_latency_constraints`'s variable tile path: `scaled_lat = raw_lat * prod(base_tile[d] / cand_tile[d])` for each candidate combo. The regression test (`test_baseline_regression_latency`) validates LAT-03 without code changes once LAT-02 is fixed.

---

## Standard Stack

### Core (already present — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gurobipy | installed | MILP constraint model, `quicksum`, `addConstr` | Project solver throughout all phases |
| Python math | stdlib | `ceil`, `floor`, `prod` for MACs formula | Used in `TileAwareLatencyEstimator` already |
| pytest | installed | Unit + regression test runner | Established project test framework |

**Installation:** No new packages required for this phase.

---

## Architecture Patterns

### The Slot Latency Constraint (Variable Tile Mode)

Current code (`_slot_latency_constraints`, lines 1252–1281):

```python
# Variable tile mode
inter_core_tiling = self.workload.get_unique_dims_inter_core_tiling(n, self.mapping)
# ...
for combo in iproduct(*option_lists):
    # compute new_factor = workload_size // opt.tile  (replaces spatial factor)
    lat_est = self.latency_estimator.estimate(n, core, tuple(current_tiling))
    jw = self._joint_binary_for_combo(per_dim_options, combo, ...)
    lat_coeffs.append((lat_est.latency_total, jw))  # BUG: raw latency, no iteration scale
expr = quicksum(lat * jw for lat, jw in lat_coeffs)
```

### Fixed Version Pattern

```python
# Source: codebase analysis + test stub scaffold (test_co_tile_variables.py:1938-1943)
for combo in iproduct(*option_lists):
    # Build current_tiling (unchanged)
    lat_est = self.latency_estimator.estimate(n, core, tuple(current_tiling))
    # Compute iteration scale: base_tile / candidate_tile per tiled dim
    scale = prod(
        self._base_orig_dim_sizes[dim] / opt.tile
        for (dim, _opts), opt in zip(per_dim_options, combo, strict=False)
        if dim in self._base_orig_dim_sizes
    )
    jw = self._joint_binary_for_combo(per_dim_options, combo, ...)
    scaled_lat = round(lat_est.latency_total * scale)
    lat_coeffs.append((scaled_lat, jw))
    self._iter_scale_by_jw[jw] = scale  # optional: cache for debugging
```

### The `_base_orig_dim_sizes` Attribute

The test stub already defines this at line 1938–1941:

```python
# Source: tests/unit/test_co_tile_variables.py lines 1938-1941
if search_space is not None and not search_space.is_empty():
    stub._base_orig_dim_sizes = {d: search_space.get(d)[0].tile for d in search_space.dims()}
else:
    stub._base_orig_dim_sizes = {}
```

This must be added to the real `TransferAndTensorAllocator.__init__` as:

```python
# In __init__, after self.search_space is set
if self.search_space is not None and not self.search_space.is_empty():
    self._base_orig_dim_sizes = {d: self.search_space.get(d)[0].tile for d in self.search_space.dims()}
else:
    self._base_orig_dim_sizes = {}
self._iter_scale_by_jw: dict = {}
```

### MACs Formula (Already Correct)

The `TileAwareLatencyEstimator.estimate()` formula is correct and does NOT need changes:

```python
# Source: stream/cost_model/tile_aware_latency.py lines 77-90
tiling_factor = prod(f for _, f in inter_core_tiling) if inter_core_tiling else 1
macs = prod(dim_sizes) // tiling_factor
ideal_ops = self._aie.ops_per_cycle(node, core)
ideal_cycles = ceil(macs / ideal_ops)
ops_per_cycle = floor(ideal_ops * utilization / 100.0)
cycles = ceil(macs / ops_per_cycle)
```

The `inter_core_tiling` passed in already contains the correct total tiling factor (spatial splits from the mapping; variable path substitutes `workload_size // opt.tile` to replace the spatial factor with the full tile factor). The formula is first-principles correct.

### The Objective (Unchanged)

```python
# Source: transfer_and_tensor_allocation.py lines 1564-1566
self.model.addConstr(
    self.total_lat
    == self.iterations * quicksum(self.slot_latency.values()) - (self.iterations - 1) * self.overlap
)
```

`self.iterations` is fixed at construction (line 87) from `calculate_iterations()` which calls `prod(ssis.get_temporal_sizes())` using `tile_options[0]` as base. This is the root of the parity issue: when a larger tile is selected, `slot_latency` must be proportionally smaller to keep the product `iterations * slot_latency` equal to the true total cycles.

### Anti-Patterns to Avoid

- **Do not change TileAwareLatencyEstimator.estimate()**: The formula is correct. The bug is in the caller, not the estimator.
- **Do not make self.iterations a Gurobi variable**: This would make the objective nonlinear. The fix is to scale `slot_latency` coefficients, not to make `iterations` variable.
- **Do not use floating-point lat coefficients in the MILP**: Use `round()` when converting scaled latency to int. Gurobi INTEGER variables require integer RHS.
- **Do not remove `_iter_scale_by_jw` from the stub**: The stub already has this attribute set to `{}` — the production code must also initialize it (even if unused in constraints, it aids debugging per D-05).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Joint binary AND across dimensions | Custom binary product logic | `_joint_binary_for_combo` (already in place) | Established Phase 3/4 pattern; handles arbitrary dimension counts |
| Per-candidate quicksum | Loop with manual coefficient accumulation | `quicksum(lat * jw for lat, jw in lat_coeffs)` | Gurobi's quicksum is numerically stable and already used |
| Baseline tile lookup | Re-parse mapping YAML | `self._base_orig_dim_sizes` from search_space.get(d)[0].tile | TileSizeOption[0] is always the baseline tile by project invariant (Phase 7 D-08) |

---

## Common Pitfalls

### Pitfall 1: Floating-Point Latency Coefficients in MILP

**What goes wrong:** `scaled_lat = raw_lat * (base/candidate)` is a float. If passed to `quicksum` without rounding, Gurobi may raise a type error or silently truncate.

**Why it happens:** `base_tile / candidate_tile` is not always an integer ratio.

**How to avoid:** Use `int(round(scaled_lat))` before appending to `lat_coeffs`. For the degenerate case (single candidate), `base/candidate = 1.0`, so rounding is a no-op.

**Warning signs:** `AttributeError: 'float' object has no attribute 'X'` or Gurobi constraint type errors.

### Pitfall 2: `_base_orig_dim_sizes` Not Initialized in Production Code

**What goes wrong:** The test stub initializes `_base_orig_dim_sizes` but the real `TransferAndTensorAllocator.__init__` does not. When `_slot_latency_constraints` references `self._base_orig_dim_sizes`, an `AttributeError` is raised.

**Why it happens:** Phase 7 added the scaffold to the test stub in anticipation; the production `__init__` was not updated.

**How to avoid:** Add initialization in `__init__` immediately after `self.search_space = search_space`. Verified: grep confirms `_base_orig_dim_sizes` appears only in test stub, not in production code.

**Warning signs:** `AttributeError: 'TransferAndTensorAllocator' object has no attribute '_base_orig_dim_sizes'`

### Pitfall 3: Scale Factor Applied to No-Tiled-Dims Fallback

**What goes wrong:** The `else` branch (no tiled dims for this node) uses `latency_estimator` with fixed tiling and appends `[(lat_est.latency_total, None)]`. If iteration scaling is accidentally applied here, nodes not in the search space get wrong latency.

**Why it happens:** Copy-paste error when adding scaling logic.

**How to avoid:** Scaling logic applies ONLY to the `if tiled_dims:` branch. The `else` branch is for nodes whose dimensions are not tiled — no iteration count change applies.

### Pitfall 4: Test Assertion Mismatch After Fix

**What goes wrong:** The test `test_slot_latency_variable_mode` currently FAILS (1 failure in 104 tests). It expects `lats == [4, 10]` but gets `[8, 10]`. After fixing, it should pass. If the implementation applies the wrong scale factor, the test will fail with a different pair of values.

**Why it happens:** `base_tile=16, candidate_tile=32, scale=16/32=0.5`, so `8 * 0.5 = 4`.

**Warning signs:** Any result other than `[4, 10]` after the fix indicates the scale computation is wrong.

### Pitfall 5: Regression Test Requires Gurobi and Full Pipeline

**What goes wrong:** The regression test `test_baseline_regression_latency` runs the full BIG BOY CO pipeline with Gurobi, taking ~70 seconds. If run without `pytest -m slow`, it is skipped and provides no validation.

**Why it happens:** Tests are marked `@pytest.mark.slow` and require `gurobi` license.

**How to avoid:** Run regression explicitly: `.venv/bin/pytest tests/regression/test_baseline.py -m slow -x`. Unit tests (< 1s) validate the fix logic; regression confirms end-to-end.

---

## Code Examples

### Current Bug: test_slot_latency_variable_mode Expected vs. Actual

```python
# Source: tests/unit/test_co_tile_variables.py lines 2003-2007
# Currently FAILING (1 failure in 104 tests):
#   AssertionError: Expected latencies [4, 10], got [8, 10]
lats = sorted(lat for lat, _ in lat_coeffs)
assert lats == [4, 10], f"Expected latencies [4, 10], got {lats}"
# base_tile=16, candidate=16: scale=16/16=1.0, raw=10 -> scaled=10  CORRECT
# base_tile=16, candidate=32: scale=16/32=0.5, raw=8  -> scaled=4   MISSING
```

### How the Scaling Fits the Objective

```
total_latency = iterations * slot_latency - (iterations - 1) * overlap

True cycles for candidate tile=32 (double the tile):
  iterations_true = iterations_base / 2
  slot_latency_true = raw_latency_32   (unchanged per-iteration compute)

In CO model with fixed iterations_base:
  iterations * slot_latency = iterations_base * slot_latency_co
  Must equal: iterations_true * raw_latency_32 = (iterations_base/2) * raw_latency_32

Therefore: slot_latency_co[tile=32] = raw_latency_32 * (base_tile / candidate_tile)
                                     = raw_latency_32 * (16 / 32)
                                     = raw_latency_32 * 0.5
```

### get_unique_dims_inter_core_tiling: What It Returns (Unmodified)

```python
# Source: stream/workload/workload.py lines 189-204
# For Gemm_Left (BIG BOY): returns [(z0, 4), (z2, 2)] from mapping's inter_core_tiling[0]
# z0 = seq_len dimension, z2 = hidden dimension
# These are SPATIAL splits only (cores in the array, not temporal iterations)
inter_core_tiling = self.workload.get_unique_dims_inter_core_tiling(n, self.mapping)
```

### TileSizeOption[0] Is Always the Baseline Tile

```python
# Source: stream/opt/search_space.py + Phase 7 D-08 invariant
# search_space.get(dim)[0].tile == baseline tile (e.g. 16 for seq_len)
# This is guaranteed by tile_options[0] = baseline tile ordering in make_swiglu_mapping_v2
self._base_orig_dim_sizes = {d: self.search_space.get(d)[0].tile for d in self.search_space.dims()}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CoreCostLUT for slot latency | TileAwareLatencyEstimator | Phase 6 | Enables variable tile enumeration; formula is correct |
| TilingGenerationStage before CO | TilingGenerationStage after CO | Phase 7 | CO now runs on untiled workload; `self.iterations` fixed at base tile |
| Fixed tile scalars in MILP | quicksum(lat[k] * jw[k]) per candidate | Phase 6 | Correct for single candidate; needs iteration scaling for multi-candidate |
| Iteration scaling: NOT implemented | Must add scaling in Phase 8 | Phase 8 (this work) | Without it, multi-candidate slot_latency is inconsistent with fixed iterations |

**Deprecated/outdated:**
- `cost_lut.get_cost()`: Removed in Phase 6. No instances remain in `_slot_latency_constraints`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (installed in .venv) |
| Config file | `pyproject.toml` or implicit |
| Quick run command | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x -q` |
| Full suite command | `.venv/bin/pytest tests/unit/ -q` |
| Regression (slow) | `.venv/bin/pytest tests/regression/test_baseline.py -m slow -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LAT-01 | TileAwareLatencyEstimator produces same MACs/cycles as old path | unit | `.venv/bin/pytest tests/unit/test_tile_aware_latency.py -x` | Yes (6 tests) |
| LAT-02 | _slot_latency_constraints applies iteration scaling without double-counting | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py::test_slot_latency_variable_mode -x` | Yes (currently FAILING — fix makes it pass) |
| LAT-03 | Single-candidate BIG BOY matches baseline: latency_total=922357343, latency_per_iteration=10716 | regression (slow) | `.venv/bin/pytest tests/regression/test_baseline.py -m slow -x` | Yes |

### Sampling Rate

- **Per task commit:** `.venv/bin/pytest tests/unit/test_co_tile_variables.py tests/unit/test_tile_aware_latency.py -x -q`
- **Per wave merge:** `.venv/bin/pytest tests/unit/ -q`
- **Phase gate:** Full unit suite green + regression test passes before `/gsd:verify-work`

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements. The unit test `test_slot_latency_variable_mode` already specifies the correct post-fix behavior (`lats == [4, 10]`) and is currently failing. No new test files need to be created; the fix must make the existing test pass.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python + .venv | All code | Yes | `.venv` present | — |
| gurobipy | MILP solving (regression only) | Yes (WLS license) | Confirmed (tests/unit pass) | — |
| pytest | Test runner | Yes | Installed in .venv | — |

---

## Open Questions

1. **Should `_iter_scale_by_jw` be populated as a dict `{jw_var: scale}` or just computed inline?**
   - What we know: The test stub initializes it as `stub._iter_scale_by_jw = {}` and the comment says it is "populated during _slot_latency_constraints"
   - What's unclear: Whether anything downstream consumes `_iter_scale_by_jw`, or if it's purely for debugging per D-05
   - Recommendation: Populate it for debugging consistency with the stub; if unused after fix confirmation, leave it in place as it costs nothing

2. **Does fusion_splits need updating in Phase 8 (D-02)?**
   - What we know: D-02 says fusion_splits are temporal T loop sizes and must be derived from variable tiles. The CONTEXT says "If currently defined as fixed, the mapping representation needs updating."
   - What's unclear: The Phase 7 summary shows `_FusionSplitsStage` computes `fusion_splits` from the untiled workload BEFORE the CO. This may already be correct for the CO's purposes.
   - Recommendation: Investigate whether `fusion_splits` affect `_slot_latency_constraints` at all (grep shows no reference to `fusion_splits` in the method). If not, D-02 is a no-op for this phase and the slot latency fix is the only change needed. This should be verified at task-execution time.

3. **Will `int(round(scaled_lat))` ever lose precision for edge cases?**
   - What we know: `raw_lat` is already an integer (from `ceil(macs/ops_per_cycle)`). `scale = base_tile / candidate_tile`. For all divisor tiles (divisibility is enforced), `scale` is a rational fraction. `raw_lat * scale` may not be an integer.
   - Recommendation: Use `int(round(...))`. The loss is at most 0.5 cycles — negligible given baseline values are in the hundreds of millions of cycles.

---

## Sources

### Primary (HIGH confidence)

- Codebase direct read: `transfer_and_tensor_allocation.py` lines 1249–1298 — slot latency constraint implementation
- Codebase direct read: `tile_aware_latency.py` — TileAwareLatencyEstimator formula
- Codebase direct read: `aie_cost_estimator.py` — reference formula for parity verification
- Codebase direct read: `tests/unit/test_co_tile_variables.py` lines 1846–2007 — stub scaffold and failing test
- Codebase direct read: `tests/regression/test_baseline.py` — regression test structure
- Codebase direct read: `stream/opt/search_space.py` — TileSizeOption and SearchSpace
- Phase summaries: `07-01-SUMMARY.md`, `07-02-SUMMARY.md`, `06-01-SUMMARY.md` — decision trail

### Secondary (MEDIUM confidence)

- Test run output: `1 failed, 103 passed` with `AssertionError: Expected latencies [4, 10], got [8, 10]` — confirms exact bug location
- Phase 7 D-08: "baseline tile must be tile_options[0]" invariant confirmed in 07-02-SUMMARY.md

---

## Metadata

**Confidence breakdown:**
- Bug diagnosis: HIGH — confirmed by running `pytest tests/unit/ -q`, exact failure message identifies missing scale factor
- Fix approach: HIGH — test stub at line 1938–1943 directly specifies the correct implementation pattern
- Regression baseline: HIGH — `922357343` and `10716` documented across multiple sources with full fixture data
- Fusion_splits scope (D-02): MEDIUM — need to verify at execution time whether fusion_splits affect _slot_latency_constraints

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable codebase, fast-moving only if Phase 9 changes the objective structure)
