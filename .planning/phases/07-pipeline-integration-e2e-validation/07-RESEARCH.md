# Phase 07: Pipeline Integration + E2E Validation — Research

**Researched:** 2026-04-08
**Domain:** CO pipeline stage ordering, CLI extension, post-solve tile extraction, E2E validation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** TilingGenerationStage moves from before the CO to after it. The CO runs on the **untiled** workload with variable tile candidates. Post-solve, TilingGenerationStage applies the chosen tiles (from `get_selected_tiles()`) to produce a tiled workload copy for inspection/visualization.
- **D-02:** The same flow applies to both fixed-tile and variable-tile modes. In fixed mode, the CO "selects" the only available tile. No conditional branching — one unified pipeline.
- **D-03:** Pipeline order becomes: `MappingParser → CandidateFilter → CoreCostEstimation → CO Allocation → TilingGeneration (post-solve)`.
- **D-04:** Extend existing `main_swiglu_v2.py` CLI args using `nargs='+'`. Example: `--seq_len_tile_size 16 32 64`. Single value = fixed mode (backward compatible). No new CLI file.
- **D-05:** The args feed into `tile_options` lists in the mapping. Multiple values per dimension trigger the variable tile CO path automatically.
- **D-06:** Add `get_selected_tiles() -> dict[LayerDim, int]` method to `TransferAndTensorAllocator`. It reads solved `w[dim,k].X` values. The allocator owns the w variables and is the right place for this.
- **D-07:** The pipeline reads `get_selected_tiles()` from the allocator result and passes the selected tiles to TilingGenerationStage for post-solve application.
- **D-08:** Validation runs `main_swiglu_v2.py` with BIG BOY config and multiple candidates. Must complete without error, report selected tile sizes, and produce CO objective at least as good as the fixed-tile baseline.
- **D-09:** Selected tile sizes must be valid divisors of their respective workload dimensions.

### Claude's Discretion

- How TilingGenerationStage is adapted to accept resolved tiles instead of computing them from mapping
- Whether the post-solve tiled workload is written back to ctx or returned separately
- Test structure for E2E validation (pytest integration test vs script)
- Whether to add a `--report` flag to print tile selection summary

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PIPE-01 | End-to-end validation: variable tile CO on SwiGLU BIG BOY config selects valid tile sizes and produces a feasible allocation | CLI nargs='+' extension + E2E test with multiple candidates; `get_selected_tiles()` + divisibility check |
| PIPE-02 | TilingGenerationStage removed from the variable tile pipeline path — tile sizes determined by the CO solver, not a preceding stage | Stage reorder in `api.py`; `fusion_splits` supplied pre-solve; TilingGenerationStage adapted to accept resolved tiles post-solve |
</phase_requirements>

---

## Summary

Phase 7 connects all prior phase work into a working end-to-end pipeline. It has three separable engineering problems:

**Problem 1 (D-03/PIPE-02):** Reorder the pipeline in `api.py` so TilingGenerationStage runs after the CO instead of before it. The CO currently requires `fusion_splits` (a `dict[LayerDim, int]`) from TilingGenerationStage; when TilingGenerationStage is removed from before the CO, `fusion_splits` must be produced another way. `determine_fusion_splits()` reads `fused_group.intra_core_tiling` which uses `tile_options[0]` (via `MappingFactory._convert_intra_core_tiling_entry`). This means a representative fusion split (using `tile_options[0]`) can be computed before the CO, and the CO's variable-tile expressions (implemented in phases 3–6 via `candidate_loop_sizes`) override the per-candidate temporal loop counts at constraint-build time. The planner must ensure `fusion_splits` is set in ctx before `ConstraintOptimizationAllocationStage` runs.

**Problem 2 (D-06/D-07):** Add `get_selected_tiles()` to `TransferAndTensorAllocator` and thread the result back through `SteadyStateScheduler` and `ConstraintOptimizationAllocationStage` to the post-solve TilingGenerationStage call.

**Problem 3 (D-04/PIPE-01):** Change scalar `--seq_len_tile_size` etc. CLI args to `nargs='+'` and update the downstream call sites so multiple values flow into `tile_options` lists. Validate end-to-end with BIG BOY multi-candidate config.

**Primary recommendation:** Implement in three sequential plans: (1) `fusion_splits` computation + pipeline reorder, (2) `get_selected_tiles()` + post-solve TilingGenerationStage, (3) CLI `nargs='+'` + E2E test.

---

## Standard Stack

No new libraries needed. All work is internal Python using existing stack.

### Core (Existing)
| Component | Version | Purpose | Role in Phase |
|-----------|---------|---------|---------------|
| gurobipy | installed | MILP solver | `w[dim,k].X` post-solve readback |
| pytest | 9.0.2 | Test framework | E2E integration test |
| argparse | stdlib | CLI parsing | `nargs='+'` multi-value args |
| Python | 3.12 | Runtime | — |

**Gurobi license:** Academic WLS license confirmed valid on this machine.

**Installation:** None required.

---

## Architecture Patterns

### Established Pipeline Communication Pattern
Stages communicate via `StageContext` (ctx). Each stage reads from ctx, does work, sets results back into ctx, then delegates to the next stage via `yield from sub_stage.run()`.

### Current Pipeline Order (to be changed)
```
optimize_allocation_co() in stream/api.py:
  AcceleratorParserStage
  ONNXModelParserStage
  MappingParserStage          # sets: mapping, tile_options_raw
  TilingGenerationStage       # sets: workload (tiled), mapping (tiled), fusion_splits  ← REMOVE FROM HERE
  CandidateFilterStage        # sets: search_space
  CoreCostEstimationStage     # sets: latency_estimator
  ConstraintOptimizationAllocationStage  # requires: fusion_splits
  MemoryAccessesEstimationStage
```

### Target Pipeline Order (D-03)
```
  AcceleratorParserStage
  ONNXModelParserStage
  MappingParserStage           # sets: mapping, tile_options_raw
  CandidateFilterStage         # sets: search_space
  CoreCostEstimationStage      # sets: latency_estimator (uses untiled workload)
  FusionSplitsStage or inline  # sets: fusion_splits (from mapping, using tile_options[0])
  ConstraintOptimizationAllocationStage  # allocator.w[dim,k] solved; allocator stored on ctx
  TilingGenerationStage        # post-solve: receives selected_tiles; applies tiling
```

### Pattern: fusion_splits Pre-Computation Without TilingGenerationStage

`determine_fusion_splits(workload, mapping)` is a pure function in `stream/workload/utils.py`. It reads `fused_group.intra_core_tiling` which, after `MappingParserStage`, contains `tile_options[0]` for each dimension (set by `MappingFactory._convert_intra_core_tiling_entry` line 122). Therefore, calling `determine_fusion_splits()` directly on the untiled workload and mapping (already in ctx after MappingParserStage) produces the same result as TilingGenerationStage produced in the pre-reorder flow — it just uses the first candidate as representative.

The existing phases 3–6 already handle variable temporal loop counts: `SteadyStateIterationSpace.candidate_loop_sizes()` computes `(K, T)` per candidate tile from the untiled `workload_size`. The SSIS temporal variable size from `fusion_splits` is overridden at constraint-build time by the CO's variable expressions.

**Implementation option:** A dedicated `FusionSplitsComputationStage` or inline in a refactored `TilingGenerationStage.__init__` can call `determine_fusion_splits()` and `ctx.set(fusion_splits=...)` before delegating to the CO. Claude's discretion per CONTEXT.md.

### Pattern: post-solve TilingGenerationStage

TilingGenerationStage currently:
1. Calls `determine_fusion_splits()` — not needed post-solve (already in ctx)
2. Calls `substitute_loop_sizes_with_tiled_sizes()` — needs to use `selected_tiles` instead of reading from mapping
3. Calls `workload.with_modified_dimension_sizes(tiled_sizes)` — reusable as-is
4. Calls `mapping.with_updated_workload(tiled_workload, workload)` — reusable as-is
5. Sets `workload=tiled_workload, mapping=tiled_mapping, fusion_splits=...` in ctx

In post-solve mode, step 2 must receive the resolved tile sizes from `get_selected_tiles()` instead of deriving them from `fusion_splits`. The simplest adaptation: add an optional `selected_tiles: dict[LayerDim, int] | None = None` parameter to TilingGenerationStage's `__init__`; if provided, use them directly in place of the fusion-split-derived sizes.

### Pattern: get_selected_tiles()

```python
# Source: direct code inspection, stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
def get_selected_tiles(self) -> dict[LayerDim, int]:
    """Return {dim: tile_size} for each dimension in the search space, post-solve."""
    if not self.w:
        return {}
    result: dict[LayerDim, int] = {}
    for dim in self.search_space.dims():
        options = self.search_space.get(dim)
        for k, opt in enumerate(options):
            if self.w[(dim, k)].X > self.VAR_THRESHOLD:
                result[dim] = opt.tile
                break
    return result
```

`self.VAR_THRESHOLD = 0.5` is already defined on the class. Method must only be called after `solve()` completes (Gurobi status OPTIMAL).

### Pattern: nargs='+' CLI Extension

```python
# Current (argparse):
parser.add_argument("--seq_len_tile_size", type=int, default=16, ...)

# New (argparse nargs='+'):
parser.add_argument("--seq_len_tile_size", type=int, nargs="+", default=[16], ...)

# args.seq_len_tile_size is already a list; no wrapping needed
seq_len_tile_options = args.seq_len_tile_size  # replaces [args.seq_len_tile_size]
```

Backward compatibility: `--seq_len_tile_size 16` (single value) remains valid and produces `[16]`, which is fixed-tile mode.

### Pattern: make_swiglu_mapping_v2 Multi-Candidate Issue

`make_swiglu_mapping_v2()` uses `tile_options[0]` to:
1. Set kernel `m`, `k`, `n` parameters in the YAML
2. Assert divisibility of workload dimensions

In multi-candidate mode, kernel `m/k/n` are set to the first candidate. `TileAwareLatencyEstimator` in phase 6 computes latency from workload dimension sizes, not from the kernel's `m/k/n` YAML params. The YAML kernel params are used by codegen (disabled for CO validation) and by `AIECostEstimator.ops_per_cycle()`. Claude should verify whether `ops_per_cycle()` reads kernel dims from the YAML or from the workload. If it reads from workload dims, the first-candidate kernel params are fine for the CO validation run.

### Anti-Patterns to Avoid

- **Calling get_selected_tiles() before solve() completes:** `w[dim,k].X` raises GurobiError on unoptimized model.
- **Applying TilingGenerationStage's tiling to the already-in-ctx workload:** The CO allocation results (scheduler, ssis, tensor allocations) reference the untiled workload's nodes. Post-solve tiling produces a separate copy for inspection; the scheduler's `steady_state_workload` must remain the untiled-but-solved workload.
- **Changing the nargs='+' default to a list literal in argparse:** `default=[16]` with `nargs='+'` works correctly. Do not use `default=16` (int) with `nargs='+'`.
- **Asserting fixed tile divisibility in make_swiglu_mapping_v2 with multi-candidates:** Current assertions use `tile_options[0]`. With multiple candidates some may be non-divisors — but CandidateFilterStage already handles filtering. The mapping assertions should use `tile_options[0]` (representative) or be removed.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Computing fusion_splits | Custom analysis of workload dims | `determine_fusion_splits(workload, mapping)` in `stream/workload/utils.py` | Already handles spatial unrolling, multi-layer, assertion logic |
| Reading solved w[dim,k] | Manual Gurobi variable iteration | `get_selected_tiles()` pattern (to be added to TransferAndTensorAllocator) | Consistent with existing `get_tensor_allocations()`, `get_transfer_routing()` post-solve patterns |
| Building tiled workload post-solve | Custom workload copy | `workload.with_modified_dimension_sizes(tiled_sizes)` (already in TilingGenerationStage) | Reuse existing method |
| Multi-value CLI args | Custom string splitting | `argparse nargs='+'` | Standard library; returns list directly |
| Verifying divisibility | Manual modulo checks | CandidateFilterStage already filters; `assert workload_size % tile == 0` if needed in TilingGeneration | Already proven in production |

---

## Common Pitfalls

### Pitfall 1: fusion_splits Missing from ctx When CO Runs
**What goes wrong:** `ConstraintOptimizationAllocationStage.REQUIRED_FIELDS` includes `fusion_splits`. If TilingGenerationStage is removed from before CO without placing fusion_splits into ctx another way, the stage fails at startup.
**Why it happens:** `fusion_splits` was a side effect of TilingGenerationStage being in-pipeline; removing it removes the side effect.
**How to avoid:** Either keep a lightweight stage that calls `determine_fusion_splits()` and sets ctx before CO, or compute it inline in a refactored pipeline entry point.
**Warning signs:** `KeyError: 'fusion_splits'` on `ctx.get("fusion_splits")` at stage initialization.

### Pitfall 2: TilingGenerationStage Modifies the Workload the Allocator References
**What goes wrong:** The CO's `SteadyStateScheduler` builds `self.steady_state_workload` referencing untiled workload nodes. If post-solve TilingGenerationStage overwrites `ctx.workload` with a tiled workload, downstream stages (e.g., MemoryAccessesEstimationStage) may receive mismatched workload/scheduler.
**Why it happens:** TilingGenerationStage currently writes `workload=tiled_workload` back to ctx unconditionally.
**How to avoid:** Post-solve TilingGenerationStage should either (a) not overwrite ctx.workload, storing tiled_workload under a different key (e.g., `tiled_workload`), or (b) comes after all stages that depend on the CO-solved workload.
**Warning signs:** Node-not-found errors or mismatched SSIS lookups in MemoryAccessesEstimationStage.

### Pitfall 3: make_swiglu_mapping_v2 Assertions Fail on Non-First Candidates
**What goes wrong:** `make_swiglu_mapping_v2` asserts `embedding_dim % embedding_tile_options[0] == 0`. If the user passes multiple values and `tile_options[0]` is a non-divisor, the mapping creation raises AssertionError before CandidateFilterStage can filter it.
**Why it happens:** Assertions were written for single-candidate mode.
**How to avoid:** Use only `tile_options[0]` for the representative assertion (this is the current behavior and will pass as long as the first candidate is valid). Alternatively, assert that at least one candidate is divisible.
**Warning signs:** `AssertionError: embedding_dim must be divisible by embedding_tile_size` on CLI invocation.

### Pitfall 4: Regression Test Breaks After Pipeline Reorder
**What goes wrong:** `tests/regression/test_baseline.py` with `seq_len_tile_options=[16]` (single candidate, fixed-tile mode) must still produce the same `latency_total=1030232714` and fire counts as the fixture.
**Why it happens:** Pipeline reorder changes stage execution order; if the untiled workload path is not equivalent to the previously tiled path for single-candidate runs, objective values diverge.
**How to avoid:** Run regression test (`pytest tests/regression/test_baseline.py -m slow`) after pipeline reorder, before E2E multi-candidate test. The two paths must be numerically equivalent.
**Warning signs:** `latency_total` regression or changed fire counts in baseline test.

### Pitfall 5: get_selected_tiles() Returns Empty dict in Fixed-Tile Mode
**What goes wrong:** If search_space has a single candidate, `w[dim,k]` still exists (one-hot with one option). `get_selected_tiles()` must handle this case and return the single tile.
**Why it happens:** D-02 says unified flow; fixed-tile mode passes through the same w-variable path.
**How to avoid:** The `get_selected_tiles()` implementation naturally handles this: iterate options, find `w[dim, 0].X > 0.5`, return `opt.tile`. No special case needed.
**Warning signs:** Empty dict from `get_selected_tiles()` for single-candidate run (would mean w variable wasn't set).

---

## Code Examples

### Example: how fusion_splits is currently set (to be replicated without TilingGenerationStage)

```python
# Source: stream/stages/generation/tiling_generation.py lines 44-55
def run(self):
    self.fusion_splits = determine_fusion_splits(self.workload, self.mapping)
    self.tiled_sizes = self.substitute_loop_sizes_with_tiled_sizes()
    self.tiled_workload = self.workload.with_modified_dimension_sizes(self.tiled_sizes)
    self.tiled_mapping = self.mapping.with_updated_workload(self.tiled_workload, self.workload)
    self.ctx.set(
        workload=self.tiled_workload,
        mapping=self.tiled_mapping,
        fusion_splits=self.fusion_splits,
    )
```

For the pre-CO step, only `fusion_splits` needs to be computed and set; `workload` and `mapping` remain untiled.

### Example: how MappingFactory sets tile_options[0] as the canonical tile for FusedGroup

```python
# Source: stream/parser/mapping_factory.py lines 121-124
if "tile_options" in entry:
    tile_value = int(entry["tile_options"][0])
else:
    tile_value = int(entry["tile"])
return dim, tile_value
```

This means `mapping.fused_groups[0].intra_core_tiling` always has a concrete int, so `determine_fusion_splits()` can run on the untiled workload without modification.

### Example: how solve() returns values (pattern for adding get_selected_tiles())

```python
# Source: stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py lines 1586-1609
def solve(self, *, tee: bool = True) -> tuple[...]:
    self.model.optimize(self._mip_progress_callback)
    # ... status check ...
    tensor_alloc = self.get_tensor_allocations()   # reads x_tensor_choice vars post-solve
    routing = self.get_transfer_routing()           # reads y_path_choice vars post-solve
    # get_selected_tiles() follows the same pattern reading w[dim,k] vars
    return (tensor_reuse_levels, tensor_alloc, routing, ...)
```

### Example: existing post-solve result extraction pattern

```python
# Source: stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py lines 1625-1636
def get_transfer_routing(self) -> TransferAlloc:
    routing: TransferAlloc = {}
    for tr in self.transfer_nodes:
        chosen = [
            choice
            for choice in self.possible_transfer_allocations[tr]
            if self.y_path_choice[(tr, choice)].X > self.VAR_THRESHOLD
        ]
        if len(chosen) != 1:
            raise ValueError(f"{tr.name}: expected exactly one routing choice, got {chosen}")
        routing[tr] = chosen[0]
    return routing
# get_selected_tiles() follows the identical pattern over self.w[(dim, k)]
```

### Example: current CLI arg pattern (to be changed)

```python
# Source: main_swiglu_v2.py lines 181-196
parser.add_argument("--seq_len_tile_size", type=int, default=16, ...)
# ...
seq_len_tile_options = [args.seq_len_tile_size]   # wraps scalar into list
```

After change:
```python
parser.add_argument("--seq_len_tile_size", type=int, nargs="+", default=[16], ...)
# ...
seq_len_tile_options = args.seq_len_tile_size     # already a list; no wrapping
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TilingGenerationStage before CO | TilingGenerationStage after CO (phase 7) | Phase 7 | CO runs on untiled workload; fusion_splits computed separately |
| CoreCostLUT for latency | TileAwareLatencyEstimator | Phase 6 | Latency computed on-demand from workload dims; works on untiled workload |
| Scalar tile CLI args | nargs='+' multi-value args | Phase 7 | Single arg invocation stays backward compatible |

---

## Open Questions

1. **Does AIECostEstimator.ops_per_cycle() use kernel m/k/n from the YAML or from workload dims?**
   - What we know: `TileAwareLatencyEstimator.estimate()` reads workload dims directly (line 76–78 in tile_aware_latency.py), uses `AIECostEstimator.ops_per_cycle(node, core)` for ops throughput.
   - What's unclear: Whether `ops_per_cycle()` dispatches based on YAML kernel params (m/k/n) or just kernel type (gemm/silu/etc.).
   - Recommendation: The implementer should check `stream/stages/estimation/aie_cost_estimator.py` before writing `make_swiglu_mapping_v2` multi-candidate handling. If `ops_per_cycle` is kernel-type-only, first-candidate kernel params are fine.

2. **Should post-solve TilingGenerationStage write tiled_workload back to ctx.workload or a new key?**
   - What we know: MemoryAccessesEstimationStage runs after TilingGenerationStage in the target pipeline order. It likely reads ctx.workload.
   - What's unclear: Whether MemoryAccessesEstimationStage needs the tiled or untiled workload.
   - Recommendation: Claude's discretion per CONTEXT.md. The safest default: write to `ctx.tiled_workload` (new key) to avoid clobbering the scheduler-referenced workload. If MemoryAccessesEstimationStage is tested and confirmed to work with tiled workload, ctx.workload can be overwritten.

3. **Does the E2E multi-candidate test need to be a pytest slow test or a standalone script?**
   - What we know: Existing regression tests use `@pytest.mark.slow` and take several minutes with Gurobi. The E2E test will also require Gurobi.
   - Recommendation: Add to `tests/regression/` with `@pytest.mark.slow`. Reuse the `run_swiglu_v2()` function from `main_swiglu_v2.py` to keep the test thin. This follows the established pattern from `test_baseline.py`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | Runtime | ✓ | 3.12.3 | — |
| gurobipy | CO solver | ✓ | installed | — |
| Gurobi license | CO solve | ✓ | Academic WLS (2425809) | — |
| pytest | Test runner | ✓ | 9.0.2 | — |
| baseline fixture | Regression test | ✓ | `tests/regression/fixtures/baseline_bigboy.json` | — |

**Missing dependencies:** None. All dependencies are available.

---

## Validation Architecture

`workflow.nyquist_validation` key is absent from `.planning/config.json` — treated as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | none (markers defined in `tests/conftest.py`) |
| Quick run command | `pytest tests/unit/ -x -q` |
| Full suite command | `pytest tests/ -x -q` |
| Slow tests | `pytest tests/regression/ -m slow -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-01 | Multi-candidate run completes, selects valid tiles, objective >= baseline | slow integration | `pytest tests/regression/test_e2e_variable_tile.py -m slow -x` | ❌ Wave 0 |
| PIPE-01 | Selected tile is a valid divisor of workload dim | unit | `pytest tests/unit/test_get_selected_tiles.py -x` | ❌ Wave 0 |
| PIPE-02 | TilingGenerationStage not in pre-CO pipeline | unit/smoke | part of `test_e2e_variable_tile.py` or separate unit test on api.py stage list | ❌ Wave 0 |
| PIPE-02 | Post-solve tiling applies selected tile correctly | unit | `pytest tests/unit/test_tiling_generation_post_solve.py -x` | ❌ Wave 0 |

**Regression gate:** `pytest tests/regression/test_baseline.py -m slow -x` must still pass after pipeline reorder (fixed-tile mode regression).

### Sampling Rate
- **Per task commit:** `pytest tests/unit/ -x -q`
- **Per wave merge:** `pytest tests/unit/ tests/regression/test_baseline.py -m slow -x -q`
- **Phase gate:** Full suite including `test_e2e_variable_tile.py` green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/regression/test_e2e_variable_tile.py` — E2E multi-candidate test covering PIPE-01 (BIG BOY + multiple candidates)
- [ ] `tests/unit/test_get_selected_tiles.py` — unit test for `get_selected_tiles()` on solved/mocked allocator (PIPE-01 divisibility check)
- [ ] `tests/unit/test_tiling_generation_post_solve.py` — unit test for TilingGenerationStage accepting `selected_tiles` argument (PIPE-02)

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `stream/api.py` — current `optimize_allocation_co()` stage list
- Direct code inspection: `stream/stages/generation/tiling_generation.py` — `run()`, `substitute_loop_sizes_with_tiled_sizes()`
- Direct code inspection: `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `solve()`, `get_transfer_routing()`, `VAR_THRESHOLD`, `self.w`, `__create_tile_selection_vars()`
- Direct code inspection: `stream/workload/utils.py` — `determine_fusion_splits()`, `_add_temporal_iteration_variables()`
- Direct code inspection: `stream/parser/mapping_factory.py` — `_convert_intra_core_tiling_entry()` tile_options[0] behavior
- Direct code inspection: `stream/stages/parsing/mapping_parser.py` — `tile_options_raw` extraction
- Direct code inspection: `stream/inputs/aie/mapping/make_swiglu_mapping_v2.py` — tile_options[0] kernel param usage
- Direct code inspection: `main_swiglu_v2.py` — current scalar CLI args and conversion to list
- Direct code inspection: `stream/workload/steady_state/iteration_space.py` — `candidate_loop_sizes()` method
- Direct code inspection: `stream/stages/allocation/constraint_optimization_allocation.py` — `REQUIRED_FIELDS` including `fusion_splits`
- Direct code inspection: `tests/regression/fixtures/baseline_bigboy.json` — baseline values for regression comparison

### Secondary (MEDIUM confidence)
- `stream/cost_model/steady_state_scheduler.py` — how `fusion_splits` flows through to `generate_ssis()` and is used in temporal iteration variable construction

---

## Metadata

**Confidence breakdown:**
- Pipeline reorder mechanics: HIGH — directly read all involved stage code
- fusion_splits supply without TilingGenerationStage: HIGH — traced determine_fusion_splits() to mapping_factory tile_options[0]
- get_selected_tiles() pattern: HIGH — direct parallel with existing get_transfer_routing()
- CLI nargs='+' change: HIGH — trivial argparse pattern, code read directly
- E2E test structure: HIGH — mirrors existing test_baseline.py pattern exactly
- AIECostEstimator ops_per_cycle behavior: MEDIUM — not read in this session; flagged as open question

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable codebase, no fast-moving dependencies)
