---
phase: 07-pipeline-integration-e2e-validation
plan: 01
subsystem: pipeline
tags: [pipeline-reorder, tiling-generation, co-allocator, post-solve, fusion-splits]
dependency_graph:
  requires:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py (w[dim,k] vars from Phase 3)
    - stream/cost_model/steady_state_scheduler.py
    - stream/stages/allocation/constraint_optimization_allocation.py
    - stream/stages/generation/tiling_generation.py
    - stream/api.py
  provides:
    - get_selected_tiles() on TransferAndTensorAllocator (reads solved w[dim,k].X)
    - scheduler.selected_tiles attribute (post-solve tile readback)
    - selected_tiles in ctx (set by ConstraintOptimizationAllocationStage)
    - TilingGenerationStage post-solve mode (substitute_loop_sizes_with_selected_tiles)
    - _FusionSplitsStage for pre-CO fusion_splits computation
    - Reordered pipeline: CO runs before TilingGenerationStage
  affects:
    - stream/api.py optimize_allocation_co() pipeline order
    - All downstream stages that read workload/mapping from ctx
tech_stack:
  added: []
  patterns:
    - types.SimpleNamespace stub pattern for allocator unit tests
    - ctx.get("selected_tiles") optional field for post-solve mode detection
    - Stage subclass _FusionSplitsStage inline in api.py
key_files:
  created:
    - tests/unit/test_get_selected_tiles.py
    - tests/unit/test_tiling_generation_post_solve.py
  modified:
    - stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
    - stream/cost_model/steady_state_scheduler.py
    - stream/stages/allocation/constraint_optimization_allocation.py
    - stream/stages/generation/tiling_generation.py
    - stream/api.py
decisions:
  - "get_selected_tiles() added to TransferAndTensorAllocator; reads w[dim,k].X post-solve (per D-06)"
  - "_FusionSplitsStage inline in api.py computes fusion_splits before CO using determine_fusion_splits() on untiled workload"
  - "TilingGenerationStage post-solve mode activated by presence of selected_tiles in ctx; skips determine_fusion_splits()"
  - "substitute_loop_sizes_with_selected_tiles() uses selected_tiles[dim] directly as tile size, not via fusion_splits inversion"
  - "Non-tiled dims in post-solve mode keep original workload size (same contract as legacy mode)"
metrics:
  duration: "~15 minutes"
  completed: "2026-04-08"
  tasks: 2
  files: 5
---

# Phase 07 Plan 01: Pipeline Integration Summary

**One-liner:** CO-first pipeline with get_selected_tiles() extraction, _FusionSplitsStage pre-computation, and post-solve TilingGenerationStage mode driven by ctx.selected_tiles.

## Objective

Reorder the pipeline so TilingGenerationStage runs after the CO (per D-01, D-03), add get_selected_tiles() to the allocator (per D-06), and adapt TilingGenerationStage to accept resolved tiles for post-solve application (per D-07).

## Tasks Completed

### Task 1: Add get_selected_tiles() and thread through scheduler to ctx

**Commits:** 7ede4e4

- Added `get_selected_tiles() -> dict[LayerDim, int]` to `TransferAndTensorAllocator`. Iterates `self.search_space.dims()`, finds `k` where `w[(dim,k)].X > VAR_THRESHOLD`, returns `{dim: opt.tile}`. Returns `{}` when `self.w` is empty (fixed-tile scalar mode with no search space).
- Added `self.selected_tiles = tta.get_selected_tiles()` in `SteadyStateScheduler.run()` after `tta.solve()` returns.
- Updated `ConstraintOptimizationAllocationStage.run()` to include `selected_tiles=scheduler.selected_tiles` in `ctx.set(...)`.
- Created `tests/unit/test_get_selected_tiles.py`: 5 tests covering empty-w, single-candidate, multi-candidate selection, multi-dim selection, and VAR_THRESHOLD boundary.

### Task 2: Reorder pipeline, pre-compute fusion_splits, adapt TilingGenerationStage for post-solve

**Commits:** 8f52642

- Added `_FusionSplitsStage` inline class in `stream/api.py`. Calls `determine_fusion_splits(workload, mapping)` on the untiled workload+mapping and sets `fusion_splits` on ctx. This replaces the side-effect previously produced by TilingGenerationStage running before the CO.
- Reordered `optimize_allocation_co()` stages per D-03:
  `AcceleratorParser -> ONNXModelParser -> MappingParser -> CandidateFilter -> _FusionSplitsStage -> CoreCostEstimation -> ConstraintOptimizationAllocation -> TilingGeneration -> MemoryAccessesEstimation`
- Adapted `TilingGenerationStage`:
  - `__init__` reads `self.selected_tiles = self.ctx.get("selected_tiles")` (None if absent)
  - `run()` branches: post-solve mode (selected_tiles not None) uses `substitute_loop_sizes_with_selected_tiles()`, skips `determine_fusion_splits()`; legacy mode preserves existing behavior
  - Added `substitute_loop_sizes_with_selected_tiles()`: uses `selected_tiles[dim]` as tile size, fills non-tiled dims from workload, asserts divisibility
- Created `tests/unit/test_tiling_generation_post_solve.py`: 6 tests covering post-solve mode (correct tiled sizes, non-divisibility assertion, non-tiled dims preserved) and legacy mode (determine_fusion_splits called, selected_tiles attribute is None).

## Verification

```
pytest tests/unit/ -x -q
104 passed in 0.79s
```

All unit tests pass with no regressions.

## Deviations from Plan

### Auto-deviation: merged arne/variable-tilesize into worktree before starting

The worktree `agent-a0b0d1cc` was based off `main` (commit `92b4ec3`) rather than `arne/variable-tilesize`. The target files (`search_space.py`, `w[dim,k]` variables in allocator, etc.) were only present on `arne/variable-tilesize`. Performed `git merge --ff-only arne/variable-tilesize` before beginning implementation. This is not a code deviation — it restores the intended execution baseline.

### Plan deviation: `substitute_loop_sizes_with_selected_tiles` simpler than specified

The plan specified using `collect_spatial_unrollings` to compute `tile_size * spatial_unrolling`. After inspecting the actual code, `selected_tiles[dim]` is already the per-core tile size (not the per-array tile size), and `with_modified_dimension_sizes` takes the absolute tiled size. The correct implementation uses `selected_tiles[dim]` directly as the tiled dimension size (same as what the existing `substitute_loop_sizes_with_tiled_sizes` computes as `wanted_tile_size`). No spatial unrolling multiplication needed at this stage.

## Known Stubs

None — all code paths are wired to real logic.

## Self-Check: PASSED

- FOUND: stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py
- FOUND: stream/cost_model/steady_state_scheduler.py
- FOUND: stream/stages/allocation/constraint_optimization_allocation.py
- FOUND: stream/stages/generation/tiling_generation.py
- FOUND: stream/api.py
- FOUND: tests/unit/test_get_selected_tiles.py
- FOUND: tests/unit/test_tiling_generation_post_solve.py
- FOUND commit 7ede4e4: feat(07-01): add get_selected_tiles() and thread through scheduler to ctx
- FOUND commit 8f52642: feat(07-01): reorder CO pipeline, add fusion_splits stage, post-solve tiling
