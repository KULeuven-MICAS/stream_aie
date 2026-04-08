# Phase 7: Pipeline Integration + E2E Validation - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Move TilingGenerationStage from before the CO to after it (post-solve). The CO runs on the untiled workload and selects tile sizes via w[dim,k]. After solving, TilingGenerationStage applies the chosen tiles to produce a tiled workload copy for downstream inspection. Extend the CLI to accept multiple tile candidates per dimension. Validate end-to-end on SwiGLU BIG BOY with multiple candidates.

</domain>

<decisions>
## Implementation Decisions

### TilingGenerationStage Relocation
- **D-01:** TilingGenerationStage moves from before the CO to after it in the pipeline. The CO runs on the **untiled** workload with variable tile candidates. Post-solve, TilingGenerationStage applies the chosen tiles (from `get_selected_tiles()`) to produce a tiled workload copy for inspection/visualization.
- **D-02:** The same flow applies to both fixed-tile (single candidate) and variable-tile (multiple candidates) modes. In fixed mode, the CO "selects" the only available tile and TilingGenerationStage applies it post-solve. No conditional branching — one unified pipeline.
- **D-03:** The pipeline order becomes: `MappingParser → CandidateFilter → CoreCostEstimation → CO Allocation → TilingGeneration (post-solve)`.

### CLI Design
- **D-04:** Extend existing `main_swiglu_v2.py` CLI args to accept multiple values using `nargs='+'`. Example: `--seq_len_tile_size 16 32 64`. Single value = fixed mode (backward compatible), multiple values = variable mode. No new CLI file needed.
- **D-05:** The args feed into `tile_options` lists in the mapping, which CandidateFilterStage processes into SearchSpace. Multiple values per dimension trigger the variable tile CO path automatically.

### Post-Solve Tile Extraction
- **D-06:** Add `get_selected_tiles() -> dict[LayerDim, int]` method to `TransferAndTensorAllocator`. It reads solved `w[dim,k].X` values to determine the selected tile size per dimension. The allocator owns the w variables and is the right place for this.
- **D-07:** The pipeline reads `get_selected_tiles()` from the allocator result and passes the selected tiles to TilingGenerationStage for post-solve application.

### E2E Validation
- **D-08:** The validation runs `main_swiglu_v2.py` with BIG BOY config and multiple candidates per dimension. It must complete without error, report the selected tile size per dimension, and produce a CO objective at least as good as the fixed-tile baseline.
- **D-09:** Selected tile sizes must be valid divisors of their respective workload dimensions.

### Claude's Discretion
- How TilingGenerationStage is adapted to accept resolved tiles instead of computing them from mapping
- Whether the post-solve tiled workload is written back to ctx or returned separately
- Test structure for E2E validation (pytest integration test vs script)
- Whether to add a `--report` flag to print tile selection summary

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline (Primary Modification Target)
- `stream/api.py` — `optimize_allocation_co()` pipeline function; current stage ordering
- `stream/stages/generation/tiling_generation.py` — `TilingGenerationStage`; substitutes loop ranges with tiled sizes
- `stream/stages/generation/candidate_filter_stage.py` — `CandidateFilterStage`; builds SearchSpace from tile_options_raw

### CO Allocator (Tile Extraction)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `TransferAndTensorAllocator`; `self.w[(dim, k)]` binary variables; `solve()` method
- `stream/opt/search_space.py` — `SearchSpace` and `TileSizeOption`

### CLI Entry Point
- `main_swiglu_v2.py` — Current CLI with scalar tile args; `run_swiglu_v2()` function
- `main_swiglu_dse.py` — Existing DSE entry point (reference for multi-candidate patterns)

### Mapping
- `stream/mapping/mapping.py` — `FusedGroup.intra_core_tiling`; how tile sizes are stored in mapping

### Tests
- `tests/unit/test_co_tile_variables.py` — CO tile variable tests
- `tests/regression/test_baseline.py` — Regression tests; must still pass

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CandidateFilterStage`: Already builds SearchSpace from tile_options_raw — reuse as-is
- `SearchSpace.get(dim)` → candidate list: Already threaded through CO pipeline
- `make_swiglu_mapping_v2()`: Already accepts `tile_options` lists per dimension
- `TilingGenerationStage`: Existing tiling logic — adapt to accept resolved tiles

### Established Patterns
- Pipeline stages communicate via `StageContext` (ctx)
- `argparse nargs='+'` for multi-value CLI args
- Post-solve result extraction already exists for z_stop, fire counts, etc.

### Integration Points
- `optimize_allocation_co()` in `api.py` controls stage ordering — reorder here
- `ConstraintOptimizationAllocationStage` returns allocator results — add `selected_tiles` to output
- `TilingGenerationStage` reads tile sizes from mapping — adapt to read from solved results instead

</code_context>

<specifics>
## Specific Ideas

- The E2E test should use BIG BOY config with multiple candidates (e.g., seq_len: [8, 16, 32], embedding: [64, 128, 256], hidden: [16, 32, 64]) and verify the CO picks valid tiles with an objective at least as good as the single-candidate baseline.
- TilingGenerationStage already knows how to apply tiling — it just needs to receive the tile sizes from a different source (solved CO result instead of fixed mapping).
- The `get_selected_tiles()` method is straightforward: iterate `self.w`, find `k` where `w[dim,k].X > 0.5`, look up the tile value from SearchSpace.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-pipeline-integration-e2e-validation*
*Context gathered: 2026-04-08*
