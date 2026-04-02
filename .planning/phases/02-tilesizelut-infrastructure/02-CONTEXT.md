# Phase 2: TileSizeLUT Infrastructure - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Pure-Python tile size utility module and candidate filtering pipeline stage. Provides on-demand functions for computing tile-dependent scalar quantities (tensor sizes, SSIS loop sizes, reuse levels, transfer latencies) and a new pipeline stage that pre-filters candidate tile lists for divisibility and memory feasibility. Results are passed via StageContext using a standardized, extensible search space class.

Note: Despite the phase name "TileSizeLUT", the decision was made to NOT build a lookup table. Instead, utility functions compute values on demand, and call-site helper variables provide natural memoization where needed.

</domain>

<decisions>
## Implementation Decisions

### Architecture: Functions over LUT
- **D-01:** No precomputed LUT data structure. Utility functions compute tile-dependent quantities on demand from first principles. Rationale: combinatorial explosion of (tile x dimension x core) entries makes a full LUT unwieldy; functions are simpler, more flexible, and equally performant since they're only called during CO model construction (not during solve).
- **D-02:** Results stored in local helper variables at the CO call site for reuse across multiple constraints. No global cache or memoization layer — the natural scope of model construction provides implicit memoization.

### Candidate Filtering Strategy
- **D-03:** Divisibility filtering is mandatory — a candidate tile must divide its workload dimension size exactly.
- **D-04:** Single-tensor memory check as a lightweight sanity filter — remove candidates where even one tensor exceeds core memory capacity. The CO handles full allocation feasibility through its capacity constraints.
- **D-05:** No full allocation feasibility pre-check — avoids duplicating CO constraint logic and avoids filtering candidates that might be feasible in combination with other choices.

### Pipeline Integration
- **D-06:** New pipeline stage for candidate filtering, running before the CO stage. Filters candidates, builds a search space object, and passes it into StageContext.
- **D-07:** Tile utility functions live in a standalone module (e.g. `stream/opt/tile_size_utils.py` or similar clear namespace). The CO allocator imports and calls them during model construction. Clean logical separation from the 1700-line allocator.

### Search Space Design
- **D-08:** Standardized search space class (e.g. `SearchSpace`) with typed option entries (e.g. `TileSizeOption`). Extensible for future optimization variables beyond tile sizes — the system should support plugging in new knobs (reuse levels, allocation strategies, etc.) without restructuring.
- **D-09:** The goal is a modular CO optimization system where optimization dimensions can be easily tuned, turned on/off. Tile size selection is the first such dimension.

### Dimension Identity Model
- **D-10:** Search space keyed by unique dimension group, not individual LayerDim names. One candidate set per unique workload dimension. Multiple LayerDims that share a workload dimension (e.g. `Gemm_Left.D1` and `Gemm_Down.D2` both being the embedding dimension) share one set of candidates and one set of CO selection variables.
- **D-11:** This aligns with how `determine_fusion_splits()` already groups dimensions in the existing pipeline.

### Claude's Discretion
- Module path and naming conventions for the utility functions and search space classes
- Internal structure of the search space class (fields, methods, serialization)
- Exact placement of the new pipeline stage in the stage chain
- Unit test organization and fixture design

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Tile-Dependent Quantity Sources
- `stream/workload/tensor.py` — `Tensor.size_bits()` computes tensor size as `bitwidth * prod(shape)`; shape depends on tile via tiling
- `stream/workload/workload.py` — `get_tensor_shape_with_tiling()` applies inter-core tiling to reduce dimensions; `unique_dimensions()` uses affine map analysis to find dimension groups
- `stream/workload/utils.py` — `determine_fusion_splits()` maps `FusedGroup.intra_core_tiling` dims to split factors; `generate_steady_state_iteration_spaces()` creates SSIS loop variables
- `stream/workload/steady_state/iteration_space.py` — `SteadyStateIterationSpace` holds kernel/temporal loop sizes and `reuse_factor()`

### CO Allocator (Consumer of Utility Functions)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — 1700-line allocator; `_memory_capacity_constraints()` enforces per-core memory budget; `_transfer_latency_for_path()` computes transfer latencies; this is where Phases 3-5 will call the utility functions
- `stream/hardware/architecture/core.py` — `get_memory_capacity()` returns core memory budget for single-tensor feasibility check

### Pipeline Stage Chain
- `stream/stages/generation/tiling_generation.py` — `TilingGenerationStage` currently computes fusion splits and modifies workload dimensions; new filtering stage should run near here
- `stream/stages/allocation/constraint_optimization_allocation.py` — CO allocation stage that instantiates `SteadyStateScheduler` and `TransferAndTensorAllocator`
- `stream/api.py` — `optimize_allocation_co()` composes the stage pipeline; reference for stage ordering

### Mapping & Data Model
- `stream/mapping/mapping.py` — `FusedGroup.intra_core_tiling` stores (LayerDim, tile_size) tuples; `Mapping` class is the parsed representation
- `stream/inputs/aie/mapping/make_swiglu_mapping_v2.py` — `make_swiglu_mapping_v2()` generates mapping with `tile_options` format

### Phase 1 Reference
- `main_swiglu_v2.py` — v2 entry point; will need to pass candidate lists through the pipeline
- `tests/regression/test_baseline.py` — Regression tests; Phase 2 unit tests should follow same pytest patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `determine_fusion_splits()`: Already groups intra_core_tiling dims and computes split factors — the dimension grouping logic can inform how the search space identifies unique dimension groups
- `Tensor.size_bits()` + `get_tensor_shape_with_tiling()`: Existing tensor size computation chain; utility functions should call into these rather than reimplementing
- `Core.get_memory_capacity()`: Ready-to-use memory budget for single-tensor feasibility check
- `generate_steady_state_iteration_spaces()`: Existing SSIS computation; utility functions for loop sizes should follow this logic

### Established Patterns
- `StageContext` shared mutable dict for passing data between pipeline stages
- Stage chain composed in `optimize_allocation_co()` with ordered callables
- `UpgradedValidator` + cerberus schema for input validation (mapping validator)

### Integration Points
- New filtering stage inserts into the stage chain in `optimize_allocation_co()`, between tiling generation and CO allocation
- Search space object passed via `StageContext` key — CO allocator retrieves it when building the model
- `MappingFactory._convert_intra_core_tiling_entry()` already extracts `tile_options[0]` — Phase 2 needs the full list passed through to the search space

</code_context>

<specifics>
## Specific Ideas

- The search space class should be designed with future extensibility in mind — tile sizes are the first optimization variable, but the user envisions a modular system where other CO knobs can be added as pluggable optimization dimensions
- Utility functions should have a clear, well-named namespace so the CO allocator reads naturally: e.g. `tile_utils.tensor_size(tensor, dim, candidate)` rather than generic names
- The filtering stage should log which candidates were removed and why (divisibility vs memory) for debugging and transparency

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-tilesizelut-infrastructure*
*Context gathered: 2026-04-02*
