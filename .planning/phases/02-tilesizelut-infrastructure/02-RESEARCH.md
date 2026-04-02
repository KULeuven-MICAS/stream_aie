# Phase 02: tilesizelut-infrastructure - Research

**Researched:** 2026-04-02
**Domain:** Pure-Python tile utility functions, candidate filtering pipeline stage, extensible SearchSpace class
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** No precomputed LUT data structure. Utility functions compute tile-dependent quantities on demand.
- **D-02:** Results stored in local helper variables at the CO call site for reuse. No global cache or memoization layer.
- **D-03:** Divisibility filtering is mandatory — a candidate tile must divide its workload dimension size exactly.
- **D-04:** Single-tensor memory check as lightweight sanity filter — remove candidates where even one tensor exceeds core memory capacity.
- **D-05:** No full allocation feasibility pre-check — avoids duplicating CO constraint logic.
- **D-06:** New pipeline stage for candidate filtering, running before the CO stage. Filters candidates, builds a search space object, passes it into StageContext.
- **D-07:** Tile utility functions live in a standalone module (e.g. `stream/opt/tile_size_utils.py`).
- **D-08:** Standardized SearchSpace class with typed option entries (e.g. `TileSizeOption`). Extensible for future optimization variables.
- **D-09:** Modular CO optimization system — tile size selection is the first pluggable optimization dimension.
- **D-10:** Search space keyed by unique dimension group, not individual LayerDim names.
- **D-11:** Aligns with how `determine_fusion_splits()` already groups dimensions.

### Claude's Discretion

- Module path and naming conventions for the utility functions and search space classes
- Internal structure of the search space class (fields, methods, serialization)
- Exact placement of the new pipeline stage in the stage chain
- Unit test organization and fixture design

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TILE-01 | User can specify a single list of candidate tile sizes as input to the optimization | MappingFactory already parses `tile_options` lists from YAML; the factory currently takes `tile_options[0]`. Phase 2 must thread the full list through to the SearchSpace. |
| TILE-02 | Precompute all tile-dependent quantities (tensor sizes, SSIS loop sizes, reuse levels, transfer sizes) for each candidate tile per unique dimension | `Tensor.size_bits(shape)`, `Workload.get_tensor_shape_with_tiling()`, `generate_steady_state_iteration_spaces()` are the exact upstream primitives to call. Utility functions wrap these per candidate. |
| TILE-04 | Candidate tile sizes are pre-filtered for divisibility and memory feasibility | `Workload.get_dimension_size(dim)` provides the workload size for divisibility; `Core.get_memory_capacity()` provides the budget for single-tensor check. |
</phase_requirements>

---

## Summary

Phase 2 builds the plumbing that converts a raw list of candidate tile sizes into a clean, filtered SearchSpace object that later phases can use to construct CO decision variables. The key insight is that no lookup table is built — instead, three things happen: (1) a standalone utility module provides named functions for computing tile-dependent quantities (tensor size, SSIS loop sizes, reuse factor, transfer size), (2) a new pipeline stage uses those functions to evaluate and filter each candidate against divisibility and single-tensor memory constraints, and (3) the surviving candidates are packaged into an extensible SearchSpace keyed by unique dimension group and injected into StageContext.

All the upstream primitives exist and are well-tested: `Tensor.size_bits()`, `Workload.get_tensor_shape_with_tiling()`, `determine_fusion_splits()`, `generate_steady_state_iteration_spaces()`, `Core.get_memory_capacity()`. The only gap is that `MappingFactory._convert_intra_core_tiling_entry()` currently takes `tile_options[0]` and discards the rest; a thin path is needed to thread the full candidate list from the YAML through `FusedGroup` and out the other side into the new stage.

The primary planning risk flagged in STATE.md is `generate_ssis()` side-effect safety: `generate_steady_state_iteration_spaces()` currently prints to stdout (line 75 of `workload/utils.py`) and may have other side effects if called per-candidate. Utility functions for SSIS loop sizes should extract the logic they need without calling the full scheduler's `generate_ssis()` path, which rebuilds a full transfer graph and updates mappings. The TILE-02 quantities are computable from lower-level primitives without that overhead.

**Primary recommendation:** Implement `stream/opt/tile_size_utils.py` (utility functions), `stream/opt/search_space.py` (SearchSpace / TileSizeOption classes), and `stream/stages/generation/candidate_filter_stage.py` (the pipeline stage), then wire the full `tile_options` list from YAML through `MappingFactory` into the stage.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python dataclasses | stdlib | SearchSpace, TileSizeOption value objects | Zero deps; frozen dataclasses give free hashing and immutability |
| pytest | >=7 (already in dev deps) | Unit tests with no Gurobi dependency | Already used in project; `@pytest.mark.slow` pattern already established |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| math.prod | stdlib | Tensor size computation from shape tuple | Used by `Tensor.size_elements()` already |
| logging | stdlib | Filtering stage candidate removal logging | Mirrors `TilingGenerationStage` pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `@dataclass(frozen=True)` for TileSizeOption | `NamedTuple` | NamedTuple is simpler for pure data; dataclass is better if methods need to be added later. Dataclass wins for extensibility (D-08, D-09). |
| Standalone `stream/opt/tile_size_utils.py` | Functions inside allocator | Allocator is already 1700 lines; standalone module keeps it readable and independently testable (TILE-04 requires no Gurobi in tests). |

---

## Architecture Patterns

### Recommended Project Structure
```
stream/
├── opt/
│   ├── tile_size_utils.py      # Pure functions: tensor_size_bits, ssis_loop_sizes, reuse_factor, transfer_size_bits
│   ├── search_space.py         # SearchSpace, TileSizeOption dataclasses
│   └── allocation/             # (existing CO allocator, will import from tile_size_utils in Phase 3+)
└── stages/
    └── generation/
        ├── tiling_generation.py        # (existing)
        └── candidate_filter_stage.py   # New: CandidateFilterStage
tests/
└── unit/
    └── test_tile_size_utils.py         # No Gurobi; tests utility functions and SearchSpace
```

### Pattern 1: Utility Function Signatures
**What:** Pure functions in `tile_size_utils.py` that take a workload, a tensor or node, and a candidate tile size, and return a Python scalar.
**When to use:** Called during CO model construction (Phases 3-5) to get coefficients; called during candidate filtering (Phase 2) for the memory check.
**Example:**
```python
# stream/opt/tile_size_utils.py
from math import prod
from stream.datatypes import LayerDim, InterCoreTiling
from stream.workload.workload import Workload
from stream.workload.tensor import Tensor

def tensor_size_bits(
    workload: Workload,
    tensor: Tensor,
    candidate_tiling: InterCoreTiling,
) -> int:
    """Bits occupied by tensor when a specific inter-core tiling is applied."""
    shape = workload.get_tensor_shape_with_tiling(tensor, candidate_tiling)
    return tensor.size_bits(shape=shape)

def ssis_loop_sizes(
    workload: Workload,
    mapping_with_tile: ...,  # Mapping built for one candidate tile
    fusion_splits_for_tile: dict[LayerDim, int],
) -> dict[...]:
    """Return temporal loop sizes per node for a specific candidate tile."""
    from stream.workload.utils import generate_steady_state_iteration_spaces
    ssis = generate_steady_state_iteration_spaces(workload, mapping_with_tile, fusion_splits_for_tile)
    return ssis
```

### Pattern 2: SearchSpace and TileSizeOption
**What:** A typed container that holds one `TileSizeOption` per (unique_dim, candidate_tile) pair. The SearchSpace is the unit of communication from the filter stage to downstream CO stages.
**When to use:** Created once by CandidateFilterStage; retrieved from StageContext by ConstraintOptimizationAllocationStage.
**Example:**
```python
# stream/opt/search_space.py
from dataclasses import dataclass, field
from stream.datatypes import LayerDim

@dataclass(frozen=True)
class TileSizeOption:
    dim: LayerDim           # unique dimension group (z0, z1, ...)
    tile: int               # candidate tile size (scalar)
    workload_size: int      # original dimension size (for reference)
    # Phase 2: no precomputed quantities stored here — utility functions compute on demand
    # Phase 3+: add tensor_size_bits, ssis_loop_sizes, etc. as needed

@dataclass
class SearchSpace:
    """Holds valid tile candidates per unique workload dimension."""
    options: dict[LayerDim, list[TileSizeOption]] = field(default_factory=dict)

    def add(self, dim: LayerDim, option: TileSizeOption) -> None:
        self.options.setdefault(dim, []).append(option)

    def get(self, dim: LayerDim) -> list[TileSizeOption]:
        return self.options.get(dim, [])

    def dims(self) -> list[LayerDim]:
        return list(self.options.keys())

    def is_empty(self) -> bool:
        return not self.options or all(not v for v in self.options.values())
```

### Pattern 3: CandidateFilterStage
**What:** A Stage subclass that reads `tile_options` from the Mapping's `FusedGroup`, filters per dimension, builds a SearchSpace, and sets it into StageContext.
**When to use:** Inserted between `TilingGenerationStage` and `CoreCostEstimationStage` in `optimize_allocation_co()`.
**Example:**
```python
# stream/stages/generation/candidate_filter_stage.py
import logging
from stream.stages.stage import Stage, StageCallable
from stream.stages.context import StageContext

logger = logging.getLogger(__name__)

class CandidateFilterStage(Stage):
    REQUIRED_FIELDS = ("workload", "mapping", "accelerator", "tile_options_raw")

    def run(self):
        # 1. Resolve unique dims from workload.unique_dimensions()
        # 2. For each (unique_dim, candidate) pair:
        #    a. Divisibility check: workload_size % (candidate * spatial_unrolling) == 0
        #    b. Single-tensor memory check: all tensor sizes <= core.get_memory_capacity()
        #    c. Log removed candidates with reason
        # 3. Build SearchSpace, set into ctx
        # 4. Assert SearchSpace is not empty (all dims must have at least one valid candidate)
        ...
        self.ctx.set(search_space=search_space)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
```

### Pattern 4: Threading tile_options Through MappingFactory
**What:** `FusedGroup` currently stores only `(LayerDim, int)` tuples in `intra_core_tiling`. To pass the full options list, either extend `FusedGroup` or extract options in a pre-processing step before `MappingFactory.create()`.
**When to use:** The simplest approach is a parallel `tile_options_map: dict[LayerDim, list[int]]` extracted during parsing and put into StageContext directly. This avoids touching `FusedGroup` which is used by `TilingGenerationStage` with a single tile value.
**Example path:** `MappingParserStage` already sets `mapping` into ctx. A minimal approach: also extract `tile_options_raw` from the YAML fused_group data and set it into ctx. CandidateFilterStage reads both.

### Pattern 5: Stage Chain Insertion
**What:** Insert `CandidateFilterStage` after `TilingGenerationStage` and before `CoreCostEstimationStage` in `optimize_allocation_co()`.
**Rationale:** `TilingGenerationStage` has already called `with_modified_dimension_sizes()` (tiled workload) and `determine_fusion_splits()`. CandidateFilterStage needs the tiled workload and the unique dimensions — both are available in ctx after tiling generation. `CoreCostEstimationStage` does not need the search space.

```python
# stream/api.py — optimize_allocation_co()
stages: list[StageCallable] = [
    AcceleratorParserStage,
    StreamONNXModelParserStage,
    MappingParserStage,
    TilingGenerationStage,
    CandidateFilterStage,       # <-- new
    CoreCostEstimationStage,
    ConstraintOptimizationAllocationStage,
    MemoryAccessesEstimationStage,
]
```

### Anti-Patterns to Avoid
- **Calling `SteadyStateScheduler.generate_ssis()` per candidate:** This rebuilds the full transfer graph, updates mapping, and updates cost LUT for each call. Use lower-level primitives (`generate_steady_state_iteration_spaces()` from `workload/utils.py`) directly, or defer SSIS computation to Phase 3 when it is needed by the CO.
- **Hardcoding `tile_options` extraction inside `MappingFactory.create()`:** Factory creates a `Mapping` object; it should not also create search-space data structures. Keep concerns separate.
- **Storing computed quantities in TileSizeOption at Phase 2:** The search space at this phase only needs to know which (dim, tile) pairs are valid. Coefficients are added in Phases 3-5 as needed.
- **Filtering against per-node memory capacity instead of per-core:** The memory capacity check uses `Core.get_memory_capacity()` which returns total top-level memory in bits. The check should use the smallest compute core's capacity as a conservative bound.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tensor size for a given tiling | Custom shape arithmetic | `Workload.get_tensor_shape_with_tiling(tensor, succ_tiling)` + `Tensor.size_bits(shape)` | Handles affine dimension relations and clipping to logical tensor bounds |
| Unique dimension grouping | Custom equivalence analysis | `Workload.unique_dimensions()` returning `(z_dims, dim_values)` | Uses RREF of the affine constraint system; replicated logic would drift |
| Fusion split computation | Custom split arithmetic | `determine_fusion_splits(workload, mapping)` | Already handles spatial unrolling interaction and divisibility assertion |
| SSIS loop sizes | Custom loop size computation | `generate_steady_state_iteration_spaces(workload, mapping, fusion_splits)` | Handles spatial, temporal, spatiotemporal variables and their ordering |

---

## Common Pitfalls

### Pitfall 1: Confusing LayerDim with Unique Dimension
**What goes wrong:** Filtering candidates by `LayerDim` name (e.g., `"Gemm_Left.D1"`) instead of by unique dimension group (`z0`). Two LayerDims that map to the same unique dimension would be treated as independent, creating redundant or conflicting candidates.
**Why it happens:** `FusedGroup.intra_core_tiling` stores `(LayerDim, tile_size)` pairs using per-node dimension names. `workload.unique_dimensions()` returns the canonical grouping.
**How to avoid:** Always resolve to unique dims via `workload.unique_dimensions()` before building the SearchSpace. Use the `z0, z1, ...` LayerDims as keys.
**Warning signs:** More entries in SearchSpace than there are unique workload dimensions; duplicate candidates per unique dimension.

### Pitfall 2: Spatial Unrolling Not Factored Into Divisibility Check
**What goes wrong:** Checking `workload_size % candidate_tile == 0` without accounting for spatial unrolling. The effective tiling is `candidate_tile * spatial_unrolling`, and this product must divide the workload dimension size.
**Why it happens:** `determine_fusion_splits()` already does this check (line 35 of `workload/utils.py`): `divmod(workload.get_dimension_size(dim), int(tile_size * unique_unrollings_dict.get(dim, 1)))`. The filter stage must replicate this.
**How to avoid:** Call `collect_spatial_unrollings(workload, mapping)` to get `unique_spatial_unrollings`, then check `workload_size % (candidate * spatial_unrolling) == 0`.
**Warning signs:** `determine_fusion_splits()` raises `AssertionError` during tiling generation when a candidate that passed the filter is selected.

### Pitfall 3: generate_steady_state_iteration_spaces() stdout Side Effect
**What goes wrong:** Calling `generate_steady_state_iteration_spaces()` per candidate produces noisy stdout output (line 75: `print(node.name, ssis_dict[node])`).
**Why it happens:** Debug print statement in `_create_steady_state_iteration_spaces()`. If called N times for N candidates, log output multiplies.
**How to avoid:** For Phase 2 (TILE-02 quantities needed for memory check), call `Workload.get_tensor_shape_with_tiling()` directly for the single-tensor memory check — this does not require SSIS. SSIS loop sizes are deferred to Phase 3 when they become CO coefficients.
**Warning signs:** Excessive stdout output during candidate filtering.

### Pitfall 4: Tiled vs. Untiled Workload Confusion
**What goes wrong:** Computing tensor sizes against the pre-tiled workload dimension sizes instead of the tiled ones. After `TilingGenerationStage`, the ctx workload has already had `with_modified_dimension_sizes()` applied.
**Why it happens:** `get_tensor_shape_with_tiling()` takes a `succ_tiling` (the inter-core tiling), not the intra-core tile size. The intra-core tiling is separate.
**How to avoid:** Use `workload.get_tensor_shape_with_tiling(tensor, inter_core_tiling)` where `inter_core_tiling` comes from `workload.get_unique_dims_inter_core_tiling(node, mapping)` — this gives the tensor size on a single core after inter-core splitting. The intra-core tile size (the candidate) then represents how many steady-state slices are processed.
**Warning signs:** Memory check incorrectly passes huge candidates because full workload tensor size is used rather than per-core tensor size.

### Pitfall 5: Empty SearchSpace Not Detected Early
**What goes wrong:** All candidates are filtered out for a dimension but the stage silently yields an empty SearchSpace. The CO stage later fails with an unhelpful error.
**Why it happens:** Aggressive filtering can leave no valid candidates.
**How to avoid:** After building the SearchSpace, assert that every unique dimension has at least one valid candidate. Raise a descriptive `ValueError` if not.
**Warning signs:** CO stage fails with "no variables created" or similar Gurobi error with no upstream warning.

### Pitfall 6: tile_options Not Threaded Through MappingFactory
**What goes wrong:** `MappingFactory._convert_intra_core_tiling_entry()` already takes `tile_options[0]` (line 122). The full list is discarded. CandidateFilterStage cannot reconstruct it from the parsed `Mapping` object.
**Why it happens:** The Mapping data model (`FusedGroup`) only stores scalar tile sizes.
**How to avoid:** Extract `tile_options_raw` from the raw YAML mapping data *before* or *during* MappingParserStage and set it into StageContext as a separate value. Do not modify `FusedGroup` for this phase.
**Warning signs:** CandidateFilterStage only sees single-element candidate lists.

---

## Code Examples

Verified patterns from source code inspection:

### Computing Tensor Size for a Specific Tiling
```python
# Existing: stream/workload/workload.py, get_tensor_shape_with_tiling()
# Existing: stream/workload/tensor.py, Tensor.size_bits()
# Pattern for utility function:

def tensor_size_bits_for_inter_core_tiling(
    workload: Workload,
    tensor: Tensor,
    inter_core_tiling: InterCoreTiling,
) -> int:
    """Size of one per-core tensor slice given inter-core tiling."""
    shape = workload.get_tensor_shape_with_tiling(tensor, inter_core_tiling)
    return tensor.size_bits(shape=shape)
```

### Divisibility Check Matching determine_fusion_splits Logic
```python
# Source: stream/workload/utils.py, determine_fusion_splits(), line 35
# Pattern:

from stream.workload.utils import collect_spatial_unrollings

def is_divisible_candidate(
    workload: Workload,
    mapping: Mapping,
    dim: LayerDim,
    candidate_tile: int,
) -> bool:
    """Check that candidate_tile * spatial_unrolling divides workload dimension size."""
    _, unique_spatial_unrollings = collect_spatial_unrollings(workload, mapping)
    unrollings_dict = dict(unique_spatial_unrollings)
    spatial_unrolling = unrollings_dict.get(dim, 1)
    workload_size = workload.get_dimension_size(dim)
    _, rem = divmod(workload_size, int(candidate_tile * spatial_unrolling))
    return rem == 0
```

### Single-Tensor Memory Check
```python
# Source: stream/hardware/architecture/core.py, Core.get_memory_capacity()
# Source: stream/workload/workload.py, get_unique_dims_inter_core_tiling()

def passes_single_tensor_memory_check(
    workload: Workload,
    mapping: Mapping,
    node: ComputationNode,
    candidate_tile: int,
    dim: LayerDim,
    core: Core,
) -> bool:
    """True if no single tensor on the core exceeds core memory capacity."""
    capacity = core.get_memory_capacity()
    # Build a tiling where the candidate replaces the current tile for this dim
    inter_core_tiling = workload.get_unique_dims_inter_core_tiling(node, mapping)
    for tensor in node.tensors:
        size = workload.get_tensor_shape_with_tiling(tensor, inter_core_tiling)
        if tensor.size_bits(shape=size) > capacity:
            return False
    return True
```

### StageContext Set/Get Pattern
```python
# Source: stream/stages/context.py
# Set in CandidateFilterStage:
ctx.set(search_space=search_space)

# Get in ConstraintOptimizationAllocationStage:
search_space: SearchSpace = ctx.get("search_space")
```

### Extracting tile_options_raw from YAML
```python
# In MappingParserStage or a thin wrapper, BEFORE MappingFactory discards them:
# Source: stream/parser/mapping_parser.py, parse_mapping_data()
# The raw mapping_data is available after MappingValidator normalizes it.
# Extraction pattern:

tile_options_raw: dict[str, list[int]] = {}
for fg in mapping_data.get("fused_groups", []):
    for entry in fg.get("intra_core_tiling", []):
        if "tile_options" in entry:
            tile_options_raw[entry["dim"]] = entry["tile_options"]
        elif "tile" in entry:
            tile_options_raw[entry["dim"]] = [entry["tile"]]
# Keys are strings like "Gemm_Left.D1" — CandidateFilterStage resolves to unique dims
ctx.set(tile_options_raw=tile_options_raw)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed `tile` in fused_group YAML | `tile_options` list in fused_group YAML | Phase 1 (this project) | Validator already accepts list; factory takes `[0]` — Phase 2 threads the rest |
| No MappingValidator tile list support | `tile_options` schema in `SCHEMA_FUSED_GROUP`, with `_validate_positive_tiling_values()` | Phase 1 (this project) | No validator changes needed for Phase 2 |

**Existing infrastructure summary:**
- `MappingValidator.SCHEMA_FUSED_GROUP`: Already validates `tile_options` as `list[int]` with non-empty check and positive-int check. No changes needed.
- `MappingFactory._convert_intra_core_tiling_entry()`: Reads `tile_options[0]` if present. Phase 2 must extract full list *before* factory creates the Mapping.
- `workload.unique_dimensions()`: Returns `(z_dims, dim_values)` — z_dims are the canonical keys for SearchSpace.
- `TilingGenerationStage`: Sets `workload`, `mapping`, `fusion_splits` into ctx. CandidateFilterStage reads these.
- `Stage` abstract base class: `__init__(list_of_callables, ctx)`, `run()` generator, `REQUIRED_FIELDS` tuple — CandidateFilterStage follows this pattern exactly.

---

## Runtime State Inventory

Not applicable. This is a greenfield phase adding new modules. No rename, refactor, or data migration involved.

---

## Open Questions

1. **Where exactly to extract tile_options_raw — in MappingParserStage or a new pre-processing step?**
   - What we know: `MappingParser.parse_mapping_data()` returns the normalized dict before `MappingFactory` processes it. `MappingParserStage` currently only calls `mapping_parser.run()` and sets `mapping` in ctx.
   - What's unclear: Whether to modify `MappingParserStage` to also set `tile_options_raw`, or to have CandidateFilterStage re-read the mapping YAML directly.
   - Recommendation: Modify `MappingParserStage` to also set `tile_options_raw` from `mapping_parser.parse_mapping_data()` before passing to `MappingFactory`. This keeps file reading in one place. Avoids CandidateFilterStage needing a YAML path.

2. **SteadyStateScheduler.generate_ssis() safety for TILE-02 quantities beyond single-tensor size**
   - What we know: TILE-02 requires "SSIS loop sizes, reuse levels" per candidate. `generate_steady_state_iteration_spaces()` has a stdout side effect. The full scheduler path rebuilds the transfer graph.
   - What's unclear: Whether TILE-02 requires these quantities to be computed at filter-stage time, or whether they are only needed when constructing CO constraints in Phase 3.
   - Recommendation: Defer SSIS loop sizes and reuse levels to Phase 3. Phase 2 only needs (a) divisibility and (b) single-tensor memory size for filtering. This avoids the side-effect problem entirely. TILE-02 quantities are stored in local variables at the CO call site (D-02), not in the SearchSpace object.

3. **How to build a mock/stub tiling for tensor size computation in the filter stage?**
   - What we know: `get_tensor_shape_with_tiling()` takes an `InterCoreTiling` (the inter-core split factors). The intra-core tile candidate affects how many times a given per-core tensor is processed, not the per-core tensor size itself.
   - What's unclear: Whether the single-tensor memory check should use the post-inter-core-split tensor size (per-core slice) or the full tensor. The inter-core tiling is already fixed from the mapping.
   - Recommendation: Use the per-core slice size (post inter-core split) since that is what must fit in core memory. Retrieve inter-core tiling from `workload.get_unique_dims_inter_core_tiling(node, mapping)` for each compute node. The candidate tile size does not affect which per-core slice fits in memory at this check level — it affects how many steady-state iterations occur. Simplify: the memory check for single-tensor feasibility is "does the per-core slice of this tensor fit in core memory?", which is independent of the intra-core tile candidate.

---

## Environment Availability

Step 2.6: SKIPPED — phase is pure Python code and configuration changes with no external dependencies beyond the existing project stack.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already installed, in `dev` extras) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]`, `testpaths = ["tests"]` |
| Quick run command | `.venv/bin/pytest tests/unit/test_tile_size_utils.py -x` |
| Full suite command | `.venv/bin/pytest tests/ -m "not slow" -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TILE-01 | SearchSpace.build() accepts a list of candidate tile sizes without error | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_search_space_accepts_candidate_list -x` | Wave 0 |
| TILE-02 | For each (unique_dim, candidate) pair, utility functions return correct tensor sizes, SSIS loop sizes, reuse levels as Python scalars | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_tensor_size_bits_matches_expected -x` | Wave 0 |
| TILE-04 | Non-divisible and memory-infeasible candidates are excluded; valid candidates survive | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_divisibility_filter -x` `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_memory_filter -x` | Wave 0 |
| TILE-04 | Unit tests pass with no Gurobi dependency (import check) | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py -x` with gurobipy mocked | Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/pytest tests/unit/test_tile_size_utils.py -x`
- **Per wave merge:** `.venv/bin/pytest tests/ -m "not slow" -x`
- **Phase gate:** Full non-slow suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/__init__.py` — create unit test subdirectory
- [ ] `tests/unit/test_tile_size_utils.py` — covers TILE-01, TILE-02, TILE-04
- [ ] `tests/unit/conftest.py` — shared fixtures: minimal SwiGLU workload + mapping at small scale (e.g., 16x64x128 with candidates [8, 16, 32])

---

## Sources

### Primary (HIGH confidence)
- `/home/micas/stream_aie/stream/workload/workload.py` — `get_tensor_shape_with_tiling`, `unique_dimensions`, `get_dimension_size`, `with_modified_dimension_sizes`
- `/home/micas/stream_aie/stream/workload/tensor.py` — `Tensor.size_bits(shape)`
- `/home/micas/stream_aie/stream/workload/utils.py` — `determine_fusion_splits`, `generate_steady_state_iteration_spaces`, `collect_spatial_unrollings`
- `/home/micas/stream_aie/stream/workload/steady_state/iteration_space.py` — `SteadyStateIterationSpace`, `IterationVariable`, `reuse_factor()`
- `/home/micas/stream_aie/stream/hardware/architecture/core.py` — `Core.get_memory_capacity()`
- `/home/micas/stream_aie/stream/stages/generation/tiling_generation.py` — Stage pattern, ctx field names
- `/home/micas/stream_aie/stream/stages/context.py` — `StageContext.set()`, `StageContext.get()`
- `/home/micas/stream_aie/stream/stages/stage.py` — `Stage` ABC, `REQUIRED_FIELDS`, `StageCallable` protocol
- `/home/micas/stream_aie/stream/api.py` — `optimize_allocation_co()` stage chain
- `/home/micas/stream_aie/stream/parser/mapping_factory.py` — `_convert_intra_core_tiling_entry()`, `tile_options[0]` extraction
- `/home/micas/stream_aie/stream/parser/mapping_validator.py` — `SCHEMA_FUSED_GROUP`, `tile_options` validation
- `/home/micas/stream_aie/stream/mapping/mapping.py` — `FusedGroup`, `Mapping`, `NodeMapping`
- `/home/micas/stream_aie/stream/inputs/aie/mapping/make_swiglu_mapping_v2.py` — YAML format with `tile_options`
- `/home/micas/stream_aie/pyproject.toml` — pytest config, test markers
- `/home/micas/stream_aie/tests/conftest.py` — `slow` marker registration
- `/home/micas/stream_aie/tests/regression/test_baseline.py` — test patterns to follow

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all primitives verified by direct source inspection; no external library research needed
- Architecture patterns: HIGH — stage pattern, ctx interface, and all upstream APIs verified by source
- Pitfalls: HIGH — each pitfall traced to exact source lines (divisibility arithmetic at utils.py:35; stdout at utils.py:75; tile_options[0] at factory.py:122)
- Open questions: MEDIUM — question 1 (tile_options threading) has a clear recommendation; questions 2 and 3 are design choices whose resolution does not block planning

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable codebase, no fast-moving dependencies)
