# Pipeline Stages

## Overview

The TETRA pipeline processes a hardware description, ONNX workload, and mapping configuration through a
sequence of stages. Each stage reads from and writes to a shared `StageContext` (a mutable dictionary
wrapper). The pipeline is assembled in `stream/api.py`, where `optimize_allocation_co` (single-mapping
constraint optimization) and `optimize_mapping` (design-space exploration) wire stages in a specific order.
Stages form a chain: each stage validates its required context keys, performs its work, updates the context,
and then delegates execution to the next stage in the chain. The final stage (a leaf stage) yields the
context directly, unwinding the generator chain back up to the caller.

---

## Pipeline Flow Diagram

The `optimize_allocation_co` pipeline runs the following stages in sequence:

    Hardware YAML
    Workload ONNX
    Mapping YAML
         |
         v
    +---------------------------+
    |   AcceleratorParserStage  |  Parses hardware YAML -> Accelerator object
    +---------------------------+
         |
         v
    +---------------------------+
    |  ONNXModelParserStage     |  Parses ONNX file -> onnx_model, workload
    +---------------------------+
         |
         v
    +---------------------------+
    |   MappingParserStage      |  Parses mapping YAML -> mapping
    +---------------------------+
         |
         v
    +---------------------------+
    |  TilingGenerationStage    |  Tiles workload -> tiled workload, mapping, fusion_splits
    +---------------------------+
         |
         v
    +---------------------------+
    | CoreCostEstimationStage   |  Estimates per-core costs -> cost_lut
    +---------------------------+
         |
         v
    +---------------------------+
    | ConstraintOptimization-   |  MILP allocation -> workload, mapping, scheduler
    | AllocationStage           |
    +---------------------------+
         |
         v
    +---------------------------+
    | MemoryAccessesEstimation- |  Estimates memory accesses -> memory_accesses (LEAF)
    | Stage                     |
    +---------------------------+
         |
         v
      StageContext result

The `optimize_mapping` pipeline inserts `MappingGenerationStage` (or its multi-threaded variant
`MappingGenerationMultiThreadedStage`) between `ONNXModelParserStage` and `MappingParserStage`.
`MappingGenerationStage` wraps the remaining downstream stages (MappingParser through
MemoryAccessesEstimation) and acts as an outer loop: it generates mapping variants, runs the full
downstream pipeline for each variant, and yields only the best result (lowest latency). When `nb_workers >
1`, the multi-threaded variant evaluates variants in parallel using a `ThreadPoolExecutor`, then re-runs
the winner single-threaded to produce a clean context.

---

## Stage Reference

Stages are documented here in execution order, grouped by their conceptual role.

### Parsing Stages

#### AcceleratorParserStage

**Source:** `stream/stages/parsing/accelerator_parser.py`

Parses a hardware YAML file into an `Accelerator` object. The stage validates the YAML against the
accelerator schema using `AcceleratorValidator` and then constructs the hardware model via
`AcceleratorFactory`. If the context already contains an `Accelerator` object (rather than a string path),
the stage passes it through unchanged without re-parsing. This makes it safe to call the pipeline
programmatically with a pre-constructed accelerator.

| Attribute           | Value                                              |
|---------------------|----------------------------------------------------|
| REQUIRED_FIELDS     | `accelerator`                                      |
| Context reads       | `accelerator` (str path or Accelerator object)     |
| Context writes      | `accelerator` (Accelerator object)                 |

#### ONNXModelParserStage

**Source:** `stream/stages/parsing/onnx_model_parser.py`

Parses an ONNX model file into the internal `Workload` graph representation. Uses `ONNXModelParser` to
extract the computation graph and build operator nodes. Also generates a PNG visualization of the workload
graph and saves it under `output_path/workload_graph.png`. The ONNX model object itself is also stored in
context for downstream inspection.

| Attribute           | Value                                              |
|---------------------|----------------------------------------------------|
| REQUIRED_FIELDS     | `workload_path`, `output_path`                     |
| Context reads       | `workload_path`, `output_path`                     |
| Context writes      | `onnx_model`, `workload`                           |

#### MappingParserStage

**Source:** `stream/stages/parsing/mapping_parser.py`

Parses a mapping YAML file that defines how workload layers map to hardware cores: spatial unrollings, tile
sizes, and resource allocation. Requires both the parsed `Accelerator` and `Workload` to be present in
context in order to resolve layer-to-core assignments. Uses `MappingParser` which validates consistency
between the mapping specification, the hardware topology, and the workload graph.

| Attribute           | Value                                              |
|---------------------|----------------------------------------------------|
| REQUIRED_FIELDS     | `accelerator`, `workload`, `mapping_path`          |
| Context reads       | `accelerator`, `workload`, `mapping_path`          |
| Context writes      | `mapping`                                          |

---

### Generation Stages

#### TilingGenerationStage

**Source:** `stream/stages/generation/tiling_generation.py`

Determines fusion split factors (how many tiles the pipeline uses for inter-layer fusion), substitutes the
original loop ranges with smaller tiled sizes, and generates steady-state iteration spaces. The stage calls
`determine_fusion_splits` on the workload and mapping to identify the split dimension and factor. It then
produces a `tiled_workload` (new `Workload` with modified dimension sizes) and a `tiled_mapping` (updated
`Mapping` reflecting the tiled workload). The tiled versions overwrite the originals in the context.
A visualization of the tiled workload is saved to `output_path/tiled_workload.png`.

| Attribute           | Value                                                         |
|---------------------|---------------------------------------------------------------|
| REQUIRED_FIELDS     | `workload`, `mapping`, `output_path`                          |
| Context reads       | `workload`, `mapping`, `output_path`                          |
| Context writes      | `workload` (tiled), `mapping` (updated), `fusion_splits`      |

#### MappingGenerationStage

**Source:** `stream/stages/generation/mapping_generation.py`
**Multi-threaded variant:** `stream/stages/generation/mapping_generation_multi.py`

Enumerates mapping variants for design-space exploration (DSE). A `MappingGenerator` is constructed from
the accelerator, workload, and optional tile size parameters. For each generated mapping variant, the stage
saves a mapping YAML file, sets the per-variant context keys (`workload`, `mapping_path`, `output_path`),
and runs the downstream pipeline (MappingParser through MemoryAccessesEstimation) to evaluate the mapping.
The variant with the lowest total latency is retained; all others are discarded. Only one `StageContext` is
yielded: the best result.

The multi-threaded variant (`MappingGenerationMultiThreadedStage`, used when `nb_workers > 1`) evaluates
variants in parallel using a `ThreadPoolExecutor` with backpressure. After finding the best variant in
parallel, it re-runs that variant single-threaded to produce a deterministic context (parallel contexts may
have interleaved writes). The re-run is required because concurrent stage execution can leave the context in
an inconsistent state.

| Attribute           | Value                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------|
| REQUIRED_FIELDS     | `accelerator`, `workload`                                                                      |
| Context reads       | `accelerator`, `workload`, `output_path`; optional: `seq_len_tile_size`, `embedding_tile_size`, `hidden_tile_size`, `last_gemm_down`, `max_nb_mappings`, `nb_rows_to_use`, `nb_cols_to_use` |
| Context writes      | `workload`, `mapping_path`, `output_path` (per variant, then best result)                      |

#### GenericMappingGenerationStage

**Source:** `stream/stages/generation/generic_mapping_generation.py`

Generates per-fusion-group mapping YAMLs from a workload and accelerator pair, using automatic core selection
and tiling inference. Before generating mappings, the stage calls `determine_fusion_cut_points(workload)` to
identify residual block boundaries (Add+Relu patterns and MaxPool front-end boundaries). These cut points are
passed to `GenericMappingGenerator.generate_all_groups(cut_points=...)`, which in turn calls
`split_fusion_groups(cut_points=...)` to split the workload into bounded groups.

For ResNet18, this produces 11 groups (1 front-end, 8 residual blocks, 1 post-residual, 1 Gemm tail).
Each group receives an independently generated and validated mapping YAML.

| Attribute           | Value                                              |
|---------------------|----------------------------------------------------|
| REQUIRED_FIELDS     | `accelerator`, `workload`, `output_path`           |
| Context reads       | `accelerator`, `workload`, `output_path`           |
| Context writes      | `group_mapping_paths`, `sub_workloads`             |
| Delegates to        | `FusionGroupIterationStage` (next in list_of_callables) |

---

### Estimation Stages

#### CoreCostEstimationStage

**Source:** `stream/stages/estimation/core_cost_estimation.py`

Computes per-core cost entries for each valid node-core allocation. Maintains a `CoreCostLUT` (lookup
table) that is cached to disk so repeated runs can reuse previously computed costs. For each computation
node in the tiled workload and each valid core it could be assigned to, the stage either reuses a cached
entry, copies from an already-computed equal node/core pair, or runs a fresh cost estimator. The cost
estimator used depends on the core type: `AIECostEstimator` for AIE2 compute cores and
`ZigZagCostEstimator` for other core types. The resulting `CoreCostLUT` feeds directly into the allocation
stage.

| Attribute           | Value                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------|
| REQUIRED_FIELDS     | `workload`, `accelerator`, `mapping`, `loma_lpf_limit`, `output_path`, `temporal_mapping_type` |
| Context reads       | `workload`, `accelerator`, `mapping`, `loma_lpf_limit`, `output_path`, `temporal_mapping_type` |
| Context writes      | `workload`, `accelerator`, `cost_lut`                                                          |

#### MemoryAccessesEstimationStage

**Source:** `stream/stages/estimation/memory_accesses_estimation.py`

Estimates memory read/write counts for each core and tensor based on the allocation result. This is the
**only LeafStage** in the standard `optimize_allocation_co` pipeline. Unlike all other stages, it does not
delegate to a sub-stage: it yields `self.ctx` directly via `yield from (self.ctx,)` and overrides
`is_leaf()` to return `True`. This terminates the generator delegation chain.

The memory access calculation logic (`calculate_memory_accesses`) is currently disabled (commented out)
and the stage serves as a pipeline terminator, setting `memory_accesses` to an empty
`CoreMemoryAccesses` object. The implementation exists for when this feature is re-enabled.

| Attribute           | Value                                              |
|---------------------|----------------------------------------------------|
| REQUIRED_FIELDS     | `workload`, `accelerator`, `mapping`, `scheduler`, `output_path` |
| Context reads       | `workload`, `accelerator`, `mapping`, `scheduler`, `output_path` |
| Context writes      | `workload`, `accelerator`, `memory_accesses`       |

---

### Allocation Stages

#### ConstraintOptimizationAllocationStage

**Source:** `stream/stages/allocation/constraint_optimization_allocation.py`

Runs MILP-based tensor placement and transfer-path allocation via `SteadyStateScheduler`. The stage reads
the `CoreCostLUT` built by the estimation stage, the fusion splits from the tiling stage, and a solver
configuration (`ConstraintOptStageConfig`). It delegates to the solver facade for the actual MILP
optimization. The solver backend (OR-Tools GSCIP by default) is selected via the `backend` context key.
Constraint groups can be selectively disabled via the `constraint_selection` key (a `ConstraintSelection`
dataclass with four boolean fields); if absent from the context, all constraints default to enabled.

| Attribute           | Value                                                                                              |
|---------------------|----------------------------------------------------------------------------------------------------|
| REQUIRED_FIELDS     | `workload`, `accelerator`, `mapping`, `cost_lut`, `fusion_splits`, `output_path`                   |
| Context reads       | `workload`, `accelerator`, `mapping`, `cost_lut`, `fusion_splits`, `output_path`, `backend` (default `"ORTOOLS_GSCIP"`), `constraint_selection`, `constraint_opt_config` |
| Context writes      | `workload`, `mapping`, `scheduler`                                                                 |

---

## Context Key Flow Table

This table maps each stage to the context keys it reads and the context keys it writes. Stages that
overwrite an existing key are marked with "(updated)" or "(tiled)" to indicate the value is replaced, not
added fresh.

| Stage                                  | Reads from Context                                                              | Writes to Context                             |
|----------------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------|
| AcceleratorParserStage                 | `accelerator` (str path or Accelerator)                                         | `accelerator` (Accelerator object)            |
| ONNXModelParserStage                   | `workload_path`, `output_path`                                                  | `onnx_model`, `workload`                      |
| MappingParserStage                     | `accelerator`, `workload`, `mapping_path`                                       | `mapping`                                     |
| TilingGenerationStage                  | `workload`, `mapping`, `output_path`                                            | `workload` (tiled), `mapping` (updated), `fusion_splits` |
| CoreCostEstimationStage                | `workload`, `accelerator`, `mapping`, `loma_lpf_limit`, `output_path`, `temporal_mapping_type` | `workload`, `accelerator`, `cost_lut` |
| ConstraintOptimizationAllocationStage  | `workload`, `accelerator`, `mapping`, `cost_lut`, `fusion_splits`, `output_path`, `backend`, `constraint_selection` | `workload`, `mapping`, `scheduler` |
| MemoryAccessesEstimationStage          | `workload`, `accelerator`, `mapping`, `scheduler`, `output_path`                | `workload`, `accelerator`, `memory_accesses`  |
| MappingGenerationStage                 | `accelerator`, `workload`, `output_path`, optional tile params                  | `workload`, `mapping_path`, `output_path` (per variant) |
| GenericMappingGenerationStage          | `accelerator`, `workload`, `output_path`                                        | `group_mapping_paths`, `sub_workloads`        |

Note that several stages overwrite keys that earlier stages wrote. For example, `TilingGenerationStage`
replaces the raw `workload` (from `ONNXModelParserStage`) with a tiled version. Later stages
(`CoreCostEstimationStage`, `ConstraintOptimizationAllocationStage`, `MemoryAccessesEstimationStage`)
further update `workload` and `accelerator`. The final context returned to the caller reflects these
accumulated updates.

---

## Two Pipeline Variants

Both pipeline variants are assembled in `stream/api.py` using `MainStage`.

### optimize_allocation_co (single-mapping CO)

Uses a 7-stage linear chain. All stages execute once in sequence. The pipeline terminates at
`MemoryAccessesEstimationStage` (the LeafStage). Used when a mapping has already been chosen and the goal
is to run allocation and cost estimation on that single mapping.

Stage order:
AcceleratorParserStage -> ONNXModelParserStage -> MappingParserStage -> TilingGenerationStage ->
CoreCostEstimationStage -> ConstraintOptimizationAllocationStage -> MemoryAccessesEstimationStage

### optimize_mapping (design-space exploration)

Inserts `MappingGenerationStage` (or `MappingGenerationMultiThreadedStage`) between
`ONNXModelParserStage` and `MappingParserStage`. `MappingGenerationStage` wraps the remaining downstream
stages as its sub-pipeline. For each generated mapping variant, the downstream pipeline
(MappingParser -> TilingGeneration -> CoreCostEstimation -> ConstraintOptimizationAllocation ->
MemoryAccessesEstimation) runs in full. After all variants are evaluated, only the best-latency context is
yielded.

Stage order:
AcceleratorParserStage -> ONNXModelParserStage -> MappingGenerationStage -> [MappingParserStage ->
TilingGenerationStage -> CoreCostEstimationStage -> ConstraintOptimizationAllocationStage ->
MemoryAccessesEstimationStage]

### optimize_allocation_co_generic (auto-mapping CO)

Uses `GenericMappingGenerationStage` in place of `MappingParserStage` to auto-infer mapping from
workload+hardware. The stage calls `determine_fusion_cut_points()` to identify residual block boundaries,
then `GenericMappingGenerator` produces per-group YAMLs. `FusionGroupIterationStage` iterates over groups,
running the inner pipeline (MappingParser -> TilingGeneration -> CoreCostEstimation -> CO Allocation ->
MemoryAccessesEstimation) once per group.

Stage order:
AcceleratorParserStage -> ONNXModelParserStage -> GenericMappingGenerationStage ->
FusionGroupIterationStage -> [MappingParserStage -> TilingGenerationStage -> CoreCostEstimationStage ->
ConstraintOptimizationAllocationStage -> MemoryAccessesEstimationStage]

### Optional Code Generation

All pipeline variants can prepend `AIECodeGenerationStage` when called with `enable_codegen=True`. This stage runs
before `AcceleratorParserStage` and requires the `npu` context key to identify the NPU variant to target.

---

## See also

- `.claude/skills/pipeline/stage-execution.md` -- Execution model details: StageContext interface,
  Stage ABC, MainStage/LeafStage composition, and how to add a new stage.
- `.claude/skills/optimization/` -- Solver backend details used by `ConstraintOptimizationAllocationStage`,
  including `ConstraintSelection`, solver backends, and the MILP formulation.
