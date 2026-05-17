# API Reference

## Overview

stream_aie exposes two programmatic entry points: `optimize_allocation_co()` runs the constraint optimization (CO) pipeline for a single, fixed mapping, and `optimize_mapping()` runs the design space exploration (DSE) pipeline that generates multiple candidate mappings and evaluates each through the CO pipeline. Both functions are importable from `stream.api`. Ten CLI scripts wrap these entry points for common workloads and hardware configurations.

---

## optimize_allocation_co()

Runs the full CO pipeline for a single hardware/workload/mapping triple. The pipeline stages are: parse accelerator, parse ONNX workload, parse mapping, generate tiling, estimate core costs, run MILP allocation (ComputeAllocator then TransferAndTensorAllocator), and estimate memory accesses. Optionally, an AIE code generation stage can be prepended.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `hardware` | `str` | required | Path to the hardware definition YAML file |
| `workload` | `str` | required | Path to the ONNX workload file |
| `mapping` | `str` | required | Path to the mapping YAML file |
| `experiment_id` | `str` | required | Subdirectory name under `output_path` for this run's output files |
| `output_path` | `str` | required | Root output directory; created if it does not exist |
| `skip_if_exists` | `bool` | `False` | If `True` and `{output_path}/{experiment_id}/ctx.pickle` exists, load and return it without re-running |
| `temporal_mapping_type` | `str` | `"uneven"` | Temporal mapping strategy for ZigZag tiling; accepts `"uneven"` or `"even"` |
| `enable_codegen` | `bool` | `False` | If `True`, prepend the AIE code generation stage to the pipeline |
| `trace_size` | `int` | `1048576` | Trace buffer size in bytes, passed to the code generation stage |
| `nb_cols_to_use` | `int` | `4` | Number of AIE array columns to use during MILP allocation |
| `npu` | `str` | `"npu2"` | NPU target variant, passed to the code generation stage |
| `backend` | `str` | `"ortools_gscip"` | Solver backend name; case-insensitive; one of `gurobi`, `ortools_gscip`, `ortools_highs`, `ortools_gurobi` |
| `constraint_selection` | `ConstraintSelection \| None` | `None` | Which hardware resource constraint groups to enforce; `None` enables all four groups |

### Return Type

Returns a `StageContext` object. The optimization result is accessible via `ctx.get("scheduler")`, which returns a `SteadyStateScheduler` instance containing the solved allocation. The updated mapping is at `ctx.get("mapping")` and the workload at `ctx.get("workload")`. `SolveStats` is available by calling `model.solve_stats()` on the solver model inside the allocator but is not persisted to the `StageContext`. If `enable_codegen=True`, the generated MLIR module is accessible via `ctx.get("module")`.

### Backend Validation

When `backend` is `gurobi` or `ortools_gurobi`, the function calls `GurobiBackend.check_license()` before running the pipeline. If a Gurobi license is not available, this raises immediately with a descriptive error, rather than failing partway through the pipeline. The default backend `ortools_gscip` requires no license.

### Caching Behavior

When `skip_if_exists=True`, the function checks for `{output_path}/{experiment_id}/ctx.pickle` before running. If the file exists, it is loaded with `pickle_load` and returned directly, skipping all pipeline stages. This is useful for re-running downstream analysis or plots without re-solving the MILP.

---

## optimize_mapping()

Runs the DSE pipeline. First, the pipeline generates up to `max_nb_mappings` candidate mappings from the hardware and workload description. Then, for each mapping, the full CO pipeline runs (parse, tile, cost, MILP, memory estimation). The function returns the context from the final mapping's pipeline run.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `hardware` | `str` | required | Path to the hardware definition YAML file |
| `workload` | `str` | required | Path to the ONNX workload file |
| `experiment_id` | `str` | required | Subdirectory name under `output_path` for this run's output files |
| `output_path` | `str` | required | Root output directory; created if it does not exist |
| `max_nb_mappings` | `int` | `20` | Maximum number of candidate mappings to generate and evaluate |
| `skip_if_exists` | `bool` | `False` | If `True` and `{output_path}/{experiment_id}/ctx.pickle` exists, load and return it without re-running |
| `temporal_mapping_type` | `str` | `"uneven"` | Temporal mapping strategy for ZigZag tiling; accepts `"uneven"` or `"even"` |
| `enable_codegen` | `bool` | `False` | If `True`, prepend the AIE code generation stage to the pipeline |
| `trace_size` | `int` | `1048576` | Trace buffer size in bytes, passed to the code generation stage |
| `nb_cols_to_use` | `int` | `8` | Number of AIE array columns to use during MILP allocation |
| `nb_rows_to_use` | `int` | `4` | Number of AIE array rows, passed to the mapping generator for shape-aware tiling |
| `seq_len_tile_size` | `int` | `32` | Tile size for the sequence length dimension (SwiGLU workloads) |
| `embedding_tile_size` | `int` | `128` | Tile size for the embedding dimension (SwiGLU workloads) |
| `hidden_tile_size` | `int` | `64` | Tile size for the hidden dimension (SwiGLU workloads) |
| `last_gemm_down` | `bool` | `False` | Whether to include the final down-projection GEMM in the SwiGLU workload |
| `npu` | `str` | `"npu2"` | NPU target variant, passed to the code generation stage |
| `nb_workers` | `int` | `1` | Number of parallel workers for mapping generation; values greater than 1 use `MappingGenerationMultiThreadedStage` |
| `backend` | `str` | `"ortools_gscip"` | Solver backend name; case-insensitive; one of `gurobi`, `ortools_gscip`, `ortools_highs`, `ortools_gurobi` |
| `constraint_selection` | `ConstraintSelection \| None` | `None` | Which hardware resource constraint groups to enforce; `None` enables all four groups |

### Return Type

Returns a `StageContext` object from the last mapping's pipeline run.

### Multi-Threaded Mapping Generation

When `nb_workers > 1`, the function substitutes `MappingGenerationMultiThreadedStage` for the single-threaded `MappingGenerationStage`, and passes `max_workers=nb_workers` into the stage context. This parallelizes only the mapping generation phase; the CO pipeline for each mapping still runs sequentially.

---

## SolveStats

`SolveStats` is a frozen dataclass returned by `SolverModel.solve_stats()` after `optimize()` completes. It is accessible from the allocation stage output in the `StageContext`. Fields that are not available for a given backend are set to `None`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `backend` | `str` | Backend class name, e.g. `"GUROBI"` or `"ORTOOLS"` |
| `solver` | `str` | Underlying solver name, e.g. `"gurobi"`, `"gscip"`, `"highs"` |
| `status` | `str` | Solve status string, e.g. `"OPTIMAL"`, `"INFEASIBLE"`, `"TIME_LIMIT"` |
| `objective` | `float \| None` | Objective value of the best solution found, or `None` if no solution was found |
| `solve_time_s` | `float` | Wall-clock solve time in seconds |
| `mip_gap` | `float \| None` | Relative MIP gap at termination; `None` for OR-Tools backends |
| `node_count` | `int \| None` | Number of branch-and-bound nodes explored; `None` for OR-Tools backends |
| `iteration_count` | `int \| None` | Number of simplex iterations; `None` for OR-Tools backends |

The `mip_gap`, `node_count`, and `iteration_count` fields are populated by the Gurobi backend, which exposes detailed solve statistics through the Gurobi callback. OR-Tools backends return `None` for these fields because the MathOpt API does not expose them at the same level of detail.

---

## Workload Utilities

### determine_fusion_cut_points()

**Source:** `stream/workload/workload.py`

Module-level function that analyzes a parsed `Workload` graph and returns a list of node names where
fusion group boundaries should be placed. Used by `GenericMappingGenerationStage` to split large workloads
into manageable groups before generating per-group mappings.

**Heuristics applied (in topological order):**
1. **MaxPool nodes:** End the front-end group (e.g., Conv1 -> Relu -> MaxPool).
2. **Relu nodes following Add:** Each Add node whose sole ComputationNode successor is a Relu marks
   a residual block boundary. The Relu is the cut point (split occurs AFTER the Relu).

**Signature:**

```python
def determine_fusion_cut_points(workload: Workload) -> list[str]:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `workload` | `Workload` | Parsed workload graph |
| **Returns** | `list[str]` | Node names to pass to `split_fusion_groups(cut_points=...)` |

**Example (ResNet18):** Returns 9 cut points (1 MaxPool + 8 Relu), producing 11 groups when combined with the existing Flatten FusionEdge split.

### split_fusion_groups() (extended)

**Source:** `stream/workload/workload.py` (method on `Workload`)

Extended to accept an optional `cut_points` parameter. When provided, the method splits at both
FusionEdge boundaries (existing behavior) AND at the specified cut-point nodes. Each cut-point node
stays in the preceding group; its output tensor becomes an OutEdge/InEdge boundary pair.

**Signature:**

```python
def split_fusion_groups(self, cut_points: list[str] | None = None) -> list[Workload]:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cut_points` | `list[str] \| None` | `None` | Node names after which to split. `None` preserves original FusionEdge-only behavior. |
| **Returns** | `list[Workload]` | | Sub-workloads, one per group |

**Backward compatibility:** When `cut_points` is `None`, behavior is identical to the original (FusionEdge-only splits).

---

## CLI Scripts

All CLI scripts live at the repository root. Each calls either `optimize_allocation_co()` or `optimize_mapping()` with pre-configured hardware, workload, and mapping paths for a specific target.

| Script | Category | Purpose | Key Flags |
|--------|----------|---------|-----------|
| `main_gemm.py` | CO | Runs CO pipeline on a parameterized GEMM workload (M, K, N dimensions, tile sizes, data types) with AIE codegen enabled | `--M`, `--N`, `--K`, `--m`, `--k`, `--n`, `--in_dtype`, `--out_dtype`, `--rows`, `--cols`, `--npu`, `--backend`, `--disable-constraints` |
| `main_aie_co.py` | CO | Runs CO pipeline on a fixed 1x1 convolution AIE workload; hardcoded paths and mode | None (config embedded in script) |
| `main_stream_co.py` | CO | Runs CO pipeline on a ResNet-18 workload with a quad-core TPU-like accelerator | None (config embedded in script) |
| `main_aie_ga.py` | GA | Runs genetic algorithm allocation on a fixed AIE 1x1 convolution workload; produces schedule and memory usage plots | None (config embedded in script) |
| `main_stream_ga.py` | GA | Runs genetic algorithm allocation on ResNet-18 with a quad-core accelerator; produces memory plot and Perfetto trace | None (config embedded in script) |
| `main_swiglu.py` | DSE | Runs CO pipeline (single mapping) on a parameterized SwiGLU workload with AIE codegen enabled | `--seq_len`, `--embedding_dim`, `--hidden_dim`, `--in_dtype`, `--out_dtype`, `--rows`, `--cols`, `--npu`, `--seq_len_tile_size`, `--embedding_tile_size`, `--hidden_tile_size`, `--no_last_gemm_down`, `--backend`, `--disable-constraints` |
| `main_swiglu_dse.py` | DSE | Runs DSE pipeline (`optimize_mapping`) over all combinations of tile size values provided for seq_len, embedding, and hidden dimensions | `--seq_len`, `--embedding_dim`, `--hidden_dim`, `--seq_len_tile_size` (list), `--embedding_tile_size` (list), `--hidden_tile_size` (list), `--rows`, `--cols`, `--backend`, `--disable-constraints` |
| `main_swiglu_dse_single.py` | DSE | Runs CO pipeline on SwiGLU with a single hardcoded mapping file from a prior DSE run | `--seq_len`, `--embedding_dim`, `--hidden_dim`, `--rows`, `--cols`, `--seq_len_tile_size`, `--embedding_tile_size`, `--hidden_tile_size`, `--no_last_gemm_down`, `--disable-constraints` |
| `main_gemm_manual.py` | Specialized | Manual MLIR generation for GEMM without using the CO pipeline; uses xDSL compiler transforms directly | `--m_size`, `--n_size`, `--k_size` (hardcoded in script body) |
| `main_aie_codegen_conv2d.py` | Specialized | Runs CO pipeline with AIE codegen on a parameterized 2D convolution workload using a single AIE tile | `--height` |

---

## Common Flags

All CO and DSE scripts that accept CLI arguments support two shared flags for controlling backend selection and constraint toggling.

### --backend

Selects the MILP solver backend. Choices: `gurobi`, `ortools_gscip`, `ortools_highs`, `ortools_gurobi`. Default: `ortools_gscip`. The default is intentionally the license-free OR-Tools GSCIP backend, so scripts work on any machine without a commercial solver license. The `gurobi` and `ortools_gurobi` choices require a Gurobi license and will fail at startup if one is not found.

### --disable-constraints

Selectively disables hardware resource constraint groups in the MILP allocation. Accepts zero or more values from: `memory_capacity`, `object_fifo_depth`, `buffer_descriptors`, `dma_channels`. When one or more groups are disabled, the script constructs a `ConstraintSelection` dataclass with the corresponding fields set to `False`. When the flag is not passed (or passed with no arguments), all four constraint groups remain enabled. Passing `--disable-constraints` with no values is equivalent to passing nothing — the empty set means all constraints remain active.

---

## See Also

- `.claude/skills/pipeline/pipeline-stages.md` — How API calls trigger and sequence the stage pipeline, stage-by-stage data flow
- `.claude/skills/optimization/solver-backends.md` — `SolverBackend` enum values, `ORToolsBackend` and `GurobiBackend` details, linearization differences
- `.claude/skills/optimization/constraint-selection.md` — `ConstraintSelection` dataclass fields, `--disable-constraints` toggle mapping, constraint group semantics
