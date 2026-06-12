# Stream AIE

Stream AIE is a design-space exploration and constraint optimization framework for AMD AIE (AI Engine) accelerators, developed as part of the TETRA project at KU Leuven MICAS. It takes ONNX workload graphs and hardware descriptions as inputs and produces optimal tensor placement and transfer-path allocation via MILP (Mixed-Integer Linear Programming).

The framework runs a pipeline of stages: parse hardware/workload/mapping, generate tilings, estimate costs, run MILP allocation, estimate memory. Computation nodes are placed on cores according to the mapping, and the `TransferAndTensorAllocator` MILP decides tensor placement and routing paths. The solver abstraction supports Gurobi and OR-Tools backends (GSCIP and HiGHS) behind a unified API.

For deep dives on specific subsystems, see `.claude/skills/`. Each group has a `SKILL.md` describing when to load it.

## Directory Structure

```
stream_aie/
├── scripts/                            # CLI entry points (main_*.py) and analysis/ (plot_*, postprocess_*)
├── stream/
│   ├── api.py                          # Public API
│   ├── datatypes.py                    # LayerDim, InterCoreTiling type aliases
│   ├── compiler/                       # MLIR code generation (dialects, kernels, transforms)
│   ├── cost_model/                     # Scheduling, cost evaluation, communication
│   ├── hardware/                       # Hardware architecture model
│   ├── inputs/                         # Hardware YAML, mapping YAML, workload generators
│   ├── ir/                             # Typed IR models (WorkloadIR, AllocationIR, AcceleratorIR)
│   ├── mapping/                        # Mapping data model and DSE variant generation
│   ├── mcp/                            # MCP server (stream-aie) exposing CO jobs to AI agents
│   ├── opt/
│   │   ├── allocation/
│   │   │   └── constraint_optimization/   # MILP-based allocation
│   │   │   │   ├── transfer_and_tensor_allocation.py  # TransferAndTensorAllocator
│   │   │   │   ├── context.py                 # TransferAndTensorContext, NamespaceConstraints
│   │   │   │   └── config.py                  # ConstraintOptStageConfig, CoreConstraintProfile
│   │   └── solver/                        # Solver facade (SolverModel ABC, backends)
│   ├── parser/                         # ONNX and workload parsing
│   ├── stages/                         # Pipeline stage framework
│   │   ├── stage.py                    # Stage, MainStage, LeafStage, StageCallable
│   │   ├── context.py                  # StageContext (shared mutable state)
│   │   ├── parsing/                    # Accelerator, mapping, ONNX parser stages
│   │   ├── generation/                 # Tiling and mapping generation stages
│   │   ├── estimation/                 # Core cost and memory estimation stages
│   │   └── allocation/                 # CO allocation stages
│   ├── visualization/                  # Plotting and trace export
│   └── workload/                       # Workload DAG representation and steady-state model
├── tests/                              # pytest test suite
└── .claude/skills/                     # AI-agent skill groups (deep-dive docs)
```

## Key Entry Points

**CLI scripts (`scripts/`, run from the repo root):**
- `scripts/main_gemm.py` -- GEMM workload: constraint optimization allocation + optional AIE codegen
- `scripts/main_swiglu.py` -- SwiGLU workload: constraint optimization allocation + optional AIE codegen
- `scripts/main_swiglu_dse_single.py` -- Single-mapping SwiGLU design space evaluation
- `scripts/main_swiglu_dse.py` -- Multi-mapping SwiGLU design space exploration
- `scripts/main_stream_co.py` -- General-purpose CO pipeline for any ONNX workload + hardware (non-AIE); manual or auto-generated mapping
- `scripts/main_aie_co.py` -- CO allocation for a hard-coded single AIE tile workload
- `scripts/main_gemm_codegen.py` -- Direct GEMM AIE MLIR codegen (xDSL transforms, no CO pipeline)
- `scripts/analysis/` -- plotting (`plot_*.py`) and trace post-processing (`postprocess_*.py`) utilities

Scripts import `stream` as an installed package (`pip install -e .`); run them from the repo root so relative input paths (e.g. `stream/inputs/...`) resolve.

**Public API (`stream/api.py`):**
- `optimize_allocation_co_generic(hardware, workload, ...)` -- Primary entry point: auto-generates the mapping, then runs the full CO pipeline (parse -> tile -> cost -> MILP -> memory estimation)
- `optimize_allocation_co_with_mapping(hardware, workload, mapping, ...)` -- CO pipeline with a hand-written mapping (`optimize_allocation_co` is a backward-compatible alias)
- `optimize_mapping(hardware, workload, ...)` -- DSE pipeline: enumerate mappings, run CO for each

## Coding Conventions

- **Formatter:** ruff-format (pre-commit hook); line length 120
- **Python target:** 3.12+; use `X | Y` union syntax, built-in generics (`list[X]`, `dict[K, V]`)
- **Imports:** absolute only (`from stream.stages.stage import Stage`); isort order: stdlib -> third-party -> internal
- **Files:** `snake_case.py`; **Classes:** `PascalCase`; Stage classes end with `Stage`
- **Private methods:** `_single_underscore`; internal MILP helpers: `__double_underscore`
- **Config types:** `@dataclass(frozen=True)` for immutable value objects
- **Linting:** `ruff check` (rules: E, F, W, I, PL, N, UP, B); `fix = true` for safe auto-fixes

## Dev Workflow

- `pre-commit install` -- set up git hooks (ruff check + ruff format)
- `pytest tests/` -- run all tests
- `pytest tests/ -x` -- run tests, stop on first failure
- `ruff check .` -- lint
- `ruff format .` -- format

## Skills

Deep-dive documentation for specific subsystems lives in `.claude/skills/`. Each group has a `SKILL.md` describing when to load it.

- `.claude/skills/optimization/` -- Solver backends (Gurobi, OR-Tools), ConstraintSelection configuration
- `.claude/skills/pipeline/` -- Pipeline stages, StageContext, MainStage/LeafStage execution model
- `.claude/skills/constraints/` -- MILP formulation, TransferAndTensorAllocator, NamespaceConstraints dispatch
- `.claude/skills/api-testing/` -- Public API reference, CLI flags, testing patterns and conventions
- `.claude/skills/hardware/` -- Hardware core model (roles, namespaces, AIE/TPU-like examples), per-core performance estimation
- `.claude/skills/ir/` -- IR models (WorkloadIR, AllocationIR, AcceleratorIR), JSON serialization and persona views
