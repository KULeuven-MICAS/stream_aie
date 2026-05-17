# Stream AIE

Stream AIE is a design-space exploration and constraint optimization framework for AMD AIE (AI Engine) accelerators, developed as part of the TETRA project at KU Leuven MICAS. It takes ONNX workload graphs and hardware descriptions as inputs and produces optimal tensor placement and transfer-path allocation via MILP (Mixed-Integer Linear Programming).

The framework runs a pipeline of stages: parse hardware/workload/mapping, generate tilings, estimate costs, run MILP allocation, estimate memory. Two MILP allocators handle the core optimization: `ComputeAllocator` assigns computation nodes to cores, and `TransferAndTensorAllocator` decides tensor placement and routing paths. The solver abstraction supports Gurobi and OR-Tools backends (GSCIP and HiGHS) behind a unified API.

For deep dives on specific subsystems, see `.claude/skills/`. Each group has a `SKILL.md` describing when to load it.

## Directory Structure

```
stream_aie/
├── main_gemm.py                        # CLI: GEMM workload CO allocation + AIE codegen
├── main_swiglu.py                      # CLI: SwiGLU workload CO allocation + AIE codegen
├── main_swiglu_dse_single.py           # CLI: Single-mapping SwiGLU DSE evaluation
├── main_swiglu_dse.py                  # CLI: Multi-mapping SwiGLU DSE exploration
├── main_aie_co.py                      # CLI: Additional CO workload variant
├── main_stream_co.py                   # CLI: Stream CO workload variant
├── main_aie_codegen_conv2d.py          # CLI: Conv2D AIE codegen
├── main_gemm_manual.py                 # CLI: Manual GEMM configuration
├── stream/
│   ├── api.py                          # Public API
│   ├── datatypes.py                    # LayerDim, InterCoreTiling type aliases
│   ├── compiler/                       # MLIR code generation (dialects, kernels, transforms)
│   ├── cost_model/                     # Scheduling, cost evaluation, communication
│   ├── hardware/                       # Hardware architecture model
│   ├── inputs/                         # Hardware YAML, mapping YAML, workload generators
│   ├── mapping/                        # Mapping data model and DSE variant generation
│   ├── opt/
│   │   ├── allocation/
│   │   │   └── constraint_optimization/   # MILP-based allocation
│   │   │   │   ├── allocation.py              # ComputeAllocator
│   │   │   │   ├── transfer_and_tensor_allocation.py  # TransferAndTensorAllocator
│   │   │   │   ├── context.py                 # ConstraintContext, NamespaceConstraints
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

**CLI scripts (repo root):**
- `main_gemm.py` -- GEMM workload: constraint optimization allocation + optional AIE codegen
- `main_swiglu.py` -- SwiGLU workload: constraint optimization allocation + optional AIE codegen
- `main_swiglu_dse_single.py` -- Single-mapping SwiGLU design space evaluation
- `main_swiglu_dse.py` -- Multi-mapping SwiGLU design space exploration
- `main_aie_co.py`, `main_stream_co.py` -- Additional workload variants
- `main_aie_codegen_conv2d.py` -- Conv2D AIE codegen
- `main_gemm_manual.py` -- Manual GEMM configuration

**Public API (`stream/api.py`):**
- `optimize_allocation_co(hardware, workload, mapping, ...)` -- Full CO pipeline: parse -> tile -> cost -> MILP -> memory estimation
- `optimize_mapping(hardware, workload, ...)` -- DSE pipeline: enumerate mappings, run CO for each

## Coding Conventions

- **Formatter:** ruff-format (pre-commit hook); line length 120
- **Python target:** 3.11+; use `X | Y` union syntax, built-in generics (`list[X]`, `dict[K, V]`)
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
