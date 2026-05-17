# Testing Patterns

## Overview

The test suite uses pytest with no custom conftest. Tests are organized into four categories: unit tests in `tests/unit/`, integration tests in `tests/integration/`, root-level tests in `tests/` for cross-cutting concerns, and standalone study and verification scripts in `tests/` that are not pytest tests. Backend patching is the key cross-cutting pattern for test isolation when tests need to control which solver backend runs without modifying API call sites.

---

## Test Directory Structure

The repository's test layout is:

`tests/unit/` — Six files testing individual components in isolation. Each file corresponds to a specific subsystem: `test_solver_facade.py` (solver abstraction layer), `test_constraint_selection.py` (ConstraintSelection dataclass), `test_linearization.py` (piecewise linearization for division encoding), `test_ortools_backend.py` (OR-Tools backend-specific behavior), `test_pipeline_threading.py` (API and stage pipeline threading of the constraint_selection parameter), and `test_no_gurobipy_leakage.py` (verifies that gurobipy is not imported when the OR-Tools backend is used).

`tests/integration/` — Two files that run the full pipeline against real YAML and ONNX inputs. `test_constraint_toggles.py` verifies that each constraint group's guard is structurally effective. `test_cross_backend.py` verifies that OR-Tools and Gurobi backends produce equivalent objectives within tolerance.

`tests/` root — Three pytest files for broader concerns: `test_accelerator_ir.py` (parametrized tests for accelerator IR parsing), `test_core_cost_lut_caching.py` (stub-stage caching behavior for core cost LUT), and `test_core_validation.py` (core attribute validation).

`tests/` standalone scripts — Four Python scripts that are not pytest tests: `study_constraint_toggles.py`, `study_constraint_toggles_cross_backend.py`, `study_swiglu_backends.py`, and `verify_backends.py`. These are run directly with Python. See the Study and Verification Scripts section for details.

---

## Unit Tests

Unit tests verify individual components in isolation, using mocks where real dependencies would introduce solver overhead or hardware requirements.

To run all unit tests: `pytest tests/unit/ -x`

### Naming Conventions

Test files are named `test_*.py`. Test functions are named `test_*`. No test classes are used — all tests are module-level functions. Module-level helper functions that are not tests use a single underscore prefix (e.g., `_trivial_model()`, `_run_gemm()`), which prevents pytest from collecting them as tests.

### Common Patterns

Parametrized tests use `@pytest.mark.parametrize` with `pytest.param` entries. Each `pytest.param` entry includes an `id=` string to make test names descriptive in output (e.g., `id="gscip"` instead of the default index-based name).

`MagicMock` from `unittest.mock` is used to stand in for TTA (TransferAndTensorAllocator) instances and pipeline stage components where the full component would require real hardware YAML files or solver initialization.

### Unit Test Files

| File | Purpose |
|------|---------|
| `test_solver_facade.py` | Verifies the SolverModel ABC contract, SolverVar arithmetic, LinExpr operations, and the create_solver factory dispatch |
| `test_constraint_selection.py` | Verifies ConstraintSelection defaults, partial overrides, immutability, and the warning for nonsensical combinations |
| `test_linearization.py` | Verifies the piecewise linearization used by OR-Tools backends for division encoding |
| `test_ortools_backend.py` | Verifies OR-Tools backend construction, variable creation, and solver invocation |
| `test_pipeline_threading.py` | Verifies that constraint_selection flows correctly through the API signature, stage context, and ConstraintOptimizationAllocationStage |
| `test_no_gurobipy_leakage.py` | Verifies that importing or running with OR-Tools backends does not trigger a gurobipy import |

---

## Integration Tests

Integration tests run the full pipeline with real workload inputs from `stream/inputs/`. They verify end-to-end correctness and cross-backend parity rather than individual component behavior.

To run all integration tests: `pytest tests/integration/ -x`

Integration tests require real hardware YAML files (e.g., `stream/inputs/aie/hardware/whole_array_strix.yaml`) and real ONNX workloads (generated in-memory for GEMM and SwiGLU by `make_gemm_workload()` and `make_swiglu_workload()`).

### test_constraint_toggles.py — Infeasibility-Flip Pattern

This test file verifies that each of the four constraint groups is structurally effective: when a hardware resource limit is set to exactly the minimum value that makes the problem infeasible under the constraint, enabling the constraint produces a `RuntimeError` and disabling it produces a successful solve. This pattern is called the infeasibility-flip test.

For each constraint group, the test creates a tight hardware configuration (e.g., memory capacity set to a value that the GEMM workload cannot satisfy), runs the optimizer twice — once with the constraint enabled, once with it disabled — and asserts the expected outcomes.

### test_cross_backend.py — Cross-Backend Parity

This test file verifies that OR-Tools backends (GSCIP, HiGHS) produce objective values within 1% of the established Gurobi baseline. The baseline objectives for GEMM (48,730,630) and SwiGLU (9,396,485) were verified during Phase 1 development. Tests in this file use backend patching to inject OR-Tools solvers while keeping the rest of the pipeline identical to a Gurobi run.

---

## Backend Patching

Backend patching is the cross-cutting technique for controlling which solver backend runs in tests without modifying API call sites or passing backend arguments through the test setup.

### The Dual-Target Problem

Both `ComputeAllocator` (in `stream/opt/allocation/constraint_optimization/allocation.py`) and `TransferAndTensorAllocator` (in `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py`) import `create_solver` from `stream.opt.solver` into their own module namespace at import time. Because Python's `unittest.mock.patch` replaces the name in the target namespace rather than the definition in the source module, patching the source (`stream.opt.solver.create_solver`) does not affect the already-imported names in the allocator modules. Two separate patch targets are required:

- `stream.opt.allocation.constraint_optimization.allocation.create_solver`
- `stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation.create_solver`

Both patches must be active simultaneously during a test run to ensure both allocators use the injected backend.

### The Factory Replacement Pattern

Tests replace `create_solver` with a factory function that ignores the requested `SolverBackend` enum value and always returns an `ORToolsBackend` instance with the desired solver type. The factory is constructed to accept the same positional and keyword arguments as `create_solver` so that the allocator call sites need no changes.

### Gurobi License Check

When the pipeline is told to use a Gurobi backend (either directly or through the replaced factory), `stream.api._sanity_check_gurobi_license` is called at startup. Tests that run on machines without a Gurobi license patch this function as a third target:

- `stream.api._sanity_check_gurobi_license`

All three patches can be stacked using `with patch(...) as ..., patch(...) as ..., patch(...) as ...:` blocks or nested `with` statements.

---

## Study and Verification Scripts

These four scripts are standalone Python programs, not pytest tests. They are designed for exploratory analysis and verification of backend behavior across larger input spaces.

| Script | Purpose | Output | How to Run |
|--------|---------|--------|------------|
| `study_constraint_toggles.py` | Enumerates all 16 combinations of the four boolean constraint groups, runs the optimizer for each, and compares objectives and solve times | Terminal table with objectives and solve times per combination; matplotlib bar charts saved to `outputs/` | `PYTHONPATH=. python tests/study_constraint_toggles.py --workload gemm` |
| `study_constraint_toggles_cross_backend.py` | Loads per-backend YAML result files from prior `study_constraint_toggles.py` runs and produces a cross-backend comparison | Terminal comparison table; grouped bar charts for objectives and solve times; a heatmap of objective deltas relative to each backend's all-enabled baseline | `python tests/study_constraint_toggles_cross_backend.py --results path/to/gurobi.yaml path/to/gscip.yaml` |
| `study_swiglu_backends.py` | Compares Gurobi, GSCIP, and HiGHS across multiple SwiGLU workload configurations (varying sequence lengths and tile sizes) | Results YAML file and comparison plots | `PYTHONPATH=. python tests/study_swiglu_backends.py` |
| `verify_backends.py` | Runs two specified backends on the same workload and compares their objective values within a configurable tolerance | Terminal comparison table; exits with code 0 on PASS and 1 on FAIL | `python tests/verify_backends.py --workload gemm` |

---

## Adding New Tests

### Adding a New Unit Test

Create a new file under `tests/unit/test_*.py`. Use a module-level docstring to describe which requirements or behaviors the file covers. Prefix helper functions that should not be collected by pytest with a single underscore. Use `@pytest.mark.parametrize` with `pytest.param(value, id="descriptive-name")` for parametrized tests to keep output readable. Mock TTA or stage components with `MagicMock` from `unittest.mock` rather than constructing real pipeline stages.

### Adding a New Integration Test

Place the file in `tests/integration/test_*.py`. Use the real workload helpers (`make_gemm_workload`, `make_swiglu_workload`) rather than hardcoded file paths. Apply backend patching using the dual-target pattern if the test needs to control which solver backend runs. For infeasibility-flip tests, follow the pattern in `test_constraint_toggles.py`: set a tight hardware limit, run with the constraint enabled (expect `RuntimeError`), run again with it disabled (expect success).

### Adding a New Constraint Group Test

If a new constraint group is added to `ConstraintSelection`, the infeasibility-flip pattern in `test_constraint_toggles.py` is the reference implementation. Identify a hardware parameter that the new constraint bounds, set it to a value that is only tight when the constraint is active, and write a parametrized test that asserts infeasibility when enabled and feasibility when disabled. Update the `ConstraintSelection` instantiation in test helpers to include the new field.

---

## See Also

- `.claude/skills/optimization/solver-backends.md` — `SolverBackend` enum values, `ORToolsBackend` and `GurobiBackend` construction details for the factory replacement pattern
- `.claude/skills/optimization/constraint-selection.md` — `ConstraintSelection` dataclass fields and constraint group semantics for writing infeasibility-flip tests
- `.claude/skills/api-testing/api-reference.md` — `optimize_allocation_co()` and `optimize_mapping()` signatures that the integration tests exercise
