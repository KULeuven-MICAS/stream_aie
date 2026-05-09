# Solver Backends

## Overview

The solver abstraction layer lives in `stream/opt/solver/solver.py` (single-module layout). `SolverModel` is an abstract base class (ABC) that defines the interface for all MILP operations — variable creation, constraint addition, objective setting, and solving. `GurobiBackend` and `ORToolsBackend` are the two concrete implementations. The factory function `create_solver()` instantiates the correct backend from a `SolverBackend` enum value, hiding all backend-specific construction from callers.

All TETRA constraint optimization code (allocation.py, transfer_and_tensor_allocation.py, context.py, api.py) works exclusively through the `SolverModel` interface — no direct gurobipy imports appear outside the backend class itself.

---

## SolverBackend Enum

The `SolverBackend` enum lists every supported backend configuration. Pass one of these values to `create_solver()`.

| Value | Backend | Underlying Solver | License |
|-------|---------|-------------------|---------|
| `GUROBI` | `GurobiBackend` | gurobipy | Commercial (academic free) |
| `ORTOOLS_GSCIP` | `ORToolsBackend` | GSCIP | Open-source (bundled) |
| `ORTOOLS_HIGHS` | `ORToolsBackend` | HiGHS | Open-source (bundled) |
| `ORTOOLS_GUROBI` | `ORToolsBackend` | Gurobi via MathOpt | Commercial (academic free) |

---

## SolverModel ABC

`SolverModel` defines the complete public interface for all solver interactions. Backends inherit from this class and must implement every abstract method. Two non-abstract methods — `supports_nonlinear` and `add_genconstr_nl` — have default implementations that signal non-support.

### Class Constants

| Constant | Type | Value | Purpose |
|----------|------|-------|---------|
| `INFINITY` | `float` | backend-specific | Use as the upper bound for unbounded variables |
| `MINIMIZE` | `str` | `"minimize"` | Sense string for `set_objective()` |
| `MAXIMIZE` | `str` | `"maximize"` | Sense string for `set_objective()` |

The `infinity()` instance method is a convenience accessor for the `INFINITY` class constant.

### Abstract Methods

| Method | Purpose | Key Notes |
|--------|---------|-----------|
| `add_var(*, vtype, lb, ub, name)` | Create a decision variable | `vtype` is a `SolverVarType`; `ub=None` means `INFINITY`; returns a `SolverVar` |
| `add_constr(expr, *, name)` | Add a constraint | `expr` is a backend constraint expression (e.g. a TempConstr from Gurobi) |
| `set_objective(expr, *, sense)` | Set the optimization objective | `sense` is `"minimize"` or `"maximize"`; raises `ValueError` for other values |
| `optimize(callback)` | Run the solver | `callback` is an optional backend-specific callable; non-Gurobi backends may ignore it |
| `set_param(param, value)` | Set a solver parameter | `param` is a `SolverParams` enum member; raises `NotImplementedError` if unsupported |
| `get_status()` | Return solve status as a string | Returns one of: `"OPTIMAL"`, `"INFEASIBLE"`, `"TIME_LIMIT"`, `"UNBOUNDED"`, or `"UNKNOWN(N)"` |
| `get_sol_count()` | Return the number of solutions found | Returns `0` if no solve has been run |
| `solve_stats()` | Return a `SolveStats` instance | Must be called after `optimize()`; unavailable fields are `None` |
| `compute_iis()` | Compute an Irreducible Infeasible Subsystem | Only meaningful on infeasible models; behavior differs by backend |
| `write(path)` | Write the model to a file | Supports `.lp`, `.mps`, `.ilp` — behavior differs by backend |
| `quicksum(iterable)` | Efficient sum over an iterable of backend expressions | Delegates to the backend's native sum implementation |
| `lin_expr(constant)` | Create a zero or constant-valued linear expression | Usable as a `defaultdict(model.lin_expr)` value factory |

### Non-Abstract Methods

| Method/Property | Purpose | Default Behavior |
|-----------------|---------|-----------------|
| `supports_nonlinear` | Whether the backend supports general non-linear constraints | Returns `False`; `GurobiBackend` overrides to `True` |
| `add_genconstr_nl(resvar, expr, *, name)` | Add a general non-linear constraint `resvar = expr` | Raises `NotImplementedError`; only `GurobiBackend` overrides this |
| `infinity()` | Convenience accessor for `INFINITY` | Returns `self.INFINITY` |

---

## SolverVar and LinExpr

### SolverVar

`SolverVar` is an abstract wrapper around a backend decision variable. It exposes two key properties:

- `.X` — the solution value after `optimize()` has been called. Raises `ValueError` if accessed before a feasible solution exists.
- `._raw` — the underlying backend variable object (e.g. `gp.Var` for Gurobi, `mathopt.Variable` for OR-Tools). Use `._raw` when building backend-specific constraint expressions.

`SolverVar` also delegates all arithmetic operators (`+`, `-`, `*`, `/`, `<=`, `>=`, `==`, unary `-`) to its underlying backend variable. This means `SolverVar` objects can be used directly in expressions without unwrapping, while still producing valid backend constraint expressions.

### LinExpr

`LinExpr` is an abstract wrapper for linear expressions. It is designed for accumulation patterns: you can use `+=` to add terms incrementally, and it supports the `defaultdict(model.lin_expr)` pattern for building per-key expressions in loops.

`LinExpr` exposes a `._raw` property for the underlying backend expression, and delegates arithmetic operators (`+`, `-`, `*`, `<=`, `>=`, `==`) to the backend. The `model.lin_expr()` factory creates a zero-valued expression ready for accumulation.

Both `SolverVar` and `LinExpr` have private backend implementations (`_GurobiVar`, `_GurobiLinExpr`, `_ORToolsVar`, `_ORToolsLinExpr`) that should never be constructed directly — always go through `SolverModel.add_var()` and `SolverModel.lin_expr()`.

---

## Backend Comparison

| Capability | GurobiBackend | ORToolsBackend |
|------------|---------------|----------------|
| Nonlinear constraints | Supported via `addGenConstrNL`; `supports_nonlinear` returns `True` | Not supported; `supports_nonlinear` returns `False`; callers must linearize (e.g. piecewise linearization for division) |
| Infeasibility handling | Computes IIS via `computeIIS()`; writes `.ilp` files natively | Logs a WARNING and suggests using `write()` for MPS export; `compute_iis()` is a no-op |
| MPS/model export | `write()` supports `.lp`, `.mps`, `.ilp` natively | `write()` always produces MPS; `.ilp` requests are silently remapped to `.mps` |
| Solver parameters | Maps all 5 `SolverParams` to native Gurobi parameter names | Maps all 5 `SolverParams`; `TIME_LIMIT` is converted from seconds to `timedelta` |
| License | Requires a gurobipy license (commercial; academic free) | GSCIP and HiGHS are open-source and bundled; ORTOOLS_GUROBI requires a Gurobi license |
| Name uniqueness | Allows duplicate variable and constraint names silently | Enforces globally unique names within a model; appends a counter suffix on collision (`name_1`, `name_2`, ...) |
| `quicksum` implementation | Delegates to `gurobipy.quicksum` for efficiency | Delegates to `mathopt.fast_sum` |
| Callback support | Passes callback to `model.optimize(callback)` | Ignores callback; OR-Tools MathOpt does not support user callbacks |
| `mip_gap`, `node_count`, `iteration_count` in `SolveStats` | Populated after solve (MIPGap, NodeCount, IterCount from Gurobi model) | Always `None`; MathOpt does not expose these statistics |

---

## Factory Pattern

`create_solver(backend: SolverBackend, name: str = "") -> SolverModel` is the single entry point for backend instantiation. It maps each `SolverBackend` member to exactly one backend configuration:

- `GUROBI` maps to `GurobiBackend(name)`.
- `ORTOOLS_GSCIP`, `ORTOOLS_HIGHS`, and `ORTOOLS_GUROBI` all map to `ORToolsBackend(name, solver_type)`, where `solver_type` is the corresponding `mathopt.SolverType` (GSCIP, HIGHS, or GUROBI respectively).
- Any unrecognized `SolverBackend` value raises `ValueError`.

The factory hides all construction details. Callers only need to choose a `SolverBackend` enum value and then interact with the returned `SolverModel`.

---

## SolveStats

`SolveStats` is a frozen dataclass returned by `solve_stats()` after `optimize()` has been called. It provides a backend-agnostic view of solve outcomes.

| Field | Type | Description |
|-------|------|-------------|
| `backend` | `str` | Backend name, e.g. `"GUROBI"` or `"ORTOOLS_GSCIP"` |
| `solver` | `str` | Underlying solver name, e.g. `"gurobi"`, `"gscip"`, `"highs"` |
| `status` | `str` | Solve status, e.g. `"OPTIMAL"`, `"INFEASIBLE"`, `"TIME_LIMIT"` |
| `objective` | `float \| None` | Objective value of the best solution found, or `None` if no solution exists |
| `solve_time_s` | `float` | Wall-clock solve time in seconds |
| `mip_gap` | `float \| None` | Relative MIP gap; `None` for OR-Tools backends |
| `node_count` | `int \| None` | Branch-and-bound nodes explored; `None` for OR-Tools backends |
| `iteration_count` | `int \| None` | Simplex iterations; `None` for OR-Tools backends |

---

## SolverParams

`SolverParams` is an enum used with `set_param(param, value)` to configure solver behavior before calling `optimize()`.

| Member | Purpose | Gurobi Parameter | OR-Tools Mapping |
|--------|---------|-----------------|-----------------|
| `VERBOSITY` | Output verbosity level | `OutputFlag` | `enable_output` (bool) |
| `TIME_LIMIT` | Maximum solve time in seconds | `TimeLimit` | `time_limit` (converted to `timedelta`) |
| `THREADS` | Number of parallel solver threads | `Threads` | `threads` (int) |
| `POOL_GAP` | Solution pool MIP gap tolerance | `PoolGap` | `relative_gap_tolerance` (float) |
| `LOG_TO_CONSOLE` | Enable/disable console logging | `LogToConsole` | `enable_output` (bool) |

Both backends support all five `SolverParams`. The key difference is that `TIME_LIMIT` is passed as a plain number (seconds) by the caller but OR-Tools converts it internally to a `datetime.timedelta` before applying it to `SolveParameters`.

---

## SolverVarType

`SolverVarType` is an enum passed to `add_var()` to declare the variable domain.

| Member | Domain | Typical Use |
|--------|--------|-------------|
| `BINARY` | `{0, 1}` | Allocation decisions, routing choices |
| `INTEGER` | Integer range `[lb, ub]` | Counts, multipliers |
| `CONTINUOUS` | Real range `[lb, ub]` | Costs, objective terms, relaxed variables |

---

## When to Use Each Backend

The right backend depends on problem structure and license availability:

- **Default to `ORTOOLS_GSCIP`** for license-free operation. GSCIP is a full MILP solver bundled with the `ortools` pip package and requires no external installation or license key.

- **Use `GUROBI`** when: (a) the MILP contains non-linear constraints (division encoding via `addGenConstrNL`), (b) you need IIS computation to debug infeasibility, or (c) maximum solver performance is required. The TETRA division encoding in the MILP formulation uses `add_genconstr_nl`, which requires a Gurobi backend.

- **Use `ORTOOLS_HIGHS`** as an alternative open-source LP/MIP solver. HiGHS is strong on LP relaxations and may be faster than GSCIP for some problem structures.

- **Use `ORTOOLS_GUROBI`** to run Gurobi through MathOpt's unified OR-Tools API. This gives you Gurobi performance with the OR-Tools interface, but still requires a Gurobi license and does not expose `supports_nonlinear`.

---

See also: `.claude/skills/optimization/constraint-selection.md` for the constraint toggling system, `.claude/skills/constraints/` for the MILP formulation and TransferAndTensorAllocator details.
