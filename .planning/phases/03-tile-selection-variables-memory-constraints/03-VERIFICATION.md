---
phase: 03-tile-selection-variables-memory-constraints
verified: 2026-04-07T00:00:00Z
status: passed
score: 4/4 must-haves verified
gaps: []
human_verification: []
---

# Phase 3: Tile Selection Variables + Memory Constraints Verification Report

**Phase Goal:** The CO model contains binary w[dim,k] tile selection variables with one-hot constraints, and memory capacity constraints use LUT-derived linear expressions with tight per-constraint big-M bounds
**Verified:** 2026-04-07
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | For each unique workload dimension, exactly one w[dim,k] binary variable is selected (one-hot constraint enforced by solver) | VERIFIED | `__create_tile_selection_vars` at line 567 creates `w[(dim,k)]` BINARY vars and adds `w_one_hot_{dim}` constraint `sum_k w[dim,k] == 1` for each dim in SearchSpace. `test_one_hot_constraint_exists` and `test_single_candidate_regression_compat` verify solver enforces it (w[dim,0].X == 1.0). |
| 2 | Memory capacity constraints use sum_k(tensor_size_lut[t,k] * w[dim,k]) in place of a scalar tensor_size constant | VERIFIED | `_memory_capacity_constraints` (line 758) calls `_joint_candidates_for_tensor(t, tr)` to get `(size_bits, joint_binary_var)` pairs and builds `tile_expr = quicksum(ceil(size_factor * sz) * jw for sz, jw in joint_candidates)`. Continuous auxiliary `lc` with big-M activation enforces `lc = tile_expr` when `uz=1`. Scalar fallback preserved when `joint_candidates` is empty. |
| 3 | Per-constraint big-M bounds are derived from max(tensor_size_lut[t,k]) over k, not the legacy scalar heuristic | VERIFIED | Line 803-804: `tensor_max = self._tensor_max_size[t]; M = ceil(size_factor * tensor_max)`. `_tensor_max_size[t]` is populated by `_joint_candidates_for_tensor` as `max(size over all joint combinations)`. `test_tight_bigm_not_legacy` verifies `lc.UB == ceil(size_factor * max_size)` not `len(nodes)+5`. |
| 4 | The regression test passes with a single-candidate degenerate input (variable tile mode recovers the fixed-tile baseline result) | VERIFIED | `test_single_candidate_regression_compat` (line 600): builds 1-candidate SearchSpace, runs `model.optimize()`, asserts `w[dim,0].X == 1.0` and `tile_var[dim].X == 16.0`. All 41 unit tests pass. Full non-slow test suite passes (78 passed, 4 deselected). |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `stream/opt/tile_size_utils.py` | `tensor_size_bits_for_candidate` and `max_tensor_size_bits` functions | VERIFIED | Both defined at lines 47 and 76. Fully implemented with itertools.product enumeration. 5 unit tests pass. |
| `stream/stages/allocation/constraint_optimization_allocation.py` | search_space read from ctx and passed to SteadyStateScheduler | VERIFIED | Line 57: `self.search_space = self.ctx.get("search_space")`. Line 81: `search_space=self.search_space` in SteadyStateScheduler call. |
| `stream/cost_model/steady_state_scheduler.py` | search_space optional kwarg threaded to TransferAndTensorAllocator | VERIFIED | Line 59: `search_space: SearchSpace | None = None` in `__init__`. Line 82: `self.search_space = search_space`. Line 125: `search_space=self.search_space` in TransferAndTensorAllocator call. |
| `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` | self.search_space stored; w/tile_var/tensor_max_size dicts; __create_tile_selection_vars; _joint_candidates_for_tensor; _joint_binary_for_combo; _tiled_dims_for_tensor; rewritten _memory_capacity_constraints | VERIFIED | All methods present (lines 567, 1484, 1502, 1578). Instance vars declared at lines 96-99. _memory_capacity_constraints at line 758 fully rewritten with load_contrib auxiliaries and tight big-M. |
| `tests/unit/test_tile_size_utils.py` | Tests for tensor_size_bits_for_candidate and max_tensor_size_bits | VERIFIED | 5 test functions at lines 183, 210, 235, 270, 296. All pass. |
| `tests/unit/test_co_tile_variables.py` | Tests for w vars, one-hot, tile_var, joint candidates, memory constraints, regression | VERIFIED | 18 test functions covering all behaviors (lines 114-618). All 41 unit tests in the file pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `constraint_optimization_allocation.py` | `steady_state_scheduler.py` | `search_space` kwarg in SteadyStateScheduler constructor | WIRED | Line 73-82: `SteadyStateScheduler(..., search_space=self.search_space)` confirmed at line 81. |
| `steady_state_scheduler.py` | `transfer_and_tensor_allocation.py` | `search_space` kwarg in TransferAndTensorAllocator constructor | WIRED | Lines 114-126: `TransferAndTensorAllocator(..., search_space=self.search_space)` confirmed at line 125. |
| `transfer_and_tensor_allocation.py` | `stream/opt/search_space.py` | `self.search_space.dims()` and `self.search_space.get(dim)` | WIRED | Lines 571-572 (`dims()`, `get()`), lines 1500, 1531. |
| `__create_tile_selection_vars` | `_create_vars` | called as last step of `_create_vars` | WIRED | Line 536: `self.__create_tile_selection_vars()` called after `__create_slot_latency_vars`. |
| `_memory_capacity_constraints` | `_joint_candidates_for_tensor` | `joint_candidates = self._joint_candidates_for_tensor(t, tr)` | WIRED | Line 779: called in variable tile path. |
| `_memory_capacity_constraints` | `_tensor_max_size` | `self._tensor_max_size[t]` for tight big-M | WIRED | Line 803: `tensor_max = self._tensor_max_size[t]` used to compute M. |
| `load_contrib auxiliary` | `self.core_load[c]` | `self.core_load[c] += lc` | WIRED | Line 832: confirmed. |

### Data-Flow Trace (Level 4)

The main artifact (`_memory_capacity_constraints`) operates on Gurobi model objects at optimization-build time (not at runtime data rendering). Data flow is through Gurobi constraint/variable creation, not through database or network fetches. The relevant data flows are:

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `_memory_capacity_constraints` | `joint_candidates` | `_joint_candidates_for_tensor(t, tr)` | Yes — returns `(size_bits, joint_binary_var)` pairs from workload dimension queries | FLOWING |
| `_memory_capacity_constraints` | `_tensor_max_size[t]` | populated by `_joint_candidates_for_tensor` as side-effect | Yes — max over all joint candidate sizes | FLOWING |
| `_memory_capacity_constraints` | `tile_expr` | `quicksum(ceil(size_factor * sz) * jw for sz, jw in joint_candidates)` | Yes — linear expression over real Gurobi binary vars | FLOWING |
| `__create_tile_selection_vars` | `w[(dim,k)]`, `tile_var[dim]` | `self.search_space.dims()`, `self.search_space.get(dim)` | Yes — populated from SearchSpace object passed via ctx | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Unit tests for utility functions pass | `.venv/bin/pytest tests/unit/test_tile_size_utils.py -q` | 36 passed in 0.95s | PASS |
| Unit tests for CO tile variables pass | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -q` | 41 passed in 0.95s | PASS |
| No regression in full non-slow suite | `.venv/bin/pytest tests/ -m "not slow" -x -q` | 78 passed, 4 deselected in 8.11s | PASS |
| All phase-03 commits exist in git log | `git log --oneline` grep for `03-01`, `03-02`, `03-03` | 9 commits found (feat/test/docs for all 3 plans) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TILE-05 | 03-01-PLAN.md | Utility functions for tensor sizes per candidate tile (incremental per phases 3-5) | SATISFIED | `tensor_size_bits_for_candidate` and `max_tensor_size_bits` defined and tested in `tile_size_utils.py`. REQUIREMENTS.md marks as Complete. |
| TILE-03 | 03-02-PLAN.md | One-hot binary selection variables w[dim,k] added to CO model with SOS1 constraints | SATISFIED | `__create_tile_selection_vars` creates BINARY `w[(dim,k)]` vars with `w_one_hot_{dim}` == 1 constraint. REQUIREMENTS.md marks as Complete. |
| CO-01 | 03-03-PLAN.md | Tensor sizes in memory capacity constraints become linear expressions over tile selection variables | SATISFIED | `_memory_capacity_constraints` uses `tile_expr = quicksum(ceil(size_factor * sz) * jw ...)` with continuous auxiliary `lc`. REQUIREMENTS.md marks as Complete. |
| CO-05 | 03-03-PLAN.md | Big-M bounds computed per-constraint using tight LUT-derived upper bounds | SATISFIED | `M = ceil(size_factor * self._tensor_max_size[t])` where `_tensor_max_size` is max over joint candidate sizes. REQUIREMENTS.md marks as Complete. |

**Orphaned requirements check:** REQUIREMENTS.md traceability table maps TILE-03, TILE-05, CO-01, CO-05 to Phase 3. All four are claimed by phase plans and verified above. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `transfer_and_tensor_allocation.py` | 845 | `# TODO: Confirm assumption that OF linking causes only single object fifo depth increase` | Info | Pre-existing comment in `_object_fifo_depth_constraints`, unrelated to phase 03 work. Not in any phase 03 modified code path. No impact on phase goal. |

No blocker or warning anti-patterns found in phase 03 code.

### Human Verification Required

None. All phase 03 success criteria are verifiable programmatically through unit tests and code inspection. The regression test (`test_single_candidate_regression_compat`) exercises the degenerate-case solver path within a unit test environment.

### Gaps Summary

No gaps found. All four ROADMAP success criteria are achieved:

1. One-hot constraints are implemented and solver-verified for the single-candidate degenerate case.
2. Memory capacity constraints use `tile_expr` linear expressions over joint binary variables via continuous `lc` auxiliaries.
3. Big-M bounds use `ceil(size_factor * _tensor_max_size[t])` — tight per-constraint LUT-derived values.
4. Single-candidate degenerate regression passes (`w[dim,0].X == 1.0`, `tile_var.X == single_tile_value`).

All 41 unit tests pass. Full non-slow test suite (78 tests) passes with no regressions.

---

_Verified: 2026-04-07_
_Verifier: Claude (gsd-verifier)_
