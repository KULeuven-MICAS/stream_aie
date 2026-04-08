---
phase: 05-variable-transfer-latency
verified: 2026-04-08T12:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 05: Variable Transfer Latency Verification Report

**Phase Goal:** Transfer latency in the CO model is a linear expression over tile selection variables; no tile-dependent quantity remains as a fixed scalar in the MILP
**Verified:** 2026-04-08T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `_active_transfer_latency` returns a `gp.Var` computed from pre-computed scalar coefficients times binary products — no `addGenConstrNL` in the model | VERIFIED | `grep -c "addGenConstrNL" transfer_and_tensor_allocation.py` returns 0; entire CO module has zero NL constraints |
| 2 | Degenerate single-candidate latency equals `ceil(tensor_size_bits / min_bw) / sf_coeff` (same as old scalar path) | VERIFIED | `test_active_transfer_latency_degenerate` passes: 1 candidate (512 bits, bw=128, sf=1) produces latency 4.0 exactly |
| 3 | Multi-candidate model selects only the active candidate's amortized latency contribution | VERIFIED | `test_active_transfer_latency_variable_mode` passes: 2 candidates, jw0 forced=1, latency=4.0 (amortized for candidate 0) |
| 4 | The model contains zero GenConstrNL constraints after full build | VERIFIED | `test_no_genconstr_nl_in_model` passes: `model.NumGenConstrs == 0`; `test_active_transfer_latency_scalar_fallback` also asserts `model.NumGenConstrs == 0` |
| 5 | Scalar fallback (no tiled dims) still works without `addGenConstrNL` | VERIFIED | `test_active_transfer_latency_scalar_fallback` passes: `_joint_candidates_for_tensor` returns `[]`, plain `(fires, sf)` tuple in `reuse_levels`, latency=4.0, `NumGenConstrs == 0` |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` | Refactored `_active_transfer_latency` with pure MILP enumeration; `lat_lc_` pattern present | VERIFIED | `lat_lc_` appears 7 times, `lat_sum_` appears 6 times; method at line 1842 uses Big-M linearization over `(k, s)` pairs |
| `tests/unit/test_co_tile_variables.py` | Unit tests for variable-mode latency containing `test_active_transfer_latency` | VERIFIED | 4 new test functions at lines 1541, 1616, 1677, 1743 |

#### Level 2 (Substantive)

- `transfer_and_tensor_allocation.py`: `_active_transfer_latency` (line 1842) is fully implemented — 145-line body using `quicksum`, `ceil(sb / min_bw)`, `isinstance(rl_check, list)` detection, Big-M constraints with `lat_lc_ub_expr_`, `lat_lc_ub_m_`, `lat_lc_lb_` names, `lat_sum_` aggregation variable, and final gating via `_add_binary_scaled_continuous`. Dead methods `_add_const_over_linexpr` and `_add_binary_times_const_over_linexpr` are absent (grep returns 0).
- `test_co_tile_variables.py`: All 4 test functions are substantive — they build Gurobi models, constrain variables, call `_active_transfer_latency`, optimize, and assert specific numeric values and `NumGenConstrs == 0`.

#### Level 3 (Wired)

- `_active_transfer_latency` is called at line 1256 inside `_slot_scheduling_constraints`; its return value is immediately constrained at line 1257: `self.model.addConstr(self.slot_latency[s] >= active_latency, name=f"tr_lat_{tr.name}_{hash(choice)}")`.
- The `lat_lc_` and `lat_sum_` auxiliary variables are created inside `_active_transfer_latency` and constrained within the same method — they feed through to the returned `active_latency` variable via `_add_binary_scaled_continuous`.

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_active_transfer_latency` | `_joint_candidates_for_tensor`, `reuse_levels`, `_add_binary_product`, `_add_binary_scaled_continuous` | enumeration of `(k, s)` pairs with pre-computed `amortized_latency` coefficients; `lat_lc_` pattern | WIRED | `lat_lc_` pattern confirmed at lines 1905, 1962; `isinstance(rl_check, list)` at line 1858; `_add_binary_scaled_continuous` called at lines 1924 and 1979 |
| `_active_transfer_latency` result (`gp.Var`) | `_slot_scheduling_constraints` (lines 1254-1257) | `self.slot_latency[s] >= active_latency` constraint | WIRED | Line 1257: `self.model.addConstr(self.slot_latency[s] >= active_latency, name=f"tr_lat_{tr.name}_{hash(choice)}")` confirmed |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `_active_transfer_latency` | `amortized_latency[k,s]` coefficients | `ceil(sb / min_bw) / sf_c` where `sb` comes from `_joint_candidates_for_tensor` (tensor size per tile candidate) and `sf_c` from `reuse_levels[(t, s)]` | Yes — scalar computed from model data structures, no hardcoded empty values | FLOWING |
| `lat_sum_` variable | `lc_parts` (list of `lat_lc_` vars) | Big-M linearized `z_stop * quicksum(amort * jw)` constraints | Yes — constrained by `lat_sum_def_` equality constraint | FLOWING |
| Final `active_latency` | `lat_sum` gated by path choice `y` | `_add_binary_scaled_continuous(binary_var=y, continuous_var=lat_sum, ...)` | Yes — established Phase 4 linearization pattern | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 4 new latency tests pass | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -k "test_active_transfer_latency or test_no_genconstr_nl" -x -q` | `4 passed, 41 deselected in 0.67s` | PASS |
| Full test file (45 tests, no regressions) | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -x -q` | `45 passed in 0.70s` | PASS |
| Zero `addGenConstrNL` in CO module | `grep -c "addGenConstrNL" stream/opt/allocation/constraint_optimization/*.py` | All files return 0 | PASS |
| Dead NL helpers removed | `grep -c "def _add_const_over_linexpr\|def _add_binary_times_const_over_linexpr" transfer_and_tensor_allocation.py` | 0 | PASS |
| New `lat_lc_` auxiliaries exist | `grep -c "lat_lc_" transfer_and_tensor_allocation.py` | 7 | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CO-03 | 05-01-PLAN.md | Transfer sizes and latencies become linear expressions over tile selection variables | SATISFIED | `_active_transfer_latency` computes `amortized_latency[k,s] = ceil(size_bits[k]/min_bw) / sf_coeff[k,s]` as scalar coefficients; binary products `joint_w[k] * z_stop[s]` select the active pair — the entire latency computation is a linear expression; `addGenConstrNL` count = 0 across full CO module |

No orphaned requirements: REQUIREMENTS.md maps CO-03 exclusively to Phase 5 (row present in traceability table, marked Complete).

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | — |

Scanned `transfer_and_tensor_allocation.py` (latency region lines 1842-1987) and `test_co_tile_variables.py` (lines 1541-1792): no TODO/FIXME comments, no empty return stubs, no hardcoded `return []` or `return {}`, no `console.log`-only handlers.

---

### Human Verification Required

None. All truths are verifiable programmatically:

- NL constraint elimination: confirmed by grep (0 occurrences) and `model.NumGenConstrs == 0` assertion in tests
- Degenerate equality: confirmed by numeric assertion in unit test
- Scalar fallback correctness: confirmed by unit test
- Regression safety: confirmed by full 45-test suite passing

---

### Gaps Summary

No gaps. All 5 must-have truths are verified. Both artifacts are substantive and wired. Both key links are confirmed at the source level. All behavioral spot-checks pass. CO-03 is satisfied.

---

_Verified: 2026-04-08T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
