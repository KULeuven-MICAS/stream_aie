---
phase: 6
slug: variable-compute-latency
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-08
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/unit/test_co_tile_variables.py, tests/regression/test_baseline.py |
| **Quick run command** | `python -m pytest tests/unit/test_co_tile_variables.py -x -q` |
| **Full suite command** | `python -m pytest tests/unit/test_co_tile_variables.py tests/unit/test_ssis_tile_aware.py tests/regression/test_baseline.py -v` |
| **Estimated runtime** | ~35 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/test_co_tile_variables.py -x -q`
- **After every plan wave:** Run full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 35 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | CO-06 | unit | `python -m pytest tests/unit/test_co_tile_variables.py -k "compute_latency" -v` | W0 | pending |
| 06-01-02 | 01 | 1 | CO-06 | regression | `python -m pytest tests/regression/test_baseline.py -v` | yes | pending |
| 06-02-01 | 02 | 2 | CO-06 | unit | `python -m pytest tests/ -k "cost_lut" -v` | W0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_co_tile_variables.py` — add compute latency unit tests
- [ ] Verify no import errors after cost_lut removal across codebase

*Existing infrastructure covers remaining phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 35s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
