---
phase: 5
slug: variable-transfer-latency
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-08
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/unit/test_co_tile_variables.py, tests/regression/test_baseline.py |
| **Quick run command** | `python -m pytest tests/unit/test_co_tile_variables.py -x -q` |
| **Full suite command** | `python -m pytest tests/unit/test_co_tile_variables.py tests/unit/test_ssis_tile_aware.py tests/regression/test_baseline.py -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/unit/test_co_tile_variables.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/unit/test_co_tile_variables.py tests/unit/test_ssis_tile_aware.py tests/regression/test_baseline.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | CO-03 | unit | `python -m pytest tests/unit/test_co_tile_variables.py -k "latency" -v` | ❌ W0 | pending |
| 05-01-02 | 01 | 1 | CO-03 | unit | `python -m pytest tests/unit/test_co_tile_variables.py -k "no_genconstr" -v` | ❌ W0 | pending |
| 05-01-03 | 01 | 1 | CO-03 | regression | `python -m pytest tests/regression/test_baseline.py -v` | yes | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_co_tile_variables.py` — add latency-specific tests for variable tile mode
- [ ] `tests/unit/test_co_tile_variables.py` — add test asserting no addGenConstrNL in model

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
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
