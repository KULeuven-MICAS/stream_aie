---
phase: 04
slug: variable-ssis-fifo-constraints
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-07
---

# Phase 04 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `.venv/bin/pytest tests/unit/ -x -v` |
| **Full suite command** | `.venv/bin/pytest tests/ -x` |
| **Estimated runtime** | ~55 seconds (includes regression) |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/pytest tests/unit/ -x -v`
- **After every plan wave:** Run `.venv/bin/pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 55 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | CO-02 | unit | `.venv/bin/pytest tests/unit/test_ssis_tile_aware.py tests/unit/test_tile_size_utils.py -v` | Task creates test_ssis_tile_aware.py | pending |
| 04-01-02 | 01 | 1 | CO-02 | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -v` | yes | pending |
| 04-02-01 | 02 | 2 | CO-02 | unit+regression | `.venv/bin/pytest tests/unit/test_co_tile_variables.py tests/regression/ -x -v` | yes | pending |
| 04-02-02 | 02 | 2 | CO-04 | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -v` | yes | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

All Wave 0 gaps resolved:
- [x] `tests/unit/test_ssis_tile_aware.py` -- created by 04-01 Task 1 (TDD: test file written first as RED step)

*Existing infrastructure covers pytest framework, regression tests, and CO tile variable tests.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 55s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
