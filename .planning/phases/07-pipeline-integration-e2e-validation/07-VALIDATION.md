---
phase: 7
slug: pipeline-integration-e2e-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-08
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/unit/test_co_tile_variables.py, tests/regression/test_baseline.py |
| **Quick run command** | `python -m pytest tests/unit/test_co_tile_variables.py -x -q` |
| **Full suite command** | `python -m pytest tests/unit/ tests/regression/test_baseline.py -v` |
| **Estimated runtime** | ~40 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 40 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | PIPE-02 | unit | `python -m pytest tests/unit/test_co_tile_variables.py -k "selected_tiles" -v` | W0 | pending |
| 07-01-02 | 01 | 1 | PIPE-02 | regression | `python -m pytest tests/regression/test_baseline.py -v` | yes | pending |
| 07-02-01 | 02 | 2 | PIPE-01 | e2e | `python -m pytest tests/regression/ -v` | W0 | pending |

---

## Wave 0 Requirements

- [ ] `tests/unit/test_co_tile_variables.py` — add get_selected_tiles unit test
- [ ] `tests/regression/` — add multi-candidate E2E test

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 40s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
