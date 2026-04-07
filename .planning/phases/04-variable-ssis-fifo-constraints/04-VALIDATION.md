---
phase: 04
slug: variable-ssis-fifo-constraints
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 04 — Validation Strategy

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
| 04-01-01 | 01 | 1 | CO-02 | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py -v` | ✅ | ⬜ pending |
| 04-01-02 | 01 | 1 | CO-02 | unit | `.venv/bin/pytest tests/unit/test_ssis_tile_aware.py -v` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 2 | CO-02 | unit+regression | `.venv/bin/pytest tests/unit/test_co_tile_variables.py tests/regression/ -x -v` | ✅ | ⬜ pending |
| 04-02-02 | 02 | 2 | CO-04 | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py -v` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_ssis_tile_aware.py` — stubs for tile-aware SSIS class (K/T loop decomposition per candidate)

*Existing infrastructure covers pytest framework, regression tests, and CO tile variable tests.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 55s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
