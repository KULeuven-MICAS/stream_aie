---
phase: 03
slug: tile-selection-variables-memory-constraints
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 03 — Validation Strategy

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
| 03-01-01 | 01 | 1 | TILE-03 | unit | `.venv/bin/pytest tests/unit/test_tile_selection_vars.py -v` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | TILE-05 | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py -v` | ✅ | ⬜ pending |
| 03-02-01 | 02 | 2 | CO-01 | unit+regression | `.venv/bin/pytest tests/unit/ tests/regression/ -x -v` | ✅ | ⬜ pending |
| 03-02-02 | 02 | 2 | CO-05 | unit | `.venv/bin/pytest tests/unit/test_big_m_bounds.py -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_tile_selection_vars.py` — stubs for TILE-03 (w[dim,k] variable creation, one-hot constraints)
- [ ] `tests/unit/test_big_m_bounds.py` — stubs for CO-05 (tight per-constraint big-M)

*Existing infrastructure covers pytest framework and regression tests.*

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
