---
phase: 1
slug: baseline-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | none — Wave 0 installs conftest.py and pytest config |
| **Quick run command** | `python -m pytest tests/regression/test_baseline.py -x` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~120 seconds (Gurobi CO solve dominates) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/regression/test_baseline.py -x`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | BASE-01 | integration | `python -m pytest tests/regression/test_baseline.py -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | BASE-02 | regression | `python -m pytest tests/regression/test_baseline.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/regression/test_baseline.py` — stubs for BASE-01, BASE-02
- [ ] `tests/conftest.py` — shared fixtures (baseline JSON path, tolerance constants)
- [ ] pytest config in `pyproject.toml` — testpaths, markers (slow)

*Existing pytest is installed in .venv but no test config or directories exist.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CO solver completes on BIG BOY config | BASE-01 | First run generates fixture; no prior baseline to compare | Run `python main_swiglu_v2.py` with defaults, verify non-zero objective printed |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
