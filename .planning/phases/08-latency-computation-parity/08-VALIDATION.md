---
phase: 8
slug: latency-computation-parity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-09
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (installed in .venv) |
| **Config file** | pyproject.toml or implicit |
| **Quick run command** | `.venv/bin/pytest tests/unit/test_co_tile_variables.py tests/unit/test_tile_aware_latency.py -x -q` |
| **Full suite command** | `.venv/bin/pytest tests/unit/ -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/pytest tests/unit/test_co_tile_variables.py tests/unit/test_tile_aware_latency.py -x -q`
- **After every plan wave:** Run `.venv/bin/pytest tests/unit/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green + regression test passes
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | LAT-01 | unit | `.venv/bin/pytest tests/unit/test_tile_aware_latency.py -x` | Yes | ⬜ pending |
| 08-01-02 | 01 | 1 | LAT-02 | unit | `.venv/bin/pytest tests/unit/test_co_tile_variables.py::test_slot_latency_variable_mode -x` | Yes (FAILING) | ⬜ pending |
| 08-01-03 | 01 | 1 | LAT-03 | regression | `.venv/bin/pytest tests/regression/test_baseline.py -m slow -x` | Yes | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No new test files needed — the fix must make the existing failing test `test_slot_latency_variable_mode` pass.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
