---
phase: 02
slug: tilesizelut-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already installed, in `dev` extras) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]`, `testpaths = ["tests"]` |
| **Quick run command** | `.venv/bin/pytest tests/unit/test_tile_size_utils.py -x` |
| **Full suite command** | `.venv/bin/pytest tests/ -m "not slow" -x` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/pytest tests/unit/test_tile_size_utils.py -x`
- **After every plan wave:** Run `.venv/bin/pytest tests/ -m "not slow" -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | TILE-01 | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_search_space_accepts_candidate_list -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | TILE-02 | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_tensor_size_bits_matches_expected -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | TILE-04 | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_divisibility_filter -x` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | TILE-04 | unit | `.venv/bin/pytest tests/unit/test_tile_size_utils.py::test_memory_filter -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/__init__.py` — create unit test subdirectory
- [ ] `tests/unit/test_tile_size_utils.py` — covers TILE-01, TILE-02, TILE-04
- [ ] `tests/unit/conftest.py` — shared fixtures: minimal SwiGLU workload + mapping

*Wave 0 tests created as part of plan tasks.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
