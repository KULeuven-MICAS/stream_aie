---
name: stream-aie-api-testing
description: Use when using stream_aie programmatically, running CLI scripts, writing tests, or setting up backend patching for test isolation in stream_aie.
---

# API & Testing -- Public Interface & Test Patterns

## When to Load This Skill

Use when:
- Calling optimize_allocation_co() or optimize_mapping() programmatically
- Running CLI scripts (scripts/main_gemm.py, scripts/main_swiglu.py, etc.)
- Writing or debugging tests for constraint groups or backends
- Setting up backend patching for test isolation
- Understanding the study scripts (constraint_toggle_study.py)

## Contents

| File | Description |
|------|-------------|
| `api-reference.md` | optimize_allocation_co() and optimize_mapping() signatures, CLI flags, common usage patterns, return types |
| `testing-patterns.md` | Test organization, backend patching patterns, study scripts, how to add new tests |

See also: `.claude/skills/pipeline/` for how API entry points trigger the stage pipeline.
