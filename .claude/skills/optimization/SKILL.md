---
name: stream-aie-optimization
description: Use when working on solver backends, adding a new solver, configuring ConstraintSelection, or debugging backend-specific behavior in stream_aie.
---

# Optimization -- Solver & Constraint Selection

## When to Load This Skill

Use when:
- Working on solver backends (GurobiBackend, ORToolsBackend)
- Adding a new solver to the SolverModel ABC
- Configuring or debugging ConstraintSelection toggles
- Understanding the factory pattern for backend dispatch
- Debugging backend-specific infeasibility handling or MPS export

## Contents

| File | Description |
|------|-------------|
| `solver-backends.md` | SolverModel ABC, GurobiBackend, ORToolsBackend, factory pattern, when to use each |
| `constraint-selection.md` | ConstraintSelection dataclass, its relationship to NamespaceConstraints, which constraints apply to which hardware |

See also: `.claude/skills/constraints/` for MILP formulation details and namespace constraint dispatch.
