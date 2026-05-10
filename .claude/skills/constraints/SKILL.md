---
name: stream-aie-constraints
description: Use when working on MILP constraint formulation, modifying TransferAndTensorAllocator, debugging constraint groups, or understanding NamespaceConstraints dispatch in stream_aie.
---

# Constraints -- MILP Formulation & Namespace Dispatch

## When to Load This Skill

Use when:
- Modifying or adding MILP constraints in TransferAndTensorAllocator
- Debugging constraint group behavior (memory, FIFO, buffer descriptors, DMA)
- Understanding how ConstraintSelection guards interact with constraint dispatch
- Working on NamespaceConstraints or AIE2Constraints
- Adding hardware-specific constraint profiles

## Contents

| File | Description |
|------|-------------|
| `milp-formulation.md` | TransferAndTensorAllocator model structure: decision variables, constraint groups, objective function, ConstraintSelection guards |
| `namespace-constraints.md` | NamespaceConstraints base class, AIE2Constraints, hardware-specific constraint dispatch |

See also: `.claude/skills/optimization/` for solver backend abstraction and ConstraintSelection dataclass.
