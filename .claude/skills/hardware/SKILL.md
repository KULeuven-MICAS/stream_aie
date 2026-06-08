---
name: stream-aie-hardware
description: Use when working on hardware core types, core namespaces, adding a new core architecture, or modifying per-core performance estimation in stream_aie.
---

# Hardware -- Core Model & Performance Estimation

## When to Load This Skill

Use when:
- Working on the hardware architecture model (`stream/hardware/`)
- Understanding core types, roles, or namespaces
- Adding a new core type (a new namespace or a new specialized compute core)
- Modifying or replacing per-core performance estimation
- Debugging why a core gets a particular estimator or LUT entry
- Understanding the `Core`, `Accelerator`, `CoreGraph` classes
- Reading or writing hardware YAML files

## Contents

| File | Description |
|------|-------------|
| `core-model.md` | Core roles and namespaces, the Core and Accelerator classes, example hardware YAMLs (AIE/TPU-like), how to add a new core type |
| `performance-estimation.md` | CoreCostEstimationStage dispatch, AIECostEstimator, ZigZagCostEstimator, CoreCostLUT caching, ZigZag fallback, how to add or swap an estimator |

See also: `.claude/skills/constraints/` for MILP constraint dispatch by namespace (NamespaceConstraints, AIE2Constraints). `.claude/skills/pipeline/` for where CoreCostEstimationStage sits in the pipeline.
