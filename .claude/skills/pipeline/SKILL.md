---
name: stream-aie-pipeline
description: Use when working on pipeline stages, debugging stage execution order, adding a new stage, or tracing data flow through StageContext in stream_aie.
---

# Pipeline -- Stages & Execution Model

## When to Load This Skill

Use when:
- Working on any pipeline stage (parsing, generation, estimation, allocation)
- Adding a new stage to the pipeline
- Debugging data flow through StageContext
- Understanding how MainStage and LeafStage compose
- Tracing how the scheduler invokes stages

## Contents

| File | Description |
|------|-------------|
| `pipeline-stages.md` | Each active stage with its responsibility, inputs/outputs, and position in the flow |
| `stage-execution.md` | StageContext, MainStage/LeafStage execution model, how stages compose and run |

*Content files will be added by Phase 12.*

See also: `.claude/skills/optimization/` for solver backend details used by the allocation stage.
