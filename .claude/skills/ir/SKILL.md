---
name: stream-aie-ir
description: Use when working with IR models (WorkloadIR, AllocationIR, AcceleratorIR), serializing workload or allocation data to JSON, choosing the right persona view for a consumer, or building MCP tool responses that expose schedule results.
---

# IR Models -- Typed Intermediate Representations

## When to Load This Skill

Use when:
- Constructing WorkloadIR, AllocationIR, or AcceleratorIR from internal stream_aie objects
- Choosing which persona view to return (algorithmic, hardware, or compiler)
- Serializing schedule results to JSON Schema-validated structures
- Building MCP tool responses that expose IR data to AI agents
- Understanding which fields are available per IR class and per view

## Contents

| File | Description |
|------|-------------|
| `ir-models.md` | Conceptual guide for all three IR classes, their construction pattern, persona views, and when to use each |

See also: `.claude/skills/pipeline/` for how the scheduler produces the data that AllocationIR wraps.
