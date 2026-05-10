# IR Models -- Typed Intermediate Representations

## What the IR Package Is

`stream/ir/` is a Pydantic layer over the internal `get_ir()` dict methods on workload, accelerator, and scheduler objects. It adds a schema contract, construction-time validation, and JSON Schema generation (`model_json_schema()`) on top of data that already exists inside stream_aie's internal pipeline.

The IR package is independent of MCP — any serialization path (REST, file, MCP tool response) can consume it. It does not add new data; it wraps and structures what `get_ir()` already returns.

## The Three IR Classes

### WorkloadIR

Wraps `Workload.get_ir()`. Describes the computation graph: nodes (ComputationNode, TransferNode), DAG edges, tensor operand shapes, loop dimension inventory, and timeslot (generation) assignments. Suitable as an input description before scheduling.

### AcceleratorIR

Wraps `Accelerator.get_ir()`. Describes the hardware topology: cores (aie2, zigzag, offchip), their positions in the 2-D grid, memory and FIFO resources, and bus/link connectivity. Suitable as a hardware description before or after scheduling.

### AllocationIR

Wraps `SteadyStateScheduler.get_ir()`, which embeds `Mapping.get_ir()`. Describes a completed schedule: latency totals, solver backend, constraint configuration, per-node resource and memory allocation, inter-core tiling, fused layer groups, and code-generation runtime args. Only valid for a post-solve scheduler.

## Construction Pattern

Always use `from_internal(obj)`. Never construct IR classes directly from external code.

- `WorkloadIR.from_internal(workload)` — call on a `Workload` object
- `AcceleratorIR.from_internal(accelerator)` — call on an `Accelerator` object
- `AllocationIR.from_internal(scheduler)` — call on a **post-solve** `SteadyStateScheduler`

`AllocationIR.from_internal()` raises `ValueError` if the scheduler has not been solved (latency_total sentinel == -1). Always check that `scheduler.run()` completed before calling it.

## Schema Versioning

Every IR model carries `schema_version: Literal['1.0']` which generates `"const": "1.0"` in the JSON Schema output. Rules:

- Minor bump (1.1): additive fields only — old consumers continue to work
- Major bump (2.0): removed or renamed fields — consumers must update

No migration infrastructure at this stage. Schema version is tracked per IR model independently.

## Per-Persona Views

Every IR model exposes view methods that return lightweight Pydantic projection models. Views do not call `get_ir()` again — they project from the already-validated IR fields. Every view also carries `schema_version`.

### Algorithmic Engineer

Concerned with workload shape, schedule quality, and solver configuration.

- `WorkloadIR.algorithmic_view()` — node and edge counts, unique dimension count, dimension expressions. Use to understand workload size and structure.
- `AllocationIR.algorithmic_view()` — full latency metrics (total, per-iteration, overlap), solver backend, constraint selection flags, fusion splits. Use to evaluate schedule quality and solver behaviour.

AcceleratorIR has no algorithmic view; hardware topology is hardware-engineer territory.

### Hardware Engineer

Concerned with physical resource usage, memory capacity, and connectivity.

- `AcceleratorIR.hardware_view()` — all cores with their type-specific resource data (memory capacity, FIFO depth for aie2; operational array for zigzag), utilization, and full bus/link connectivity. Use to audit resource budgets.
- `AllocationIR.hardware_view()` — per-node resource allocation (which cores or paths are used per slot) and memory allocation (which core IDs hold each tensor). Use to understand physical resource usage after scheduling.

WorkloadIR has no hardware view; the workload graph is algorithm-level.

### Compiler Engineer

Concerned with code generation: node-to-core mapping, tiling, transfer routing, fused groups.

- `WorkloadIR.compiler_view()` — full node list (types, dimensions, tensor operands), DAG edges, tensor metadata, and generation (timeslot) assignments. Use for placement and routing decisions.
- `AcceleratorIR.compiler_view()` — core topology (id, type, row/col position) and connectivity. Use for placement decisions without type-specific resource noise.
- `AllocationIR.compiler_view()` — per-node inter-core tiling and resource mapping, fused groups with intra-core tiling, runtime args for code generation. Use to drive MLIR code generation.

## Anti-Patterns

Do not bring `stream.ir` into `stream/workload/`, `stream/mapping/`, or `stream/cost_model/`. The IR package depends on internal classes (guarded by TYPE_CHECKING) — not the reverse. A reverse dependency creates circular imports.

Do not call `get_ir()` directly in view methods or construct IR instances without `from_internal()`. The `from_internal()` classmethod is the single validated entry point.

Do not call `AllocationIR.from_internal()` on a pre-solve scheduler. The scheduler uses -1 as a sentinel for unsolved latency fields; the IR layer enforces this with a ValueError.
