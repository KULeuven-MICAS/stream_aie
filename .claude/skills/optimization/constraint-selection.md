# Constraint Selection

## Overview

The constraint selection system controls which hardware-resource constraint groups are active in the `TransferAndTensorAllocator` MILP. It has two layers: `ConstraintSelection` (the user-facing coarse toggle, defined in `stream/opt/solver/solver.py`) and `NamespaceConstraints` (the hardware-specific dispatch, defined in `stream/opt/allocation/constraint_optimization/context.py`). Together they let operators relax hardware constraints for debugging or feasibility exploration without restructuring the solver model.

---

## ConstraintSelection Dataclass

`ConstraintSelection` is a frozen dataclass defined in `stream/opt/solver/solver.py`. It has four boolean fields, all defaulting to `True` (fully constrained). Setting a field to `False` skips that entire constraint group — no variables are created, no constraints are added, and for `dma_channels`, the associated objective terms are also omitted.

Because it is a frozen dataclass, instances are immutable once created. The default `ConstraintSelection()` constructor produces a fully-constrained configuration.

---

## Constraint-to-Hardware Mapping

| Field | Hardware Resource | Effect When Disabled | Nonsensical Warning |
|-------|------------------|----------------------|---------------------|
| `memory_capacity` | Per-core memory capacity | Skips memory capacity constraints; tensors may exceed physical on-chip memory | None |
| `object_fifo_depth` | Per-core object FIFO depth (AIE2) | Skips FIFO depth limits; may over-subscribe FIFO slots | WARNING if `memory_capacity=False` (see below) |
| `buffer_descriptors` | Per-core buffer descriptor count (AIE2) | Skips buffer descriptor limits; may exceed the hardware-imposed BD count | None |
| `dma_channels` | DMA channel count per tile type (AIE2) | Skips DMA channel constraints AND omits DMA usage terms from the objective | None |

---

## Nonsensical Combination Warning

`ConstraintSelection.__post_init__` checks for the combination `memory_capacity=False` with `object_fifo_depth=True` and emits a `WARNING` log message. This combination is nonsensical because object-FIFO depth constraints assume memory capacity is enforced — without memory capacity bounds, FIFO depth limits become meaningless as a resource guard. The solver continues regardless; it is a warning, not an error. The warning text explicitly states the reason: FIFO depth constraints assume memory capacity is enforced.

---

## Threading Through the Pipeline

`ConstraintSelection` is threaded from the public API all the way down to the allocator:

- In `stream/api.py`, the `constraint_selection` argument defaults to `None`.
- Inside `Stage`, a `None` value is replaced with `ConstraintSelection()` (all fields `True`, fully constrained). This means omitting the argument is identical to requesting all constraints.
- The `TransferAndTensorAllocator` receives the `ConstraintSelection` instance and checks each boolean field before adding the corresponding constraint group to the MILP model.

This design keeps the default behavior safe (all constraints active) while allowing targeted relaxation at the call site.

---

## NamespaceConstraints Pattern

`NamespaceConstraints` is the base class for hardware-specific MILP constraint dispatch, defined in `context.py`. Each subclass targets one hardware namespace:

- `NAMESPACE` — a class-level string attribute identifying the hardware namespace (e.g. `"aie2"`).
- `applies_to(core)` — returns `True` if `core.namespace == self.NAMESPACE`. Used to filter cores inside constraint methods.
- `add_object_fifo_constraints(model, object_fifo_depth)` — enforces FIFO depth limits; no-op in the base class.
- `add_buffer_descriptor_constraints(model, buffer_descriptor_depth)` — enforces buffer descriptor limits; no-op in the base class.
- `add_dma_usage_constraints(model, dma_usage_in, dma_usage_out)` — enforces DMA channel limits; returns an empty list in the base class.

Subclasses override only the methods relevant to their hardware. Methods left unoverridden remain no-ops, so only the constraints that apply to a given namespace are ever emitted. This is a strategy pattern: each hardware namespace provides its own constraint logic without modifying the allocator.

---

## AIE2Constraints

`AIE2Constraints` is the concrete `NamespaceConstraints` subclass for the AMD AIE2 tile array, with `NAMESPACE = "aie2"`. It enforces three resource categories:

**Object-FIFO depth:** For each AIE2 core, the accumulated FIFO depth expression is constrained to be at most `core.max_object_fifo_depth`.

**Buffer descriptors:** For each AIE2 core, the accumulated buffer descriptor expression is also constrained to be at most `core.max_object_fifo_depth` (the same hardware register governs both).

**DMA channels:** DMA channel limits depend on the tile type. Constraints apply to both S2MM (input) and MM2S (output) directions independently.

| Tile Type | Default Max DMA Channels |
|-----------|--------------------------|
| Compute | 8 |
| Memory | 6 |
| Shim | 2 |

These defaults are passed as constructor parameters to `AIE2Constraints` and can be overridden via `build_transfer_context()`. The shim tile limit applies to the off-chip core (identified by `offchip_core_id`).

---

## TransferAndTensorContext

`TransferAndTensorContext` is a frozen dataclass in `context.py` that serves as the shared context for the transfer and tensor allocation MILP. Its most relevant field is `namespace_constraints`, a tuple of `NamespaceConstraints` instances.

Three dispatch methods iterate over `namespace_constraints` and call the corresponding method on each strategy:

- `add_object_fifo_constraints(model, object_fifo_depth)` — calls each namespace's FIFO constraint method.
- `add_buffer_descriptor_constraints(model, buffer_descriptor_depth)` — calls each namespace's buffer descriptor constraint method.
- `add_dma_usage_constraints(model, dma_usage_in, dma_usage_out)` — calls each namespace's DMA constraint method.

The `build_transfer_context()` factory function constructs the context. It scans the accelerator's core list for known namespace strings and instantiates the appropriate `NamespaceConstraints` subclass. Currently only `"aie2"` is recognized — if the accelerator has any `aie2` cores, an `AIE2Constraints` instance is added. Future namespaces follow the same pattern: add a subclass, detect the namespace string in `build_transfer_context()`, and append the instance.

---

## How ConstraintSelection and NamespaceConstraints Interact

The two layers form a clean separation of concerns:

`ConstraintSelection` is the **coarse toggle**: it decides whether a constraint group runs at all. It is hardware-agnostic and lives at the API boundary. When a field is `False`, the allocator skips the entire group — no namespace dispatch occurs for that group.

`NamespaceConstraints` is the **fine-grained dispatch**: it decides how constraints are applied for a specific hardware target. It is hardware-aware and lives inside the context. When a constraint group is enabled (field is `True`), the allocator calls `TransferAndTensorContext`'s dispatch method, which delegates to each namespace strategy in `namespace_constraints`.

The allocator checks `ConstraintSelection` first. Only if a group is enabled does it proceed to call the corresponding `TransferAndTensorContext` dispatch method, which in turn iterates the namespace strategies and adds hardware-specific constraints.

This two-layer design means you can disable a constraint group entirely (via `ConstraintSelection`) without any code changes to the namespace strategies, and you can add a new hardware namespace (via a new `NamespaceConstraints` subclass) without changing the `ConstraintSelection` API.

---

See also: `.claude/skills/optimization/solver-backends.md` for the solver abstraction layer, `.claude/skills/constraints/` for the full MILP formulation and TransferAndTensorAllocator details.
