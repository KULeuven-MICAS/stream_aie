# Namespace Constraints

## Overview

The namespace constraints system provides hardware-specific MILP constraint dispatch using the Strategy pattern. A `NamespaceConstraints` base class defines three overridable constraint methods covering object-FIFO depth, buffer descriptors, and DMA channels. Each hardware namespace — for example, AMD AIE2 — provides a concrete subclass that overrides the methods relevant to its resource limits. The `TransferAndTensorContext` holds a tuple of namespace strategy instances and dispatches constraint calls to all of them. This design separates hardware-specific resource limits from the generic MILP formulation in `TransferAndTensorAllocator`, so adding support for a new hardware target requires no changes to the allocator itself. Source: `stream/opt/allocation/constraint_optimization/context.py`.

---

## NamespaceConstraints Base Class

`NamespaceConstraints` is the base class for all hardware-specific constraint strategies.

`NAMESPACE` is a class-level string attribute identifying the hardware namespace this strategy targets — for example, `"aie2"`. It is used by `applies_to()` to filter cores.

`applies_to(core)` returns `True` if `core.namespace == self.NAMESPACE`. Constraint methods use this check to iterate only over cores belonging to this namespace, ignoring cores that belong to other namespaces or have no namespace set.

The base class defines three overridable constraint methods, all implemented as no-ops:

- `add_object_fifo_constraints(model, object_fifo_depth)`: enforces per-core object-FIFO depth limits. Parameters are the solver model and a dictionary mapping `Core` objects to their accumulated FIFO depth linear expressions. A no-op in the base class; subclasses override this to add per-core upper-bound constraints.

- `add_buffer_descriptor_constraints(model, buffer_descriptor_depth)`: enforces per-core buffer descriptor count limits. Parameters are the solver model and a dictionary mapping `Core` objects to their accumulated buffer descriptor linear expressions. A no-op in the base class.

- `add_dma_usage_constraints(model, dma_usage_in, dma_usage_out)`: enforces per-core DMA channel limits in both directions. Parameters are the solver model and two dictionaries mapping `Core` objects to their accumulated incoming and outgoing DMA usage variables. Returns a list of DMA-related variables (empty list in the base class).

The no-op design means a subclass only needs to override the constraint methods relevant to its hardware. Methods left unoverridden remain silent — no constraints are added for that resource category on cores belonging to that namespace.

---

## AIE2Constraints

`AIE2Constraints(NamespaceConstraints)` is the concrete subclass for the AMD AIE2 tile array, with `NAMESPACE = "aie2"`.

The constructor accepts three DMA channel limit parameters and an off-chip core ID:

- `max_compute_tile_dma_channels` (default: 8) — DMA channel limit for compute tiles
- `max_mem_tile_dma_channels` (default: 6) — DMA channel limit for memory tiles
- `max_shim_tile_dma_channels` (default: 2) — DMA channel limit for shim tiles
- `offchip_core_id` — the ID of the off-chip (DRAM) core; used to identify shim tiles

These parameters come from `TransferMilpConfig` in `config.py` and are passed through `build_transfer_context()`.

### Object-FIFO Depth

For each core where `applies_to(core)` is `True`, `add_object_fifo_constraints` adds a constraint requiring that the accumulated FIFO depth expression for that core be at most `core.max_object_fifo_depth`. The `max_object_fifo_depth` attribute is set per-core in the hardware model and reflects the maximum number of FIFO slots available on that AIE2 tile.

### Buffer Descriptors

For each qualifying core, `add_buffer_descriptor_constraints` adds a constraint requiring that the accumulated buffer descriptor expression be at most `core.max_object_fifo_depth`. On AIE2 hardware, the same hardware register governs both the object-FIFO depth and the buffer descriptor count, so the same per-core limit applies to both.

### DMA Channels

DMA channel limits vary by tile type on the AIE2 array. Constraints apply independently to both the S2MM (incoming, store-to-memory) and MM2S (outgoing, memory-to-stream) directions. For each direction, the core's DMA usage expression must not exceed the tile-type-specific limit.

| Tile Type | Default Max DMA Channels | Identification |
|-----------|--------------------------|----------------|
| Compute | 8 | `core.type == "compute"` |
| Memory | 6 | `core.type == "memory"` |
| Shim | 2 | `core.id == offchip_core_id` |

The `get_max_dma_channels(core)` method selects the correct limit by checking `core.id` against `offchip_core_id` first (shim tile identification), then falling back to `core.type` for compute versus memory tiles.

`add_dma_usage_constraints` iterates over all DMA-participating cores, adds an upper-bound constraint for each core in both the incoming (`dma_usage_in`) and outgoing (`dma_usage_out`) directions, and returns any objective-penalty variables. The current implementation returns an empty list and does not add objective terms from `AIE2Constraints` — the DMA objective terms (`maxCoreDmaIn`, `maxCoreDmaOut`) are added by `TransferAndTensorAllocator._set_total_latency_and_objective` directly.

---

## TransferAndTensorContext

`TransferAndTensorContext` is a frozen dataclass that serves as the shared context for the transfer and tensor allocation MILP. Its key field is:

- `namespace_constraints: tuple[NamespaceConstraints, ...]` — a tuple of namespace strategy instances, one per detected hardware namespace

Other fields include `offchip_core_id`, `mem_cores`, `force_double_buffering`, and `force_io_transfers_on_mem_tile`, which carry topology information used directly by `TransferAndTensorAllocator`.

Three dispatch methods iterate over `namespace_constraints` and call the corresponding method on each strategy:

- `add_object_fifo_constraints(model, object_fifo_depth)` — calls each strategy's `add_object_fifo_constraints` method
- `add_buffer_descriptor_constraints(model, buffer_descriptor_depth)` — calls each strategy's `add_buffer_descriptor_constraints` method
- `add_dma_usage_constraints(model, dma_usage_in, dma_usage_out)` — calls each strategy's `add_dma_usage_constraints` method

`TransferAndTensorAllocator` calls these dispatch methods directly, not the individual namespace strategies. This means the allocator is fully agnostic to which hardware namespaces are present — it makes exactly three dispatch calls, and the context handles routing to the appropriate strategies.

---

## build_transfer_context Factory

`build_transfer_context(accelerator, ...)` is the factory function that assembles a `TransferAndTensorContext` for a given accelerator.

It scans the accelerator's core list to discover which hardware namespaces are present. For each recognized namespace, it instantiates the appropriate `NamespaceConstraints` subclass with the configured limit parameters.

Currently only `"aie2"` is recognized: if any core in the accelerator has `namespace == "aie2"`, an `AIE2Constraints` instance is created and added to the context. The DMA channel limits passed to `AIE2Constraints` come from the function's keyword parameters, which default to `max_compute_tile_dma_channels=8`, `max_mem_tile_dma_channels=6`, and `max_shim_tile_dma_channels=2`.

Extending to a new hardware namespace follows a two-step pattern:

1. Create a new subclass of `NamespaceConstraints`, set its `NAMESPACE` class attribute, and override the constraint methods relevant to the new hardware.
2. In `build_transfer_context()`, add a detection block: if any core has the new namespace string, instantiate the subclass and append it to `ns_constraints`.

No changes to `TransferAndTensorAllocator` or `TransferAndTensorContext` are needed. The three dispatch methods automatically delegate to the new strategy for cores matching its namespace.

---

## How the Layers Interact

Constraint dispatch in `TransferAndTensorAllocator` operates through three levels of filtering:

Level 1 — `ConstraintSelection` toggle (coarse): The allocator checks the boolean fields of its `ConstraintSelection` instance before doing anything else. If a field is `False`, the entire constraint group is skipped — no dispatch to `TransferAndTensorContext` occurs, and no namespace strategy is called. This is hardware-agnostic and lives at the API boundary.

Level 2 — `TransferAndTensorContext` dispatch (namespace routing): If a constraint group is enabled, the allocator calls the corresponding dispatch method on its `TransferAndTensorContext`. The dispatch method iterates over every strategy in `namespace_constraints` and calls each one's constraint method.

Level 3 — `applies_to()` core filtering (per-core): Inside each strategy's constraint method, `applies_to(core)` filters the core dictionaries to include only cores belonging to that namespace. Cores from other namespaces — or cores without a namespace — are silently skipped.

For reference, the constraint-selection guard applies to the following groups, which then flow through the remaining two levels when active:

| Enabled Group | Dispatch Method | Strategy Method |
|---------------|-----------------|-----------------|
| `memory_capacity` | (direct, no namespace dispatch) | (computed internally by TTA) |
| `object_fifo_depth` | `context.add_object_fifo_constraints` | `ns.add_object_fifo_constraints` |
| `buffer_descriptors` | `context.add_buffer_descriptor_constraints` | `ns.add_buffer_descriptor_constraints` |
| `dma_channels` | `context.add_dma_usage_constraints` | `ns.add_dma_usage_constraints` |

Note that memory capacity constraints are enforced directly inside `_memory_capacity_constraints()` using the per-core capacity values from the `Accelerator` model; they do not go through namespace dispatch.

See `.claude/skills/optimization/constraint-selection.md` for a full description of the two-layer interaction and the nonsensical-combination warning.

---

## See Also

- `.claude/skills/optimization/constraint-selection.md` — `ConstraintSelection` dataclass, toggle-to-hardware mapping, and the coarse toggle layer that gates namespace dispatch
- `.claude/skills/constraints/milp-formulation.md` — Full MILP structure for `TransferAndTensorAllocator`, including variable families and constraint groups
- `.claude/skills/optimization/solver-backends.md` — The `SolverModel` interface used by constraint methods when adding constraints to the model
