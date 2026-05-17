# MILP Formulation

## Overview

The TETRA constraint optimization pipeline uses two MILP (Mixed-Integer Linear Programming) models in sequence. First, `ComputeAllocator` assigns computation nodes to hardware cores and time slots. Then, `TransferAndTensorAllocator` decides where tensors are stored and how data transfers are routed across the communication network. Both models are built through the solver facade (`SolverModel` ABC) and solved using either Gurobi or OR-Tools backends. The build pipeline for both follows the same pattern: prepare data, create variables, add constraints, set objective, solve. Source files: `stream/opt/allocation/constraint_optimization/allocation.py` (ComputeAllocator, 646 lines) and `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` (TransferAndTensorAllocator, 2155 lines).

---

## Two-Stage Allocation

The two MILPs execute sequentially, with the first stage's output serving as a fixed input to the second.

ComputeAllocator runs first. It determines which computation node runs on which hardware core, how many cores each node uses (k-split factor), and which time slot each node occupies. Its output — the node-to-core-and-slot mapping — becomes a fixed parameter in the second stage.

TransferAndTensorAllocator runs second. Given the fixed node placement, it determines where each tensor is stored (which memory cores), which routing path carries each data transfer between cores, how data reuse levels are structured, and how DMA channels are allocated. This is the larger and more complex of the two models, at 2155 lines.

Both models minimize total execution latency, accounting for pipelining overlap: when the workload runs for multiple iterations, steady-state schedules can be pipelined so that idle periods on one resource overlap with active periods on another. The objective captures this by subtracting an overlap term from the raw iteration latency.

---

## ComputeAllocator

`ComputeAllocator` solves the node-to-core placement problem. It answers: for each computation node in the workload, which core(s) should run it, and in which time slot?

### Build Pipeline

`get_optimal_allocations()` is the public entry point. It calls `_prepare_constants()` to precompute all static data into a frozen `ComputeAllocatorConstants` dataclass — latencies, energies, dependency maps, node groups, per-core weights, and per-core capacities. Then `_build_constraint_model(const)` constructs the MILP by running:

    _create_basic_sets  →  _create_variables  →  constraint methods  →  _set_objective

Each step reads from `ComputeAllocatorConstants` and writes solver variables and constraints into the `SolverModel`.

### Decision Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `k_vec[n,k]` | Binary | One-hot encoding of how many cores node `n` uses (k-split factor); exactly one value of `k` is 1 per node |
| `k_splits[n]` | Integer | Derived k-split count for node `n`, equal to the sum of `k * k_vec[n,k]` over all `k` |
| `core_asgn[c,n]` | Binary | Whether core `c` is assigned to node `n` |
| `slot_asgn[s,n]` | Binary | Whether node `n` is assigned to time slot `s` (one-hot over slots) |
| `asgn[c,s,n]` | Binary | Whether node `n` runs on core `c` in slot `s` (3D assignment tensor) |
| `slot_idx[n]` | Integer | Derived slot index for node `n`, computed as the weighted sum of `slot_asgn` |
| `lat_id_core[n,c]` | Integer | Latency of node `n` on core `c`, selected from the cost LUT based on `k_vec` |
| `lat_core_slot[c,s]` | Integer | Latency contribution of core `c` in slot `s` |
| `lat_slot[s]` | Integer | Maximum latency across all cores in slot `s` |
| `w_split[n]` | Integer | Per-node weight (capacity usage) normalized to the node's k-split |
| `w_core[c]` | Integer | Total weight allocated to core `c` across all nodes |
| `idle_start[c,s]` / `idle_end[c,s]` | Binary | Idle period indicators: 1 if core `c` has not yet started (or has finished) by slot `s` |
| `idle_sum[c]` | Integer | Total idle slot-latency for core `c`, summed across all idle slots |
| `lat_iter` | Integer | Per-iteration latency, equal to the sum of all slot latencies |
| `idle_min` | Integer | Minimum idle time across all cores, representing achievable pipelining overlap |
| `total_lat` | Integer | Total latency across all iterations, accounting for overlap |

### Constraint Groups

Each constraint group is implemented as a separate method called by `_build_constraint_model`:

- `_add_k_split_constraints`: Each node selects exactly one k-split factor via the one-hot `k_vec` encoding; `k_splits[n]` is defined as the resulting integer value.
- `_add_core_assignment_constraints`: The number of cores assigned to a node must equal its k-split value; each core is assigned to at most one node per slot; the allowed (core, k-split) combinations are gated by a precomputed split table from the cost LUT.
- `_add_slot_assignment_constraints`: Each node is assigned to exactly one slot; the 3D `asgn[c,s,n]` tensor is consistent with both `core_asgn` and `slot_asgn`; at most one node runs on each core per slot.
- `_add_group_constraints`: Nodes in the same group (e.g., tiles of the same layer) must share the same core assignment pattern — they use identical cores across slots.
- `_add_dependency_constraints`: If node A depends on node B, A's slot index must be strictly greater than B's, enforcing topological ordering.
- `_add_weight_constraints`: Per-core capacity limits — the combined weight of all nodes assigned to a core must not exceed that core's capacity.
- `_add_latency_constraints`: Selects the correct latency from the cost LUT based on the core and k-split assignment; `lat_slot[s]` is the maximum latency across all cores active in slot `s`. Because the objective minimizes total latency (which sums `lat_slot`), lower-bounding constraints are sufficient — the solver drives `lat_slot` to its true maximum.
- `_add_overlap_constraints`: Computes idle start/end indicators for each core-slot pair to determine pipelining overlap. `idle_sum[c]` accumulates slot latency during idle periods. `idle_min` is the minimum of all `idle_sum` values, representing the overlap achievable when the steady-state schedule repeats across iterations.

### Objective

    minimize total_lat = iterations * lat_iter - (iterations - 1) * idle_min

`lat_iter` is the sum of all slot latencies (the per-iteration cost). `idle_min` is the minimum total idle time across all cores, representing the maximum achievable overlap during pipelining. When `iterations = 1`, the overlap term vanishes and the objective reduces to minimizing the sum of slot latencies.

---

## TransferAndTensorAllocator Build Pipeline

`TransferAndTensorAllocator` solves the tensor placement and transfer routing problem. Its constructor categorizes the workload into computation nodes, transfer nodes, tensors, and their possible allocations, then calls `_build_model()`.

The build pipeline is:

    _create_vars()  →  _index_choice_metadata()  →  _create_constraints()  →  _overlap_and_objective()

Variable creation uses double-underscore private helpers (`__create_tensor_placement_vars`, `__create_transfer_path_vars`, `__create_reuse_vars`, `__create_slot_latency_vars`). Choice metadata indexing precomputes per-choice link sets, source/destination core sets, and empty-path flags. Constraint groups are each a separate single-underscore method. The overlap and objective phase computes idle periods, DMA accounting, and sets the final objective.

---

## TTA Variable Families

The primary decision variables use short prefix identifiers that appear throughout the constraint code:

| Prefix | Variable | Type | Purpose |
|--------|----------|------|---------|
| `x_` | `x_tensor_choice[(t, choice)]` | Binary | Whether tensor `t` uses placement `choice` (a tuple of cores); one-hot over all placement options for `t` |
| `y_` | `y_path_choice[(tr, choice)]` | Binary | Whether transfer `tr` uses routing path `choice` (a `MulticastPathPlan`); one-hot over all path options for `tr` |
| `z_` | `z_stop[(t, stop)]` | Binary | Whether tensor `t`'s reuse stop level is `stop` (one-hot over reuse levels from -1 to len(temporal_vars)-1) |
| `L_` | `slot_latency[s]` | Integer | Latency of time slot `s`; lower-bounded by all compute and transfer latencies assigned to that slot |
| `fires_` | `fires[tr]` | Integer | Number of times transfer `tr` fires per iteration (derived from the selected reuse stop level) |
| `reuse_factor_` | `reuse_factors[tr]` | Integer | Data reuse multiplier for transfer `tr` (also derived from the reuse stop level) |

Additional variables are created during the overlap and objective computation: `idleS`/`idleE` (binary idle indicators per resource per slot), `idle_lat` (integer idle latency per resource, computed using binary-times-continuous linearization), `overlap` (integer, the minimum idle latency across all resources), and `link_used` (binary, whether a communication link is active at all). When DMA channel constraints are enabled, the model also creates `coreDmaIn`/`coreDmaOut` (integer, per-core directional DMA usage counts) and `maxCoreDmaIn`/`maxCoreDmaOut` (integer, maximum DMA usage across all cores, included in the objective).

---

## TTA Constraint Groups

`_create_constraints()` calls each constraint group method in sequence.

### Always-Active Constraints

These constraints are unconditionally part of the model and are never guarded by `ConstraintSelection`:

- `_tensor_placement_constraints`: Each tensor with more than one possible placement selects exactly one choice — the `x_` variables for each tensor form a one-hot vector.
- `_path_choice_constraints`: Each transfer selects exactly one routing path — the `y_` variables for each transfer form a one-hot vector. Additional sub-constraints enforce source-tensor coherence (if a path routes from core A, the source tensor must be placed on core A) and destination-tensor coherence. Empty-path choices (zero-link paths) require source and destination tensors to be co-located on a common core.
- `_transfer_fire_rate_constraints`: Defines `fires_[tr]` as the fire count implied by the selected reuse stop level. The fire count determines how many times a transfer executes per iteration; it decreases as the reuse level increases.
- `_reuse_factor_rate_constraints`: Defines `reuse_factor_[tr]` as the reuse multiplier implied by the selected reuse stop level. A higher reuse factor means the same data is reused more times, reducing transfer frequency.
- `_link_contention_constraints`: Each communication link can carry at most one active transfer per time slot. Transfers assigned to the same slot that share a link are mutually exclusive.
- `_slot_latency_constraints`: `slot_latency[s]` is lower-bounded by the latency of every compute node and every active transfer path assigned to slot `s`. Because the objective minimizes total latency, the lower bounds drive `slot_latency` to its true value.
- `_force_nonconstant_reuse_levels`: For compute-to-compute transfers, forces the reuse stop level to be at or above the outermost irrelevant temporal loop, preventing unnecessary re-transfers.
- `_force_final_output_reuse_levels`: For compute-to-memory transfers (final outputs), forces the reuse stop level to be at or above the outermost irrelevant temporal loop in the input tensor.
- `_ensure_memory_and_compute_reuse_compatibility`: For compute-to-memory and memory-to-compute transfers, forces input and output tensors to use the same reuse stop level, ensuring consistency between the compute and memory sides of the transfer.
- `_force_reuse_includes_spatial`: Reuse stop levels must be at or above the outermost spatial loop variable. This ensures that tensors buffered across spatial dimensions are properly accounted for.

### ConstraintSelection-Guarded Constraints

See `.claude/skills/optimization/constraint-selection.md` for the full toggle-to-hardware mapping, including the nonsensical-combination warning for `memory_capacity=False` with `object_fifo_depth=True`.

The following methods are conditionally called based on the `ConstraintSelection` instance passed to the allocator. When a field is `False`, the allocator logs a WARNING and skips the entire group:

| ConstraintSelection Field | Guarded Method | Location | Effect When Disabled |
|---------------------------|----------------|----------|---------------------|
| `memory_capacity` | `_memory_capacity_constraints()` | `_create_constraints()` | Per-core memory usage is no longer bounded; tensors may exceed physical on-chip memory limits |
| `object_fifo_depth` | `_object_fifo_depth_constraints()` | `_create_constraints()` | Object-FIFO depth limits are not enforced; may over-subscribe FIFO slots on AIE2 tiles |
| `buffer_descriptors` | `_buffer_descriptor_constraints()` | `_create_constraints()` | Buffer descriptor count limits are not enforced; may exceed the hardware-imposed BD count |
| `dma_channels` | `_add_dma_usage_constraints()` | `_overlap_and_objective()` | DMA channel limits are not enforced AND DMA usage terms are removed from the objective |

The DMA constraints live in `_overlap_and_objective()` rather than `_create_constraints()` because DMA accounting depends on the idle indicator variables and overlap-phase variables created during that phase. When `dma_channels` is disabled, the objective function simplifies from `minimize total_lat + maxCoreDmaIn + maxCoreDmaOut` to just `minimize total_lat`.

The guarded constraint methods — `_memory_capacity_constraints`, `_object_fifo_depth_constraints`, and `_buffer_descriptor_constraints` — all use `_add_binary_product` to linearize the conjunction of tensor placement and reuse stop level selections. They then call the corresponding dispatch method on `TransferAndTensorContext`, which routes to the appropriate `NamespaceConstraints` strategy.

---

## TTA Overlap and Objective

`_overlap_and_objective()` computes the pipelining overlap and constructs the final objective.

The method tracks idle periods on each resource — both communication links and compute cores. For each resource, prefix and suffix accumulators track cumulative activity from the start and end of the schedule, respectively. Binary idle indicators `idleS[res, s]` and `idleE[res, s]` are 1 when the resource has not yet been active by slot `s` (prefix idle) or has finished all activity before slot `s` (suffix idle).

Idle latency per resource is the sum of slot latencies during idle periods. Because this requires multiplying binary idle indicators by the (continuous) slot latency variables, the computation uses `_add_binary_scaled_continuous` to produce an exact linear formulation. The per-resource idle latency is then stored in `idle_lat[res]`.

The `overlap` variable is constrained to be at most each resource's idle latency:

    overlap <= idle_lat[res]  for all resources

This forces `overlap` to equal the minimum idle latency across all resources, representing the maximum pipelining benefit achievable when the steady-state schedule repeats.

Total latency and objective:

    total_lat = iterations * sum(slot_latency) - (iterations - 1) * overlap

When DMA channel constraints are enabled:

    minimize total_lat + maxCoreDmaIn + maxCoreDmaOut

When DMA channel constraints are disabled:

    minimize total_lat

The DMA terms in the objective act as a soft pressure: even though the hard DMA limits come from `NamespaceConstraints` (via `AIE2Constraints`), including the peak DMA usage in the objective discourages the solver from concentrating all DMA activity on a single core when a more balanced allocation is available.

---

## Linearization Helpers

Both allocators use standard MILP linearization techniques. The following helper methods in `TransferAndTensorAllocator` encapsulate the most common patterns:

`_add_binary_product(a, b, base_name)` linearizes the AND of two binary variables. It introduces an auxiliary binary variable `w` with three constraints: `w <= a`, `w <= b`, `w >= a + b - 1`. This is the standard LP relaxation of a binary product. It is used heavily in memory capacity and buffer descriptor constraints, where two binary conditions — tensor placement choice and reuse stop level — must both hold simultaneously.

`_add_binary_scaled_continuous(binary_var, continuous_var, continuous_ub, base_name)` linearizes the product of a binary variable and a continuous variable using the McCormick envelope. The result `z` satisfies four constraints that collectively force `z = binary_var * continuous_var` when both variables are in their valid ranges. This helper is used throughout the idle latency computation, where each slot latency (continuous) is scaled by an idle indicator (binary).

`_add_binary_times_const_over_linexpr(binary_var, numerator, denominator_expr, ...)` linearizes the expression `binary_var * (numerator / denominator_expr)`. This is the only backend-dependent linearization path in the codebase. When the active backend supports nonlinear constraints (Gurobi, via `model.supports_nonlinear`), it uses `add_genconstr_nl` to encode the exact division. When the backend is linear-only (OR-Tools), it falls back to `_add_const_over_discrete_denominators`, which uses piecewise enumeration: since the denominator can only take discrete values determined by the one-hot `z_stop` selector, the division is expressed as a weighted sum `sum(z_k * (numerator / d_k))`. After computing the ratio, `_add_binary_scaled_continuous` gates the result by the binary variable. This helper is used in transfer latency computation where slot latency must be divided by the reuse factor and then gated by the path choice variable.

---

## See Also

- `.claude/skills/optimization/constraint-selection.md` — The `ConstraintSelection` dataclass, toggle-to-hardware mapping, and the two-layer interaction between coarse toggles and namespace dispatch
- `.claude/skills/constraints/namespace-constraints.md` — `NamespaceConstraints` base class, `AIE2Constraints`, `TransferAndTensorContext` dispatch, and hardware-specific constraint details
- `.claude/skills/optimization/solver-backends.md` — The `SolverModel` ABC, `GurobiBackend`, `ORToolsBackend`, and the `supports_nonlinear` dispatch used by linearization helpers
