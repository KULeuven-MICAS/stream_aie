# Hardware Core Model

## Overview

Stream AIE is a framework for **general heterogeneous dataflow accelerators** composed of heterogeneous cores. The core is the fundamental unit of heterogeneity: different cores within the same accelerator may have different processing capabilities, different hardware architectures, and different cost characteristics. AMD AIE2 tile arrays and TPU-like PE-array accelerators are two example architectures; the framework is not limited to either.

The hardware model lives under `stream/hardware/architecture/`. The core object (`stream/hardware/architecture/core.py`) carries an identity, a role, a namespace, and an optional pluggable backend. The accelerator (`stream/hardware/architecture/accelerator.py`) holds a directed graph of cores connected by communication links.

The codebase default namespace is `"zigzag"` (see `default_namespace` in `stream/parser/core_validator.py`). The main tested end-to-end workload (ResNet18) runs on TPU-like hardware. AIE and TPU-like are co-equal examples in the framework.

---

## Core Roles (`core.type` / `core.kind`)

Every core has a **role** identifying how the pipeline should treat it. The role is the suffix of the fully-qualified `core_type` string (e.g. the `"compute"` in `"aie2.compute"`). It is accessed as `core.type` or equivalently `core.kind`.

The four allowed roles are defined in `ALLOWED_KINDS` in `stream/parser/core_validator.py`:

- `"compute"` — a processing tile that runs computation nodes. This is the common case. The MILP allocation assigns workload nodes to `compute` cores.
- `"memory"` — an on-chip memory or cache tile. Holds tensors and routes DMA traffic between compute tiles and off-chip memory. Present in AIE2 hardware (mem_tile_256KB) but optional in TPU-like designs.
- `"shim"` — a DMA interface tile bridging on-chip and off-chip data transfers. In the AIE2 column, row 0 is always the shim DMA tile. Does not run computation.
- `"offchip"` — an off-chip memory controller (DRAM). Present in TPU-like hardware as the top-level memory node. A PROJECT.md decision treats `"offchip"` like `"shim"` for transfer-type classification.

Roles drive the pipeline in several places: `GenericMappingGenerator._select_cores_for_node()` (in `stream/mapping/generic_generator.py`) has `_SKIP_TYPES = {"offchip", "shim"}` which excludes those roles from computation assignment; transfer-type classification treats `"offchip"` and `"shim"` cores as off-chip transfer endpoints (see the decision noted above); and `AIE2Constraints.get_max_dma_channels()` applies different DMA channel limits per role.

---

## Dataflow Architectures (`core.namespace`)

Every core also belongs to a **namespace** that identifies its dataflow architecture — the hardware platform it represents. The namespace is the prefix of `core_type` before the dot (e.g. the `"aie2"` in `"aie2.compute"`). It is accessed as `core.namespace`, which returns `core_type.split(".")[0]` if the string contains a dot, or `""` otherwise.

The two recognized namespaces are defined in `ALLOWED_NAMESPACES` in `stream/parser/core_validator.py`:

- `"aie2"` — AMD AIE2 tile array. Uses `AIE2CoreBackend` (`stream/hardware/architecture/backends/aie2.py`), a lightweight frozen dataclass carrying memory capacity and bandwidth. Uses `AIECostEstimator` for performance estimation. Dispatches to `AIE2Constraints` for FIFO depth, buffer descriptor, and DMA channel MILP constraints.

- `"zigzag"` — ZigZag-modeled core (e.g. a TPU-like PE array). Uses `ZigZagCoreBackend` (`stream/hardware/architecture/backends/zigzag.py`), which inherits from ZigZag's `Accelerator` class and carries a full `operational_array`, `memory_hierarchy`, and `dataflows` model. Uses `ZigZagCostEstimator` for performance estimation. No namespace-specific MILP constraints currently.

At the hardware level, namespace drives three things:
1. **Backend selection** — `AcceleratorFactory.create_core()` in `stream/parser/accelerator_factory.py` branches on `namespace` to construct `AIE2CoreBackend` (for `"aie2"`) or a ZigZag-backed core (for `"zigzag"`).
2. **Estimator selection** — `CoreCostEstimationStage.get_estimator()` in `stream/stages/estimation/core_cost_estimation.py` checks whether the core is an `aie2.compute` core (via `is_aie_compute_core()`), dispatching to `AIECostEstimator` or `ZigZagCostEstimator`.
3. **MILP constraint dispatch** — `build_transfer_context()` scans all core namespaces to instantiate the right `NamespaceConstraints` subclass.

For how namespace drives MILP constraint dispatch (NamespaceConstraints, AIE2Constraints, build_transfer_context), see `.claude/skills/constraints/namespace-constraints.md`.

---

## The Two Senses of "Core Type" — A Precision Callout

The word "type" is overloaded in the codebase. These must be distinguished:

- `core.core_type` — the **full** `"<namespace>.<kind>"` string (e.g. `"aie2.compute"`, `"zigzag.offchip"`). This is what is declared in YAML as `type: aie2.compute`.
- `core.type` (== `core.kind`) — the **kind suffix** only (e.g. `"compute"`, `"memory"`). The code sets `self.type = self.core_type.split(".")[-1]`.
- `core.namespace` — the **namespace prefix** only (e.g. `"aie2"`, `"zigzag"`).

When a YAML core file uses a bare kind string (e.g. `type: compute`), `CoreValidatorRegistry.normalize_core_type()` in `stream/parser/core_validator.py` promotes it to `"zigzag.compute"` using `default_namespace = "zigzag"`. If `type` is omitted entirely, the default is `"zigzag.compute"`.

---

## The Core and Accelerator Classes

### `Core` (`stream/hardware/architecture/core.py`)

`Core` is a thin identity and scheduling object. Hardware-specific details live in a pluggable backend (`self._backend`). Unknown attribute lookups fall through `__getattr__` to the backend, so `core.operational_array` or `core.memory_hierarchy` are resolved transparently.

Key fields set at construction:

- `id` (`int`) — unique integer index, matching the core's position in the core list.
- `name` (`str`) — human-readable tile name from YAML.
- `core_type` (`str`) — fully-qualified `"<namespace>.<kind>"` string.
- `type` (`str`) — derived from `core_type.split(".")[-1]`; the kind suffix alone.
- `utilization` (`int`, default `100`) — fraction of peak throughput in percent; used by `AIECostEstimator`.
- `max_object_fifo_depth` (`int`, default `0`) — maximum FIFO slots on this tile; AIE2-specific hardware limit enforced by `AIE2Constraints.add_object_fifo_constraints()`. Lives on `Core`, not on the backend.
- `col_id`, `row_id` (`int | None`) — 2-D grid position from `core_coordinates`; used by `build_transfer_context()` to filter memory cores within `nb_cols_to_use`.
- `_backend` (`AnyBackend | None`) — `AIE2CoreBackend` or `ZigZagCoreBackend` (see `stream/hardware/architecture/backends/__init__.py`).
- `operator_types` (`list[str] | None`) — not a constructor parameter; set dynamically by `AcceleratorFactory.create_core()`. `None` means the core accepts all operator types.

`has_same_performance(other)` is used by `CoreCostLUT` to reuse cached cost entries for structurally identical cores. For ZigZag-backed cores it compares `operational_array`, `memory_hierarchy`, and `dataflows`; for AIE2-backed cores it delegates to `AIE2CoreBackend.__eq__` (a frozen dataclass).

### Backends (`stream/hardware/architecture/backends/`)

`AnyBackend = ZigZagCoreBackend | AIE2CoreBackend` (exported from `stream/hardware/architecture/backends/__init__.py`). Both implement an informal three-method protocol (not enforced by an ABC): `get_memory_capacity() -> int`, `get_max_memory_bandwidth(type) -> int`, and `get_ir() -> dict`.

`AIE2CoreBackend` (`stream/hardware/architecture/backends/aie2.py`) is a frozen dataclass with `memory_capacity_bits`, `bandwidth_min`, `bandwidth_max`. It has no ZigZag dependency.

`ZigZagCoreBackend` (`stream/hardware/architecture/backends/zigzag.py`) inherits from ZigZag's `zigzag.hardware.architecture.accelerator.Accelerator`. It carries the full `operational_array`, `memory_hierarchy`, and `dataflows` model required by `ZigZagCostEstimator`, which calls `core.to_zigzag_core()` to extract it.

### `Accelerator` (`stream/hardware/architecture/accelerator.py`)

Key fields: `name`, `cores` (a `CoreGraph`, subclass of ZigZag's `DiGraphWrapper[Core]` — nodes are `Core` objects, edges carry `CommunicationLink` objects), `offchip_core_id` (integer ID of the DRAM core, or `None`), `nb_shared_mem_groups`, `communication_manager`.

Key methods: `get_core(core_id)` returns a `Core` by integer ID; `core_list` returns all cores as a list; `get_spatial_mapping_from_core(core_allocation)` returns the common dataflow for a set of core IDs; `get_ir()` serializes to a dict.

`CoreGraph` edges are added by `AcceleratorFactory.create_core_graph()` in `stream/parser/accelerator_factory.py`. Two connectivity types:
- `"link"` — directed point-to-point; creates two directed edges (bidirectional=False).
- `"bus"` — shared medium; creates a single `CommunicationLink` instance shared by all connected cores (bidirectional=True).

---

## YAML → Accelerator Parse Path

Parsing proceeds in this order:

1. `AcceleratorValidator` (cerberus, `stream/parser/accelerator_validator.py`) validates the top-level YAML schema — `name`, `cores` dict, `offchip_core_id`, `core_connectivity`, `core_coordinates`, `core_memory_sharing`. For each core YAML filename in `data["cores"]`, it loads the referenced file relative to `stream/inputs/`.

2. `CoreValidatorRegistry` in `stream/parser/core_validator.py` selects the per-type cerberus validator (e.g. `AIE2ComputeCoreValidator`, `ZigZagComputeCoreValidator`) and validates the individual core YAML.

3. `AcceleratorFactory.create()` in `stream/parser/accelerator_factory.py` iterates the validated data and calls `create_core()` per core. `create_core()` branches on `namespace`:
   - `namespace == "aie2"` → construct `AIE2CoreBackend`, create `Core`.
   - `namespace == "zigzag"` → run `ZigZagCoreFactory(core_data).create(core_id)`, upgrade class to `ZigZagCoreBackend`, create `Core`.
   - Unknown namespace → `ValueError`.

---

## Specialized Compute Cores via `operator_types`

`operator_types` is an **optional YAML field** on a core — it is not a new role or kind. It refines a `compute`-role core to restrict which operator types it handles. Both `zigzag.compute` and `aie2.compute` cores support it. `None` (the default) means the core accepts all operator types.

The pooling and SIMD cores in the TPU-like example are both `zigzag.compute` — they differ only in `operator_types`:
- `pooling.yaml` → `operator_types: [MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool]`
- `simd.yaml` → `operator_types: [Add, Relu]`

Core selection in `GenericMappingGenerator._select_cores_for_node()` (`stream/mapping/generic_generator.py`) follows a priority rule: (1) if any core's `operator_types` list contains `node.type`, use those specialized cores exclusively; (2) otherwise, use cores where `operator_types is None`; (3) final fallback: all `compute`-kind cores.

One subtlety: `operator_types` is read directly from the raw YAML dict by `AcceleratorFactory.create_core()` via `core_data.get("operator_types", None)` **before** cerberus validation runs. It is not declared in the cerberus schemas for either `AIE2BaseCoreValidator` or `ZigZagBaseCoreValidator`, so validation neither requires nor validates the field.

---

## Example Hardware Configurations

### AIE Single Column (`stream/inputs/aie/hardware/single_col.yaml`)

- Core 0: `shim_dma.yaml` → `aie2.shim` — DMA interface tile; `offchip_core_id = 0` (the shim acts as the off-chip interface in the MILP).
- Core 1: `mem_tile_256KB.yaml` → `aie2.memory` — 256 KB on-chip cache tile.
- Cores 2-5: `aie_tile.yaml` → `aie2.compute` — 64 KB L1, 12 FIFO slots each.
- Connectivity: linear `link` chain (shim → mem → compute tiles).

### TPU-Like Quad Core (`stream/inputs/examples/hardware/tpu_like_quad_core.yaml`)

- Cores 0-3: `tpu_like.yaml` → `zigzag.compute` — 2 MB SRAM, 32×32 PE array each.
- Core 4: `pooling.yaml` → `zigzag.compute` + `operator_types: [MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool]`.
- Core 5: `simd.yaml` → `zigzag.compute` + `operator_types: [Add, Relu]`.
- Core 6: `offchip.yaml` → `zigzag.offchip` — DRAM controller; `offchip_core_id = 6`.
- Connectivity: 2-D mesh among compute cores (`link` entries) + shared `bus` to offchip.

This TPU example demonstrates heterogeneous compute specialization **without** a distinct hardware architecture difference: pooling and SIMD are both ZigZag-backed `compute` cores, distinguished only by `operator_types`.

---

## Adding a New Core Type

### Scenario A: Specialized compute core via `operator_types` (YAML only)

This is a configuration-only change — no Python modifications required.

1. Create a new core YAML in `stream/inputs/<hardware>/hardware/cores/` with `type: zigzag.compute` (or `aie2.compute`) and an `operator_types: [...]` list.
2. Reference it in the top-level hardware YAML under `cores:`.
3. Add it to `core_connectivity:` so it is reachable from other compute cores.

The existing priority logic in `GenericMappingGenerator._select_cores_for_node()` and the MILP allocation machinery handle the rest automatically.

### Scenario B: New dataflow architecture namespace (3–5 file changes)

Adding a new namespace means a new hardware platform with its own backend, validator, and factory branch. These changes are **required**:

1. **`stream/parser/core_validator.py`** — Add the namespace string to `ALLOWED_NAMESPACES`. Define a `_<NS>_EXTRA_SCHEMA` dict with any namespace-wide YAML fields. Subclass an existing base validator (or create a new one) and register per-kind validators with `@CoreValidatorRegistry.register`.

2. **`stream/hardware/architecture/backends/`** — If the new architecture needs hardware properties not covered by an existing backend, add a new backend module (e.g. `mynewns.py`) implementing the three-method protocol: `get_memory_capacity()`, `get_max_memory_bandwidth()`, `get_ir()`. Export it from `__init__.py` and add it to the `AnyBackend` union. If an existing backend's property set is sufficient, reuse it.

3. **`stream/parser/accelerator_factory.py`** — Add an `if namespace == "mynewns":` branch in `create_core()` that constructs the new backend and `Core`.

4. **`stream/stages/estimation/core_cost_estimation.py`** — Update `get_estimator()` to detect the new namespace and dispatch to an appropriate estimator (see `performance-estimation.md`).

These changes are **optional**, needed only if the new hardware has FIFO/DMA-style constraints expressible as MILP:

5. **`stream/opt/allocation/constraint_optimization/context.py`** — Add a `NamespaceConstraints` subclass with `NAMESPACE = "mynewns"` and a detection block in `build_transfer_context()`. See `.claude/skills/constraints/namespace-constraints.md` for the two-step extension pattern.

6. **`stream/opt/allocation/constraint_optimization/config.py`** — Add `CoreConstraintProfile` entries in `default_core_profiles()` if the new architecture needs custom role or DMA limit configuration.

In summary: a new namespace absolutely requires the validator (#1), the factory branch (#3), and either a new backend (#2) or reuse of an existing one. The estimator update (#4) is required for correct cost modeling. The MILP constraint side (#5/#6) is optional if the hardware has no FIFO/DMA-style resource limits that differ from the generic model.

---

## See Also

- `.claude/skills/hardware/performance-estimation.md` — CoreCostEntry, get_estimator dispatch, AIE/ZigZag estimators, fallback, LUT, and how to add or swap an estimator
- `.claude/skills/constraints/namespace-constraints.md` — NamespaceConstraints, AIE2Constraints, and how namespace drives MILP constraint dispatch
- `.claude/skills/pipeline/` — Pipeline stage order, StageContext, and where CoreCostEstimationStage sits in the execution flow
