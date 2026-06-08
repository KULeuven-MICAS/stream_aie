# Per-Core Performance Estimation

## Where Estimation Fits in the Pipeline

`CoreCostEstimationStage` (`stream/stages/estimation/core_cost_estimation.py`) runs after `TilingGenerationStage` and before `ConstraintOptimizationAllocationStage`. It populates a `CoreCostLUT` with `CoreCostEntry` objects for each valid (node, core) pair. This LUT is the sole per-core-cost input to the MILP allocation stage: the MILP reads latencies from it when scheduling nodes to cores.

See `.claude/skills/pipeline/` for the full stage order and how `StageContext` carries the LUT forward.

---

## CoreCostEntry: The Common Result Type

`CoreCostEntry` (`stream/cost_model/core_cost.py`, dataclass) is what every estimator returns. All fields:

- `latency_total` — the primary latency cost used by the MILP for scheduling.
- `ideal_cycle` — theoretical minimum cycles (no memory bottleneck).
- `ideal_temporal_cycle` — ideal cycles including temporal mapping overhead.
- `energy_total` — energy estimate; currently `0.0` for the AIE path (marked TODO).
- `cme` — the raw `CostModelEvaluation` from ZigZag, or `None` for the AIE and fallback paths.
- `mapping` — the chosen temporal mapping; `None` for the AIE path.
- `metadata` — an estimator-specific dict (e.g. `{"utilization": 95}`) for debugging.

Every estimator returns this type — it is the shared contract that makes the LUT estimator-agnostic.

---

## Estimator Dispatch — The Seam

`CoreCostEstimationStage.get_estimator(core)` in `stream/stages/estimation/core_cost_estimation.py` is the dispatch seam. It is a straightforward `if/else`:

```
if is_aie_compute_core(core):
    return AIECostEstimator(workload, mapping)
return ZigZagCostEstimator(workload, accelerator, mapping, ...)
```

`is_aie_compute_core(core)` returns `True` when `str(core.core_type).startswith("aie2.")` and `core.type == "compute"`. All other cores — including `aie2.memory`, `aie2.shim`, and all `zigzag.*` cores — fall through to `ZigZagCostEstimator`.

**The dispatch is a plain `if/else` — there is no extension hook, no base class to inherit, and no function to call to register a new estimator.** Adding a new estimator means editing `get_estimator()` in the source file directly.

---

## AIECostEstimator

`AIECostEstimator` (`stream/stages/estimation/aie_cost_estimator.py`, dataclass) is used exclusively for `aie2.compute` cores.

Estimation logic:
1. Compute MACs = product of all layer dimension sizes, divided by the total inter-core tiling factor from the mapping.
2. Look up `ops_per_cycle` from a hardcoded table keyed by `(input_dtype, output_dtype, core_type)`. Example entries: `aie2.compute` + BFloat16 = 32 ops/cycle; `aie2.compute` + Float32 = 16 ops/cycle.
3. Adjust for `kernel.utilization` from the mapping: `cycles = ceil(macs / (ops_per_cycle * utilization / 100))`.
4. Return `CoreCostEntry` with `energy_total=0` (a TODO marker for future energy modeling).

The AIE estimator does **not** use ZigZag. It requires the mapping to carry a `kernel` block with a `utilization` field set.

---

## ZigZagCostEstimator

`ZigZagCostEstimator` (`stream/stages/estimation/zigzag_cost_estimator.py`, dataclass) is used for all non-AIE-compute cores — every `zigzag.*` core and any `aie2.*` core that is not of kind `"compute"`.

Estimation logic:
1. Build a `ZigZagLayerNode` from the `ComputationNode`, converting Stream's dimension/affine mapping representation into ZigZag's layer-equation format.
2. Instantiate a ZigZag `MainStage` pipeline: `SpatialMappingGeneratorStage → MinimalLatencyStage → TemporalMappingGeneratorStage → CostModelStage`.
3. Pass `core.to_zigzag_core()` — the inner `ZigZagCoreBackend` — as the accelerator. This is why ZigZag-backed cores are required for this estimator path.
4. Run the ZigZag pipeline; receive a `CostModelEvaluation`.
5. Apply `increase_cc_per_op()` for special operations: silu, sigmoid, and exp cost 4 cycles per operation (default: 1 cycle per operation).
6. Return `CoreCostEntry` with `latency_total = cme.latency_total2`.

A detail: `get_memory_operand_links()` uses `assert len(memory_operands) >= len(node.tensors)` (relaxed from `==`). This accommodates pooling cores that have extra memory operands (`I1/I2/O`) for operations that use fewer than three tensors.

---

## ZigZag Fallback

If `run_zigzag()` raises any exception — most commonly a spatial-mapping-generation crash for certain Conv configurations — `ZigZagCostEstimator.estimate()` catches it and falls back to:

```
ideal_cycle = product of all layer_dim_sizes values
```

The fallback is logged at WARNING level. The resulting `CoreCostEntry` has `energy_total=0`, `cme=None`, and `mapping=None`. This behavior is a documented project decision (PROJECT.md): "ZigZag fallback uses product of layer_dim_sizes as ideal-cycle estimate when spatial mapping generation crashes." The fallback fires on **any** exception from `run_zigzag()`, not only spatial-mapping errors.

---

## CoreCostLUT — Caching

`CoreCostLUT` (`stream/cost_model/core_cost_lut.py`) is a two-level dict `{ComputationNode → {Core → CoreCostEntry}}`, persisted to `output_path/core_cost_lut.pickle` and loaded at the start of each run.

Cache lookups use **semantic equality**, not object identity:
- Node lookup uses `node.has_same_performance(n)` — compares `layer_dim_sizes`, the operator equation, and other performance-determining fields.
- Core lookup uses `core.has_same_performance(c)` — for ZigZag-backed cores, compares `operational_array`, `memory_hierarchy`, and `dataflows`; for AIE2-backed cores, uses frozen-dataclass `__eq__`.

This means a re-run with an identical workload and hardware reuses all cached entries without re-invoking ZigZag or the AIE estimator. A best-effort YAML summary is also written to `core_cost_lut.yaml` (non-blocking).

---

## Adding or Swapping an Estimator — The Honest Seam

`get_estimator()` in `stream/stages/estimation/core_cost_estimation.py` is the only place to edit. Steps:

1. **Create the estimator class.** It must implement `estimate(node: ComputationNode, core: Core) -> CoreCostEntry`. There is no base class to inherit — just match the `CoreCostEntry` field contract. Place the file under `stream/stages/estimation/`.

2. **Edit `get_estimator()`** to add a detection branch **before** the final `return ZigZagCostEstimator(...)`. The branch can test any `core` attribute — most naturally `core.namespace`, `core.core_type`, or `core.type`.

3. **Context needs:** `CoreCostEstimationStage.__init__` passes `workload`, `accelerator`, `mapping`, `temporal_mapping_type`, `loma_lpf_limit`, and `nb_spatial_mappings_generated` when constructing `ZigZagCostEstimator`. A new estimator can take a subset of these or pull additional values from `self.ctx`.

4. **LUT compatibility:** `CoreCostLUT` stores `CoreCostEntry` objects; it is estimator-agnostic. After switching estimators, delete `core_cost_lut.pickle` from the output directory to force fresh estimation — stale pickle entries will otherwise be reused.

5. **Replacing ZigZag entirely:** To swap out `ZigZagCostEstimator` for all non-AIE cores, replace the final `return ZigZagCostEstimator(...)` with a return of the new estimator. Nothing else in the pipeline needs to change, because the downstream stages consume only the `CoreCostLUT` (containing `CoreCostEntry` objects).

**What NOT to expect:** There is no registration function, no common base class to subclass, and no runtime configuration flag for selecting estimators. The seam is one `if/else` method in one file. Developers edit the source directly.

---

## See Also

- `.claude/skills/hardware/core-model.md` — Core roles, namespaces, and the `core.namespace` / `core.core_type` / `core.type` distinction that drives estimator dispatch
- `.claude/skills/pipeline/` — Stage ordering and how `CoreCostEstimationStage` fits into the pipeline
- `.claude/skills/constraints/namespace-constraints.md` — How the same namespace concept drives MILP constraint dispatch on the allocation side
