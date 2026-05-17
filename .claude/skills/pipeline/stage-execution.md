# Stage Execution Model

## Overview

The pipeline execution model is built on generator-based delegation. A `Stage` abstract base class defines
the interface, and stages compose by passing a list of "callables" (factory functions for each stage class)
that each stage uses to instantiate the next stage in the chain. `MainStage` serves as the entry point
that drives the generator chain. `StageContext` is a shared mutable dictionary that stages use to pass
data between each other.

All classes are defined in two files: `stream/stages/stage.py` (Stage, StageCallable, MainStage,
LeafStage) and `stream/stages/context.py` (StageContext).

---

## StageContext

**Source:** `stream/stages/context.py`

`StageContext` is a `@dataclass` defined in `stream/stages/context.py`. It wraps a `dict[str, Any]` with a
clean interface for reading and writing pipeline state.

**Fields:**

- `data: dict[str, Any]` -- The underlying dictionary holding all pipeline state. Stages should not access
  `data` directly; use `get()`, `set()`, `require_fields()`, or `require_value()` instead.

**Methods:**

- `from_kwargs(**kwargs) -> StageContext` -- Class method that constructs a `StageContext` from keyword
  arguments. Used by `stream/api.py` to set the initial context before pipeline execution begins.

- `get(key, default=None)` -- Read a value from the dictionary. Returns `default` if the key is absent.
  Equivalent to `dict.get(key, default)`.

- `set(**kwargs)` -- Merge keyword arguments into the dictionary, adding new keys or overwriting existing
  ones. Equivalent to `dict.update(kwargs)`.

- `require_fields(fields, stage_name)` -- Validates that all specified field names exist in the context and
  are not `None`. Raises `ValueError` listing the missing fields and the stage name. Called automatically
  by `Stage.__init__` if the stage declares `REQUIRED_FIELDS`.

- `require_value(key, stage_name)` -- Validates that a single field exists and is not `None`. Returns the
  field's value. Raises `ValueError` if the field is absent or `None`.

**The shared-context pattern:**

The context is shared and mutable. All stages in the pipeline receive the same `StageContext` instance.
A stage reads its input data via `get()` or `require_value()`, performs its work, and then writes results
back via `set()`. Later stages see the updated values. There is no explicit function-call return value
passing between stages -- data flows implicitly through the shared context. This means understanding
which stage writes a key is essential for debugging, since any later stage can overwrite it.

---

## Context Key Flow Table

This table maps each of the 8 active stages to the context keys it reads and the context keys it writes.

| Stage                                  | Context Keys Read                                                               | Context Keys Written                          |
|----------------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------|
| AcceleratorParserStage                 | `accelerator` (str path or Accelerator object)                                  | `accelerator` (Accelerator object)            |
| ONNXModelParserStage                   | `workload_path`, `output_path`                                                  | `onnx_model`, `workload`                      |
| MappingParserStage                     | `accelerator`, `workload`, `mapping_path`                                       | `mapping`                                     |
| TilingGenerationStage                  | `workload`, `mapping`, `output_path`                                            | `workload` (tiled), `mapping` (updated), `fusion_splits` |
| CoreCostEstimationStage                | `workload`, `accelerator`, `mapping`, `loma_lpf_limit`, `output_path`, `temporal_mapping_type` | `workload`, `accelerator`, `cost_lut` |
| ConstraintOptimizationAllocationStage  | `workload`, `accelerator`, `mapping`, `cost_lut`, `fusion_splits`, `output_path`, `backend`, `constraint_selection` | `workload`, `mapping`, `scheduler` |
| MemoryAccessesEstimationStage          | `workload`, `accelerator`, `mapping`, `scheduler`, `output_path`                | `workload`, `accelerator`, `memory_accesses`  |
| MappingGenerationStage                 | `accelerator`, `workload`, `output_path`; optional: `seq_len_tile_size`, `embedding_tile_size`, `hidden_tile_size`, `last_gemm_down`, `max_nb_mappings`, `nb_rows_to_use`, `nb_cols_to_use` | `workload`, `mapping_path`, `output_path` (per variant) |

---

## Stage ABC

**Source:** `stream/stages/stage.py`

`Stage` is an abstract base class (using `metaclass=ABCMeta`) that defines the interface for all pipeline
stages except `MainStage`.

**Class attribute:**

- `REQUIRED_FIELDS: tuple[str, ...] = ()` -- A tuple of context key names that this stage requires. If
  non-empty, `Stage.__init__` automatically calls `ctx.require_fields(REQUIRED_FIELDS, class_name)` to
  validate required data is present before the stage runs.

**Constructor:**

- `__init__(list_of_callables, ctx)` -- Stores the callable chain and context. Validates two invariants:
  leaf stages must have an empty `list_of_callables`; non-leaf stages must have a non-empty
  `list_of_callables`. Failing either raises `ValueError`. Then validates `REQUIRED_FIELDS` if declared.

**Methods:**

- `is_leaf() -> bool` -- Returns `False` by default. Leaf stages override this to return `True`. Used by
  `__init__` to enforce the callables constraint above.

- `run() -> StageContext` -- Abstract method. Non-leaf stages implement this to do work, update context,
  and delegate to the next stage by calling `yield from sub_stage.run()`. Leaf stages implement this to
  yield the context directly without delegating.

- `__iter__()` -- Returns `self.run()`, making any Stage instance iterable. This allows stages to be used
  in `for` loops and `list()` calls.

**The delegation pattern:**

A non-leaf stage's `run()` method performs its work, updates context, then instantiates the next stage by
calling `self.list_of_callables[0](self.list_of_callables[1:], self.ctx)` and doing
`yield from sub_stage.run()`. Each stage peels off the first callable from the front of the list and
passes the remainder to the next stage. This forms a recursive chain that unwinds when it reaches a
`LeafStage`. The pattern is sometimes called a "Russian nesting doll" -- each stage wraps the remainder
of the pipeline.

---

## StageCallable Protocol

**Source:** `stream/stages/stage.py`

`StageCallable` is a `Protocol` (with `runtime_checkable=True`) that defines the callable signature:
`(list_of_callables: list[StageCallable], ctx: StageContext) -> Stage`. Any callable satisfying this
signature can be used in the pipeline.

In practice, stage classes themselves satisfy the `StageCallable` protocol through their `__init__`
method -- you pass the class object (not an instance) into the callable list. The class's `__init__` takes
exactly `(list_of_callables, ctx)` and returns an instance of the class (a `Stage`). This means a pipeline
is assembled as a list of classes:

    [AcceleratorParserStage, ONNXModelParserStage, MappingParserStage, ...]

Each class in the list is a `StageCallable`. `MainStage` calls `callables[0](callables[1:], ctx)` to
instantiate the first stage, then the first stage calls `callables[0](callables[1:], ctx)` to instantiate
the second stage (where `callables` is now the tail of the original list), and so on.

---

## MainStage

**Source:** `stream/stages/stage.py`

`MainStage` is the pipeline entry point. It is **not** a subclass of `Stage` -- it does not inherit from
`Stage` and does not implement `run()` as a generator. Its `run()` method returns `list[StageContext]`
(a plain list, not a generator), making it a terminal collector rather than a link in the generator chain.

`MainStage.run()` calls `self.list_of_callables[0](self.list_of_callables[1:], self.ctx)` to create and
instantiate the first stage in the chain, then iterates over `stage.run()` using a plain `for` loop,
collecting all yielded contexts into a list. It returns that list to the caller.

Typically the pipeline yields exactly one `StageContext` (a single solution). The design supports multiple
results for future extensions such as multi-objective exploration, but `stream/api.py` currently asserts
exactly one result after calling `mainstage.run()`.

`stream/api.py` instantiates `MainStage` with the assembled stage list and the initial context, calls
`.run()`, and returns the single result context to the API caller.

---

## LeafStage

**Source:** `stream/stages/stage.py`

`LeafStage` is the terminal stage in the delegation chain. It inherits from `Stage`, overrides `is_leaf()`
to return `True`, and its `run()` method simply yields `self.ctx`. Its `__init__` asserts that
`list_of_callables` is empty (there is nothing left to delegate to), which is enforced by `Stage.__init__`
as well.

`LeafStage` itself is used directly in utility pipelines such as `parse_accelerator_ir` in `stream/api.py`,
where the pipeline is just `AcceleratorParserStage -> LeafStage`. In the standard `optimize_allocation_co`
pipeline, `MemoryAccessesEstimationStage` acts as the implicit leaf: it overrides `is_leaf()` to return
`True` and yields `self.ctx` directly via `yield from (self.ctx,)` without using `LeafStage` as a
separate final stage.

---

## How Stages Compose

This section walks through a full pipeline execution end-to-end.

**Setup in api.py:**

`stream/api.py` creates a `StageContext` via `StageContext.from_kwargs(...)` with initial values
(hardware path, workload path, configuration parameters). It also creates a list of stage callables
(the stage classes in execution order). `MainStage` is then instantiated with these two arguments.

**Execution:**

1. `MainStage.run()` calls `callables[0](callables[1:], ctx)` -- this instantiates the first stage
   (e.g., `AcceleratorParserStage`) by calling its `__init__`. The `__init__` validates `REQUIRED_FIELDS`
   and stores the remaining callables and the context.

2. `MainStage.run()` then calls `.run()` on the first stage and iterates over its output.

3. Inside `AcceleratorParserStage.run()`: the stage parses the hardware YAML, updates the context with the
   `Accelerator` object via `ctx.set(accelerator=accelerator)`, then calls
   `self.list_of_callables[0](self.list_of_callables[1:], self.ctx)` to instantiate the second stage
   (`ONNXModelParserStage`). It then does `yield from sub_stage.run()`.

4. Each subsequent stage repeats the same pattern: validate, work, update context, delegate to next stage,
   yield from next stage's results.

5. The final stage (`MemoryAccessesEstimationStage` in the standard pipeline) overrides `is_leaf()` to
   return `True`. Its `run()` does the final context update and then yields `self.ctx` directly.

6. The `yield` unwinds back through the entire generator stack: each `yield from` propagates the yielded
   context upward until `MainStage.run()` collects it in its `answers` list.

7. `MainStage.run()` returns `[ctx]`. `api.py` asserts `len(answers) == 1` and returns `answers[0]`.

**Key insight:** Each stage "wraps" the remainder of the pipeline. The first stage in the list runs first
and is the last to return. The final stage runs last and is the first to yield a result. This nesting
means that if you need to run cleanup or post-processing after the entire downstream pipeline completes,
you can do so in an outer stage after its `yield from sub_stage.run()` call.

---

## Adding a New Stage

To add a stage to the pipeline:

1. Create a class inheriting from `Stage` in an appropriate file under `stream/stages/`.

2. Define `REQUIRED_FIELDS` as a class attribute listing the context keys the stage requires. These will
   be validated automatically in `__init__` before `run()` is called.

3. Implement `run()` as a generator: read from context using `get()` or `require_value()`, perform the
   stage's work, write results back using `ctx.set(...)`, then delegate to the next stage:
   `yield from self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run()`.

4. Add the class to the appropriate stage list in `stream/api.py` at the correct position in the execution
   order. The stage must appear after any stage that writes context keys it reads, and before any stage
   that reads context keys it writes.

If the new stage is the final stage in a pipeline (no further delegation needed), inherit from `LeafStage`
instead of `Stage`, override `run()` to yield `self.ctx`, and do not add any callables after it in the
stage list.

---

## See also

- `.claude/skills/pipeline/pipeline-stages.md` -- Per-stage details: responsibility, context key tables,
  two pipeline variant descriptions, and ASCII flow diagram.
- `.claude/skills/optimization/` -- Solver backend details used by `ConstraintOptimizationAllocationStage`,
  including `ConstraintSelection`, solver backends, and the MILP formulation.
