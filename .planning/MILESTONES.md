# Milestones

## v2.0 Variable Tile Size Optimization (Shipped: 2026-04-08)

**Phases completed:** 7 phases, 15 plans, 22 tasks

**Key accomplishments:**

- One-liner:
- BIG BOY baseline fixture (seq=256, emb=2048, hid=8192, tiles 16/128/32) captured at latency_total=922357343 with 14 fire counts and 15 z_stop entries; 4-function slow regression test suite asserts CO determinism.
- SearchSpace/TileSizeOption frozen dataclasses and 3 pure utility functions (tensor_size_bits, is_divisible_candidate, passes_single_tensor_memory_check) created with 13 unit tests passing, no Gurobi dependency.
- MappingParserStage now extracts tile_options_raw from YAML and sets it in StageContext; CandidateFilterStage filters candidates by divisibility and builds SearchSpace; wired into api.py pipeline. 5 unit tests + regression tests pass.
- SearchSpace threaded end-to-end from StageContext to TransferAndTensorAllocator as None-default kwarg; tensor_size_bits_for_candidate and max_tensor_size_bits added to tile_size_utils.py with 5 new unit tests
- w[dim,k] binary variables, tile_var[dim] INTEGER variables, one-hot constraints, and _joint_candidates_for_tensor joint enumeration added to TransferAndTensorAllocator with 14 unit tests
- `stream/workload/steady_state/iteration_space.py`
- Four CO constraint methods (fire rate, reuse factor, FIFO depth, BD) linearized via big-M lc auxiliaries when reuse_levels/tiles_needed_levels/bds_needed_levels hold candidate-indexed lists; scalar path preserved for backward compatibility.
- Pure MILP _active_transfer_latency via (tile_candidate, stop_level) enumeration with pre-computed amortized coefficients, eliminating the only addGenConstrNL call in the CO model
- One-liner:
- One-liner:
- 1. [Rule 1 - Bug] Updated unit test for scalar fallback path
- One-liner:
- One-liner:

---
