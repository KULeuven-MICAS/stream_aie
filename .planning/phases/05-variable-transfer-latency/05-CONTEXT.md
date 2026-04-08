# Phase 5: Variable Transfer Latency - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Make transfer latency a tile-dependent linear expression in the CO model and eliminate the only `addGenConstrNL` call, making the entire model a pure MILP. Transfer latency is the last tile-dependent quantity still treated as a fixed scalar. After this phase, CO-03 is satisfied: no tile-dependent quantity remains as a fixed scalar in the MILP.

</domain>

<decisions>
## Implementation Decisions

### Latency Numerator Linearization
- **D-01:** Pre-compute per-candidate latency coefficients `latency_lut[k] = ceil(tensor_size_bits[k] / min_bw)` for each joint candidate combination, reusing the same `_joint_candidates_for_tensor` enumeration pattern from Phase 3. Bandwidth (`min_bw`) is path-specific but tile-independent, so only `tensor_size_bits` varies per candidate.
- **D-02:** For multi-dimensional tensors, joint candidate enumeration across tiled dimensions produces the linearization (same as Phase 3 memory constraints). Single-dimensional tensors use direct per-candidate coefficients.

### Eliminating addGenConstrNL — Pure MILP Latency
- **D-03:** The existing `_add_const_over_linexpr` uses `addGenConstrNL` (line 1707) for `constant / linexpr` division. This is the only nonlinear constraint in the entire model. It must be eliminated and replaced with a pure MILP formulation.
- **D-04:** Replace the `constant / reuse_factor` ratio with enumeration over (tile_candidate, stop_level) pairs. For each pair `(k, s)`, pre-compute `amortized_latency[k,s] = ceil(tensor_size[k] / min_bw) / reuse_factor_coeff[k,s]`. The active combination is selected by `joint_w[k] AND z_stop[s]` binary products, gated by path choice `y`. All scalar coefficients x binary variables — pure MILP.
- **D-05:** The (tile_candidate, stop_level) enumeration produces binary products `joint_w[k] * z_stop[s]` via `_add_binary_product`. The result is further gated by path choice `y` via `_add_binary_scaled_continuous`. This follows established patterns from Phases 3-4.

### Helpers to Refactor/Replace
- **D-06:** `_add_const_over_linexpr` and `_add_binary_times_const_over_linexpr` are eliminated or refactored. The new latency formulation does not need a ratio helper — it's all pre-computed coefficients x binary products. If these helpers are not used elsewhere, they can be removed entirely.
- **D-07:** `_transfer_latency_for_path` (line 243) stays as-is for computing per-candidate latency scalars (called once per candidate in the enumeration loop). It just won't be called with the default tensor shape anymore.

### Scope Confirmation
- **D-08:** Transfer latency is the ONLY remaining tile-dependent scalar in the CO model. All other quantities were linearized in Phases 3-4: tensor sizes (memory), SSIS loop sizes, reuse levels, fire counts, tiles_needed, bds_needed, FIFO depth, buffer descriptors.
- **D-09:** DMA constraints, force_nonconstant_reuse_levels, variable bounds, and computation node latencies are tile-independent and require no changes.

### Claude's Discretion
- Whether to keep `_add_const_over_linexpr`/`_add_binary_times_const_over_linexpr` as dead code or remove them
- How to structure the (k, s) pair enumeration loop (inline in `_active_transfer_latency` vs separate helper)
- Whether the latency cache `_transfer_latency_cache` needs restructuring for per-candidate values
- Unit test design for the new pure MILP latency formulation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Transfer Latency (Primary Modification Target)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 243-249 — `_transfer_latency_for_path()` static method computing `ceil(tensor.size_bits() / min_bw)`
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1923-1943 — `_active_transfer_latency()` method using scalar latency constant divided by reuse_factor via NL constraint
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1254-1257 — Slot latency constraint consuming `_active_transfer_latency()` result

### Nonlinear Constraint to Eliminate
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1665-1713 — `_add_const_over_linexpr()` with `addGenConstrNL` at line 1707
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 1756-1785 — `_add_binary_times_const_over_linexpr()` wrapper

### Reuse Factor (Already Linearized, Consumed by Latency)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` lines 819-860 — `_reuse_factor_rate_constraints()` with variable tile mode producing linear expression reuse_factors[tr]

### Phase 3-4 Infrastructure (Reusable Patterns)
- `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `_joint_candidates_for_tensor()` (line ~1819), `_joint_binary_for_combo()` (line ~362), `_add_binary_product()` (line 1787), `_add_binary_scaled_continuous()` (line 1715)
- `stream/opt/tile_size_utils.py` — `tensor_size_bits_for_candidate()`, `max_tensor_size_bits()`, `ssis_loop_sizes_for_candidate()`, `reuse_coefficients_for_sizes()`
- `stream/opt/search_space.py` — `SearchSpace` and `TileSizeOption` dataclasses

### Tests
- `tests/unit/test_co_tile_variables.py` — Existing CO tile variable tests (79 tests); Phase 5 extends these
- `tests/regression/test_baseline.py` — Regression tests; degenerate single-candidate must pass

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_joint_candidates_for_tensor(tensor, transfer)`: Returns `list[(size_bits, joint_binary_var)]` — reuse for computing per-candidate latency coefficients
- `_add_binary_product(a, b)`: Binary product linearization — use for `joint_w[k] * z_stop[s]` products
- `_add_binary_scaled_continuous(binary, continuous, ub)`: Exact linearization of `binary * continuous` — use for gating by path choice `y`
- `_ssis_coefficients_for_transfer(tr)`: Already enumerates (candidate, stop_level) combinations for reuse coefficients — the same iteration structure can produce latency coefficients
- `reuse_levels[(t, s)]` in variable mode: stores `[(fires_coeff, sf_coeff, joint_binary_var), ...]` — reuse_factor coefficients (`sf_coeff`) are needed for computing amortized latency

### Established Patterns
- Big-M continuous auxiliary: `lc <= expr`, `lc <= M*binary`, `lc >= expr - M*(1-binary)` — Phase 3/4 pattern
- `isinstance(rl_check, list)` check on `reuse_levels[(t, -1)]` to detect variable vs scalar mode
- Per-candidate coefficient enumeration with joint binary variables — Phase 3 memory, Phase 4 SSIS

### Integration Points
- `_active_transfer_latency(tr, choice, y)` is the sole method to refactor — transforms from NL ratio to pure MILP enumeration
- `_slot_scheduling_constraints()` (line 1254) consumes `_active_transfer_latency` — no changes needed there, just receives a different variable type
- `_transfer_latency_cache` may need restructuring if enumeration changes the caching key

</code_context>

<specifics>
## Specific Ideas

- The (tile_candidate, stop_level) enumeration mirrors the structure in `_ssis_coefficients_for_transfer()` which already loops over candidates and stop levels. The latency computation can be integrated into the same loop or done as a separate pass using the same data.
- For the degenerate single-candidate case: one `(k=0, s)` pair per stop level, collapsing to `ceil(fixed_tensor_size / min_bw) / reuse_factor[s]` — functionally identical to the current scalar path (minus the NL constraint).
- The `_add_const_over_linexpr` helper (and its wrapper) may become dead code after Phase 5 since latency was their only consumer. Whether to remove or keep as utility is Claude's discretion.
- The amortized_latency coefficients can be pre-computed alongside the reuse coefficients in `_init_transfer_fire_helpers()` since all the needed data (tensor sizes per candidate, reuse factors per candidate/stop) is available there.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 05-variable-transfer-latency*
*Context gathered: 2026-04-08*
