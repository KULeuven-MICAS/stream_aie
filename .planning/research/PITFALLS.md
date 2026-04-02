# Pitfalls Research

**Domain:** Adding variable tile size selection to an existing Gurobi-based MILP constraint optimizer for AIE accelerator allocation
**Researched:** 2026-04-02
**Confidence:** HIGH (grounded in Gurobi official documentation, direct codebase analysis, and MILP formulation literature)

---

## Critical Pitfalls

### Pitfall 1: Loose Big-M Bounds When Tile Size Makes Values Variable

**What goes wrong:**
The existing `big_m` in `transfer_and_tensor_allocation.py` is set to `len(workload.nodes()) + 5` — a small integer that worked because tensor/transfer sizes were constants baked in before model construction. Once tile size selection is introduced, constraint coefficients that were previously scalars (e.g. `req_size = ceil(size_factor * tensor_size)` at line 733) become functions of the selected tile. Any big-M constraint that gates on whether a tensor is allocated to a core (`u`) combined with reuse level (`z_stop`) now has a coefficient that depends on which tile was chosen. If the M bound is computed from a worst-case tile size but not updated to reflect the tightest bound for each tile option, the LP relaxation becomes loose and branch-and-bound explores exponentially more nodes.

**Why it happens:**
Developers naturally reuse the existing `self.big_m` scalar and extend it to cover variable-size scenarios by using the maximum possible tile size. This gives a valid but excessively loose bound. For example, if tiles are `[16, 32, 64, 128]` and a constraint reads `req_size * uz <= M`, setting M to `128 * max_size_factor` instead of the per-tile-selection-specific bound creates an LP relaxation gap proportional to the ratio of tile sizes (8x in this case).

**How to avoid:**
Compute per-selection big-M values. When a tile size selection variable `s_k` (one-hot over K tile options) is introduced, any constraint that multiplies a tile-dependent coefficient by a binary should use the tightest coefficient specific to that tile option. Use indicator constraints (`model.addGenConstrIndicator`) when per-option M values are different; Gurobi internally applies the tighter bound. For the memory capacity constraint specifically (line 733), replace the scalar `req_size` with a linear expression `quicksum(tile_size_k * s_k for k in tile_options)` and linearize the product of this expression with the binary allocation variable using McCormick bounds — not a single global M.

**Warning signs:**
- LP relaxation bound is far from the best integer solution found (large MIP gap after presolve)
- Gurobi log shows "Model contains large matrix coefficient range" warning
- Coefficient range in `model.printStats()` exceeds six orders of magnitude between smallest and largest coefficient
- Solve time jumps 10x or more when adding even a small tile option list (3-4 options)

**Phase to address:**
Phase where tile selection binary variables are introduced into the CO — before any other constraints reference tile-dependent quantities. Establish the per-option M bound pattern before connecting to memory capacity, SSIS loop count, or latency constraints.

---

### Pitfall 2: `_ensure_same_ssis_for_all_transfers` Breaks With Variable Tile Sizes

**What goes wrong:**
Lines 234-246 in `transfer_and_tensor_allocation.py` enforce that all transfer nodes share the same total SSIS size (product of temporal loop sizes). This precondition is asserted at construction time using fixed SSIS objects built from the pre-tiled workload. When tile size becomes a CO decision variable, the SSIS loop trip counts change with the tile — a tile of 16 produces different `get_temporal_sizes()` values than a tile of 128. If SSIS objects are still pre-built for a single fixed tile and passed in, the CO is optimizing allocation but the SSIS structure is wrong for the tile it might select. If SSIS objects are pre-built for each candidate tile, the `_ensure_same_ssis_for_all_transfers` assertion will fire for mixed-tile scenarios unless the validation logic is updated.

**Why it happens:**
The SSIS is constructed upstream (in `TilingGenerationStage`) and passed as a completed data structure into the CO. The CO treats it as read-only. When the CO is extended to select tile sizes, there is a mismatch: the CO now makes decisions that should affect the SSIS structure, but the SSIS is fixed at construction time. Developers often add tile selection to the CO variables without updating the upstream data flow.

**How to avoid:**
Two valid approaches:
1. Pre-compute one SSIS per candidate tile size and index them by tile selection. The CO holds `ssis_by_tile: dict[tile_size, dict[Node, SSIS]]`. When a tile-size selection variable `s_k` is active, the corresponding SSIS is used. The `_ensure_same_ssis_for_all_transfers` validation must be run per candidate tile, not across all candidates simultaneously.
2. Make SSIS loop sizes themselves CO variables (INTEGER vars for each temporal dimension's trip count), constrained to valid values from the tile size selection. This requires restructuring all `reuse_levels`, `tiles_needed_levels`, and `bds_needed_levels` dicts to be parameterized by both `(tensor, stop, tile_option)`.

Approach 1 is lower risk because it preserves the existing constraint structure and adds a layer of tile-indexed dispatch on top.

**Warning signs:**
- `ValueError: Transfer X has different SSIS total size` at construction time when passing multi-tile options
- z_stop variable counts differ across SSIS options (len mismatch in `range(-1, len(sizes))` loops)
- Latency or reuse factor expressions silently use wrong trip counts (no assertion failure, wrong objective value)

**Phase to address:**
Must be resolved before any CO variable for tile selection is introduced. Update the SSIS data flow contract first; validate with a single fixed tile producing the same results as baseline, then extend to multiple tile options.

---

### Pitfall 3: Binary Variable Count Explosion From Cross-Products With z_stop

**What goes wrong:**
The existing CO already has `_add_binary_product(u, z_stop[(t, stop)])` called in three constraint groups: memory capacity (line 734), object fifo depth (line 760), and buffer descriptor depth (line 788). Each call adds 1 binary auxiliary variable and 3 constraints. With N tensors, C candidate cores, L reuse stop levels, and K tile options, adding tile selection introduces a cross-product. Naively, `u x z_stop x s_k` requires a 3-way binary product, yielding `N * C * L * K` auxiliary binaries. For the BIG BOY config (4x8 array, SwiGLU 5 nodes, multiple tensors, 4+ candidate stops, 4+ tile options), this can reach thousands of auxiliary binaries in just the memory constraint group alone.

**Why it happens:**
It is natural to add tile selection as another binary that gates each constraint, treating it symmetrically with `u` and `z_stop`. But the existing `_add_binary_product` only handles pairwise products; a 3-way binary product `a * b * c` requires two auxiliary variables and five constraints (`w1 = a AND b`, `w2 = w1 AND c`), doubling the blowup.

**How to avoid:**
Restructure the tile selection to avoid the three-way product. The key insight: tile selection `s_k` determines fixed scalar coefficients (like `req_size`, `tiles_needed`, `bds_needed`) — it is a parameter selector, not a tensor placement or reuse level indicator. Model the tile-dependent coefficient as a piecewise-constant linear expression: create an INTEGER variable `tile_coeff_var[t, stop]` defined by `tile_coeff_var == quicksum(coeff_for_tile_k[t, stop] * s_k for k)`. Then the memory constraint becomes `tile_coeff_var[t, stop] * uz` where `uz` is the existing pairwise binary product. This is a binary-times-integer product, which linearizes cleanly with McCormick bounds using the tight interval `[0, max_coeff_over_tiles]`.

**Warning signs:**
- Model construction time grows super-linearly with number of tile options
- `model.NumVars` and `model.NumConstrs` are much larger than `K * (existing_count)` — suggests cubic or worse growth
- Gurobi presolve takes longer than the solve itself
- Adding a 5th tile option takes twice as long to construct as adding a 4th

**Phase to address:**
Before implementing any constraint that combines tile selection with existing reuse/placement binaries. Design the `tile_coeff_var` abstraction before the first constraint is written.

---

### Pitfall 4: Tensor Size Used as Constant in Transfer Latency Computation

**What goes wrong:**
`_transfer_latency_for_path` at line 226 calls `tensor.size_bits()` as a scalar constant and stores the result as a float in `latency_constant`. This float is then used in `_add_binary_times_const_over_linexpr` which divides it by a linear expression (reuse factor). When tile size is variable, `tensor.size_bits()` is no longer a constant — it depends on the selected tile. The transfer latency becomes a nonlinear expression (tile-dependent numerator divided by reuse-factor denominator). If this is not handled explicitly, either: (a) the latency is computed at construction time from a single fixed tile, silently using the wrong value for all other tile options; or (b) the developer tries to introduce a tile-dependent numerator, creating a ratio of two decision-dependent quantities — which is non-convex and requires `addGenConstrNL`.

**Why it happens:**
The latency computation was explicitly designed as a constant-times-binary-over-linexpr pattern to avoid nonlinearity. When tile sizes become variable, developers may miss that the "constant" numerator has become a decision-dependent expression.

**How to avoid:**
For each path choice and tile option k, the latency is a known constant `latency_k = ceil(tensor_size_k / min_bw)`. Model this as a tile-indexed lookup: `transfer_latency_var[tr, choice] = quicksum(latency_k * s_k for k) / reuse_factor`. The numerator becomes a linear expression gated by tile selection binaries, and the denominator is the existing linear reuse factor. This is again a ratio of a linear expression to a linear expression — non-convex in general. The safe formulation: pre-compute `latency_k / reuse_level_l` for all combinations of tile `k` and reuse stop level `l`, then use these as scalar coefficients in a joint `s_k * z_stop_l` product. This avoids the nonlinear division entirely.

**Warning signs:**
- Transfer latency variables have the same value regardless of which tile is selected in the solution
- `_transfer_latency_cache` is populated at construction time from a single tensor size (check: latency values don't vary between tile options in solution output)
- `addGenConstrNL` appearing in the model (indicates nonlinear division was introduced)
- Gurobi model type changes from MIP to MIQCP or MINLP

**Phase to address:**
When implementing variable transfer sizes in CO. Must redesign `_active_transfer_latency` and `_add_binary_times_const_over_linexpr` to accept tile-selection-indexed constants rather than a single scalar.

---

### Pitfall 5: Numerical Instability From Mixed-Magnitude Coefficients (Memory vs. Latency)

**What goes wrong:**
Memory capacity values are in bits (e.g., AIE core local memory is 64KB = 524,288 bits), while latency values are in cycles (tens to hundreds for the BIG BOY config). When tile sizes vary, the largest memory coefficient can be `max_tile_size * size_factor * tensor_bits`, which could reach 10^7 or higher for large tensors. Gurobi's recommended coefficient range is `[10^-3, 10^6]`. Coefficients outside this range cause the LP relaxation to become numerically ill-conditioned: the solver may accept solutions that violate constraints by amounts that appear small in relative terms but are large in absolute terms (e.g., a memory constraint violated by 100 bits but reported as feasible because `100 / 524288 < FeasibilityTol`).

**Why it happens:**
Memory and latency constraints share the same model, and developers often normalize one but not the other. When tile sizes grow, memory coefficients grow proportionally while latency coefficients grow more slowly (latency scales with tile only through reuse factor). The ratio between the largest and smallest coefficient grows with tile size range.

**How to avoid:**
Normalize memory constraints to KB or 32-bit words rather than bits. Check coefficient range with `model.printStats()` after model construction and before the first solve. Target: all coefficients in `[1e-3, 1e6]`. If memory constraints use bits and latency uses cycles, and memory capacity is 524,288 bits, convert to kilobits (512) to bring into the same order of magnitude as latency. Set `model.setParam('NumericFocus', 1)` as a precaution — but treat this as a diagnostic aid, not a fix. The underlying fix is normalization.

**Warning signs:**
- Gurobi log prints: `"Warning: Model contains large matrix coefficient range"` or `"Numeric trouble encountered"`
- `model.printStats()` shows coefficient range spanning more than 6 orders of magnitude
- Solutions reported as optimal violate memory capacity constraints when verified manually post-solve
- Different tile options produce the same allocation despite clearly different memory requirements

**Phase to address:**
During the memory capacity constraint implementation for variable tile sizes. Establish a unit normalization convention (e.g., bits, words, KB) across all constraints at the start of this phase.

---

### Pitfall 6: z_stop Reuse Level Semantics Change With Different Tile Sizes

**What goes wrong:**
The `z_stop[(t, stop)]` variables select a reuse level from the SSIS's temporal iteration variables. The index `stop` maps to an `IterationVariable` with a specific trip count (`iv.size`). When tile size changes, the same `stop` index can correspond to a different loop dimension's trip count — the SSIS structure changes shape. For example, stop level 2 at tile=16 might correspond to a trip count of 8, while stop level 2 at tile=64 corresponds to a trip count of 2. The `reuse_levels[(t, stop)]` dict stores `(fires, size_factor)` tuples computed at construction time. If these are computed for one tile and reused for another, `fires` and `size_factor` are silently wrong.

**Why it happens:**
`reuse_levels`, `tiles_needed_levels`, and `bds_needed_levels` are populated in `_init_transfer_fire_helpers` at construction time from the SSIS objects passed in. Developers extending this to variable tile sizes may update the SSIS objects but forget to regenerate these helper dicts, or may generate them once from a "representative" tile.

**How to avoid:**
Make `reuse_levels`, `tiles_needed_levels`, and `bds_needed_levels` tile-indexed: `reuse_levels[(t, stop, tile_option_k)]`. Regenerate from each candidate tile's SSIS during initialization. The z_stop constraints that use these dicts (lines 603, 623, 732, 757, 787) then become parameterized by which tile is selected.

This directly connects to Pitfall 3 (binary product explosion). Use the `tile_coeff_var` pattern: for each `(t, stop)` pair, create an INTEGER variable that is constrained to equal the tile-selected scalar from `reuse_levels`.

**Warning signs:**
- Reuse factor values in the solution are inconsistent with the tile size that was selected
- `_eval_and_print_linexpr` output shows `reuse_factor` values that don't match the product of relevant SSIS loop sizes for the chosen tile
- `z_stop` validation at line 1213 (`assert stop >= 0`) fails post-solve for some tile options

**Phase to address:**
Same phase as Pitfall 2. These are tightly coupled: fixing SSIS data flow automatically surfaces the need to update the helper dicts.

---

### Pitfall 7: Missing Regression Test for Fixed-Tile Baseline After Each Structural Change

**What goes wrong:**
The existing CO is 1717 lines with interdependent constraint groups. Each extension for variable tile sizes touches multiple constraint groups simultaneously (memory capacity, SSIS, reuse variables, latency). Without a fixed-tile regression test that runs after each structural change, it is easy to introduce a bug that affects the fixed-tile case — which produces wrong results silently (the model is still feasible and produces an objective value, just not the previously-validated one).

**Why it happens:**
Developers focus on making the new tile-selection functionality work and do not run the baseline configuration as a regression after each change. MILP bugs are especially insidious: a wrong coefficient or missing constraint changes the optimal solution without causing an exception or infeasibility signal.

**How to avoid:**
Implement a regression test harness before making any changes:
1. Capture baseline: run the existing CO on the BIG BOY config with fixed tiles (16/128/32). Record: objective value, selected reuse levels, memory allocations, z_stop assignment, fires per transfer.
2. After every structural change, run the same config with the tile-selection machinery but only one tile option per dimension (the original fixed size). Assert: objective == baseline, z_stop assignments == baseline, fires == baseline.
3. The single-option constraint is that `s_k` is forced to 1 for the only tile option — this should reduce exactly to the fixed-tile formulation.

Additionally: when adding variable tile sizes, first validate with a 2-option list where one option is the current tile and the other is a clearly inferior tile (e.g., one that violates memory). The solver must select the current tile and the objective must match baseline.

**Warning signs:**
- Not having captured baseline outputs before starting changes
- First test run after changes uses the new multi-option configuration (skipping fixed-tile regression)
- `test_co.py` does not have a test that asserts specific objective values against a known baseline

**Phase to address:**
Before any code changes. Baseline capture and regression test infrastructure is a prerequisite for all subsequent phases.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use global `self.big_m` scalar for all new tile-dependent constraints | No refactor needed | Loose LP relaxation; 10x+ slower solve for multi-option tile lists | Never acceptable once tile sizes vary |
| Pre-compute SSIS for a single "representative" tile and reuse for all tiles | Avoids SSIS indexing refactor | Silently wrong reuse factors, fires, and size factors for non-representative tiles | Never |
| Build separate CO instance per tile option and pick best post-hoc | Avoids all variable tile changes in CO | Exponential solve time; no joint optimization over placement and tile selection | Only acceptable for 2-3 tile options as a validation step before implementing integrated approach |
| Normalize memory to bits because that's what `size_bits()` returns | No unit conversion needed | Coefficient range warnings from Gurobi; possible silent constraint violations | Never once tile sizes are large |
| Add `s_k` tile selection binaries using one-hot without symmetry constraints | Simple encoding | Gurobi cannot exploit ordered structure of tile sizes; branch-and-bound less efficient | Acceptable if tile options are unordered; use SOS1 if ordered by size |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| TilingGenerationStage → CO | Generating one SSIS at fixed tile, passing to CO that now selects tiles | Generate SSIS per candidate tile size upstream; pass `dict[tile_size, SSIS]` to CO |
| `get_tensor_of_transfer_to_single_core` → memory constraint | Calling once at construction time for the current tiled workload | Either compute per-tile-option tensor sizes upfront and index them, or make the workload tile-parameterized |
| `reuse_levels` / `tiles_needed_levels` / `bds_needed_levels` dicts | Computing from single SSIS, reusing across all tile options | Compute per `(tile_option, tensor, stop)` triple; see Pitfall 6 |
| Gurobi `addGenConstrNL` for latency/reuse division | Necessary when numerator and denominator are both decision-dependent | Avoid by enumerating `latency[tile_k] / reuse[stop_l]` as scalar constants and selecting via joint binary (see Pitfall 4) |
| Baseline test harness | Running existing `test_co.py` without capturing expected values | Capture exact objective, z_stop, fires before any changes; assert exact match after each phase |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| 3-way binary product for `(u, z_stop, s_k)` | Model construction time > 10s; `NumVars` grows as `O(N * C * L * K)` | Use `tile_coeff_var` abstraction to factor out tile dimension from binary products | 3+ tile options with 4+ reuse levels and 8+ candidate cores |
| Loose big-M from max-tile-size scaling | MIP gap >50% after presolve; long branch-and-bound; solve time scales super-linearly with K | Per-option M values or indicator constraints (see Pitfall 1) | 2+ tile options where max/min tile ratio > 4 |
| Gurobi presolve removing variables incorrectly | Optimal solution changes non-monotonically as tile options are added | Verify that presolve is not eliminating valid tile selections by checking `model.NumVarsAfterPresolve` | When tight bounds on tile selection variables are missing |
| `_ensure_same_ssis_for_all_transfers` called once at construction | `ValueError` or silently wrong SSIS for multi-tile runs | Validate per tile option in initialization, not once globally | First run with 2+ tile options |

---

## "Looks Done But Isn't" Checklist

- [ ] **Tile selection variables introduced:** Verify that the model with a single forced tile option produces the same objective and solution as the original fixed-tile CO run. If not, the structural extension broke an existing constraint.
- [ ] **Memory constraint with variable tile:** Verify memory usage reported in solution matches `tile_size_selected * size_factor * element_size` for every tensor on every allocated core — not the max tile size.
- [ ] **Reuse levels consistent with selected tile:** After solve, for each transfer node, confirm that `fires = product(ssis_loop_sizes_for_selected_tile)` divided by the reuse factor implied by the `z_stop` selection. A mismatch means the `reuse_levels` dict is using wrong SSIS sizes.
- [ ] **Transfer latency consistent with selected tile:** Verify `latency_var.X` for each transfer equals `ceil(tensor_size_for_selected_tile / min_bw) / reuse_factor`. Mismatch means the latency is computed from a fixed size.
- [ ] **SSIS loop structure consistent across tile options:** For each candidate tile, independently construct the SSIS and verify that `get_applicable_temporal_variables()` returns the expected number of levels. Mismatches cause `range(-1, len(...))` loops to iterate over the wrong set.
- [ ] **Coefficient range acceptable:** After model construction, call `model.printStats()` and confirm the coefficient range is within 6 orders of magnitude. A wider range means normalization is needed.
- [ ] **Baseline regression passes:** The single-tile-option test matches the original fixed-tile result before declaring any phase complete.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Loose big-M discovered after 10+ phases are implemented | HIGH | Audit all constraint groups for M values; replace global M with per-constraint tight bound; re-validate each group individually |
| SSIS mismatch discovered mid-implementation | MEDIUM | Roll back to single-SSIS architecture, capture correct baseline, then re-implement multi-SSIS dispatch from scratch |
| Binary product explosion after constraints written | MEDIUM | Introduce `tile_coeff_var` abstraction; replace all direct `s_k * existing_binary` products with the coeff-var pattern; regression test each constraint group |
| Coefficient range warning ignored until final integration | MEDIUM | Identify the constraint group producing the largest coefficients (likely memory in bits); normalize that group; re-run; iterate |
| Regression baseline not captured before changes | LOW–MEDIUM | Run the latest main-branch code on BIG BOY config; capture outputs; compare to known-good outputs from git history if available |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Loose big-M bounds (Pitfall 1) | Phase: CO variable introduction | After phase: LP relaxation bound is within 20% of best integer solution at termination |
| SSIS mismatch (Pitfall 2) | Phase: SSIS data flow refactor — before CO changes | After phase: single-tile CO with new SSIS dispatch produces identical output to baseline |
| Binary product explosion (Pitfall 3) | Phase: CO variable introduction — design `tile_coeff_var` before first constraint | After phase: `model.NumVars` growth is linear in K, not cubic |
| Variable transfer latency (Pitfall 4) | Phase: variable transfer size in CO | After phase: latency values in solution match `ceil(tile_k_tensor_size / bw) / reuse` |
| Mixed-magnitude coefficients (Pitfall 5) | Phase: memory capacity constraint for variable tile | After phase: `model.printStats()` shows coefficient range ≤ 6 orders of magnitude |
| z_stop / reuse dict desync (Pitfall 6) | Phase: SSIS data flow refactor | After phase: `reuse_levels[(t, stop, k)]` correctly reflects each candidate tile's SSIS |
| No regression baseline (Pitfall 7) | Phase 0 — prerequisite before any changes | Before phase: baseline outputs captured and asserted in `test_co.py` |

---

## Sources

- Gurobi official documentation: [Dealing with Big-M Constraints](https://www.gurobi.com/documentation/9.1/refman/dealing_with_big_m_constra.html) — recommends minimizing M using domain knowledge; warns loose M degrades LP relaxation quality
- Gurobi official documentation: [Tolerances and User-Scaling](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html) — coefficient range recommendation `[10^-3, 10^6]`; big-M guidance to use tightest possible bound
- Gurobi official documentation: [Constraints — Indicator vs SOS vs Big-M](https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/constraints.html) — `PreSOS1BigM`/`PreSOS2BigM` controls, warning that "large values of M can lead to numerical issues"
- Gurobi community: [Why is using Indicator Constraints significantly slower than Big-M?](https://support.gurobi.com/hc/en-us/community/posts/32306615842321) — indicator constraints translated to SOS1 then big-M internally; explicit tight big-M can outperform indicator constraints
- Gurobi community: [Big-M clarification](https://support.gurobi.com/hc/en-us/community/posts/27648660606865) — numerical instability with large M; indicator constraints as alternative
- Direct codebase analysis: `stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py` — `big_m = len(workload.nodes()) + 5` (line 91), `_add_binary_product` pattern (line 1373), `_init_transfer_fire_helpers` (line 248), `_ensure_same_ssis_for_all_transfers` (line 234)
- MILP formulation reference: [Mixed-Integer Linear Programming Formulation Techniques (Vielma, 2015)](https://juan-pablo-vielma.github.io/publications/Mixed-Integer-Linear-Programming-Formulation-Techniques.pdf) — SOS1 encoding, binary product linearization, bounds tightness impact on LP relaxation
- Gurobi [Solver Parameters for Numerical Issues](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/numeric_parameters.html) — `NumericFocus`, `IntFeasTol`, `Aggregate` parameters for managing ill-conditioning

---
*Pitfalls research for: Gurobi MILP variable tile size extension for AIE allocator*
*Researched: 2026-04-02*
