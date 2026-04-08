"""Post-solve analysis and inspection for TransferAndTensorAllocator.

Provides a read-only view into the solved CO model, extracting variable values,
constraint slack, latency breakdowns, and allocation decisions into structured
data for debugging and validation.

Usage:
    from stream.opt.allocation.constraint_optimization.co_analysis import COAnalysis
    analysis = COAnalysis(allocator)  # after allocator.solve()
    analysis.print_summary()
    analysis.print_latency_breakdown()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, Any

import gurobipy as gp

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

logger = logging.getLogger(__name__)

VAR_THRESHOLD = 0.5


@dataclass
class SlotLatencyEntry:
    """Latency breakdown for a single timeslot."""

    slot_index: int
    slot_latency_value: float
    compute_nodes: list[dict[str, Any]] = field(default_factory=list)
    transfers: list[dict[str, Any]] = field(default_factory=list)
    binding_constraint: str = ""


@dataclass
class TileSelection:
    """Selected tile size for a dimension."""

    dim: Any  # LayerDim
    selected_tile: int
    candidates: list[int]
    w_values: dict[int, float]  # k -> w[dim,k].X


@dataclass
class TransferReuse:
    """Reuse decision for a tensor."""

    tensor_name: str
    transfer_name: str
    reuse_factor: int
    stop_level: int
    fire_count: int


class COAnalysis:
    """Post-solve analysis of a TransferAndTensorAllocator.

    Reads solved variable values from the Gurobi model and provides structured
    inspection methods. Does NOT modify the model or allocator.
    """

    def __init__(self, allocator: "TransferAndTensorAllocator") -> None:
        self._tta = allocator
        self._model = allocator.model
        if self._model.Status != gp.GRB.OPTIMAL:
            logger.warning(f"Model status is {self._model.Status}, not OPTIMAL. Values may be unreliable.")

    # ------------------------------------------------------------------
    # Tile selection
    # ------------------------------------------------------------------

    def get_tile_selections(self) -> list[TileSelection]:
        """Extract solved tile size per dimension."""
        selections = []
        if not hasattr(self._tta, 'w') or not self._tta.w:
            return selections

        ss = self._tta.search_space
        if ss is None or ss.is_empty():
            return selections

        for dim in ss.dims():
            options = ss.get(dim)
            w_values = {}
            selected_tile = None
            for k, opt in enumerate(options):
                val = self._tta.w[(dim, k)].X
                w_values[k] = val
                if val > VAR_THRESHOLD:
                    selected_tile = opt.tile
            selections.append(TileSelection(
                dim=dim,
                selected_tile=selected_tile or options[0].tile,
                candidates=[o.tile for o in options],
                w_values=w_values,
            ))
        return selections

    # ------------------------------------------------------------------
    # Slot latency breakdown
    # ------------------------------------------------------------------

    def get_slot_latency_breakdown(self) -> list[SlotLatencyEntry]:
        """Per-slot latency breakdown with compute and transfer contributions."""
        entries = []
        for s, slot_var in sorted(self._tta.slot_latency.items(), key=lambda x: x[0]):
            entry = SlotLatencyEntry(
                slot_index=s,
                slot_latency_value=slot_var.X,
            )

            # Compute nodes in this slot
            for n in self._tta.ssc_nodes:
                if self._tta.slot_of[n] != s:
                    continue
                lat_coeffs = self._tta._ssc_node_lat_coeffs.get(n, [])
                if lat_coeffs:
                    # Variable mode: find active coefficient
                    active_lat = None
                    for lat, jw in lat_coeffs:
                        if jw is None or jw.X > VAR_THRESHOLD:
                            active_lat = lat
                            break
                    entry.compute_nodes.append({
                        "name": n.name,
                        "latency": active_lat or 0,
                        "all_candidates": [(lat, jw.X if jw else 1.0) for lat, jw in lat_coeffs],
                    })
                else:
                    entry.compute_nodes.append({"name": n.name, "latency": 0, "all_candidates": []})

            # Transfer nodes in this slot
            for (tr, choice), y in self._tta.y_path_choice.items():
                if self._tta.slot_of[tr] != s:
                    continue
                if y.X < VAR_THRESHOLD:
                    continue
                entry.transfers.append({
                    "name": tr.name,
                    "path": str(choice),
                    "y_value": y.X,
                })

            # Identify binding constraint
            max_compute = max((cn["latency"] for cn in entry.compute_nodes), default=0)
            entry.binding_constraint = "compute" if max_compute >= entry.slot_latency_value - 1 else "transfer"

            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Reuse / fire counts
    # ------------------------------------------------------------------

    def get_reuse_decisions(self) -> list[TransferReuse]:
        """Extract reuse level, fire count per transfer."""
        results = []
        for tr in self._tta.transfer_nodes:
            t = tr.inputs[0] if tr.inputs else None
            if t is None:
                continue

            # Find active z_stop
            stop = -1
            for s_candidate in range(-1, 10):
                key = (t, s_candidate)
                if key not in self._tta.z_stop:
                    break
                if self._tta.z_stop[key].X > VAR_THRESHOLD:
                    stop = s_candidate

            # Reuse factor
            rf = self._tta.reuse_factors[tr].X if tr in self._tta.reuse_factors else 1

            # Fire count
            fires = self._tta.fires[tr].X if tr in self._tta.fires else 0

            results.append(TransferReuse(
                tensor_name=t.name,
                transfer_name=tr.name,
                reuse_factor=round(rf),
                stop_level=stop,
                fire_count=round(fires),
            ))
        return results

    # ------------------------------------------------------------------
    # Variable inspection
    # ------------------------------------------------------------------

    def get_var(self, name: str) -> float | None:
        """Look up a solved variable by name."""
        try:
            var = self._model.getVarByName(name)
            return var.X if var is not None else None
        except Exception:
            return None

    def get_vars_matching(self, pattern: str) -> dict[str, float]:
        """Return all solved variables whose name contains the pattern."""
        results = {}
        for v in self._model.getVars():
            if pattern in v.VarName:
                try:
                    results[v.VarName] = v.X
                except Exception:
                    pass
        return results

    def get_constraint_slack(self, name_pattern: str) -> dict[str, float]:
        """Return slack for constraints matching the pattern."""
        results = {}
        for c in self._model.getConstrs():
            if name_pattern in c.ConstrName:
                try:
                    results[c.ConstrName] = c.Slack
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def get_model_stats(self) -> dict[str, Any]:
        """Basic model statistics."""
        return {
            "status": self._model.Status,
            "objective": self._model.ObjVal if self._model.Status == gp.GRB.OPTIMAL else None,
            "num_vars": self._model.NumVars,
            "num_binary": self._model.NumBinVars,
            "num_integer": self._model.NumIntVars,
            "num_continuous": self._model.NumVars - self._model.NumBinVars - self._model.NumIntVars,
            "num_constraints": self._model.NumConstrs,
            "num_gen_constraints": self._model.NumGenConstrs,
            "solve_time": self._model.Runtime,
            "mip_gap": self._model.MIPGap if self._model.Status == gp.GRB.OPTIMAL else None,
        }

    # ------------------------------------------------------------------
    # Latency comparison helper
    # ------------------------------------------------------------------

    def compare_latency_with_estimator(self) -> list[dict[str, Any]]:
        """Compare the CO's slot latency with what TileAwareLatencyEstimator produces.

        This is the key diagnostic: if the CO's per-iteration latency differs from
        what the estimator computes with the solved tiling, there's a formulation bug.
        """
        if self._tta.latency_estimator is None:
            return []

        results = []
        tile_selections = {ts.dim: ts.selected_tile for ts in self.get_tile_selections()}

        for n in self._tta.ssc_nodes:
            s = self._tta.slot_of[n]
            co_slot_lat = self._tta.slot_latency[s].X

            # Get the CO's latency for this node
            co_node_lat = 0
            lat_coeffs = self._tta._ssc_node_lat_coeffs.get(n, [])
            for lat, jw in lat_coeffs:
                if jw is None or jw.X > VAR_THRESHOLD:
                    co_node_lat = lat
                    break

            # Compute what the estimator would give with the solved tiling
            try:
                core = next(iter(self._tta.mapping.get(n).resource_allocation[0]))
                # Build tiling from original inter_core_tiling
                orig_n = self._tta._orig_workload.get_node_by_name(n.name)
                orig_tiling = list(self._tta._orig_workload.get_unique_dims_inter_core_tiling(
                    orig_n, self._tta._orig_mapping
                ))

                # Substitute selected tiles
                for i, (dim, factor) in enumerate(orig_tiling):
                    ssw_dim = self._tta._orig_dim_to_ssw(dim)
                    if ssw_dim in tile_selections:
                        wdim_size = self._tta._orig_workload.get_dimension_size(dim)
                        orig_tiling[i] = (dim, wdim_size // tile_selections[ssw_dim])

                ssw_tiling = self._tta._translate_tiling_to_ssw(orig_tiling)
                est = self._tta.latency_estimator.estimate(n, core, tuple(ssw_tiling))
                est_lat = est.latency_total

                results.append({
                    "node": n.name,
                    "slot": s,
                    "co_node_latency": co_node_lat,
                    "estimator_latency": est_lat,
                    "co_slot_latency": co_slot_lat,
                    "match": abs(co_node_lat - est_lat) <= 1,
                    "tiling_used": orig_tiling,
                })
            except Exception as e:
                results.append({
                    "node": n.name,
                    "slot": s,
                    "co_node_latency": co_node_lat,
                    "error": str(e),
                })

        return results

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a concise post-solve summary."""
        stats = self.get_model_stats()
        print(f"\n{'='*60}")
        print(f"CO Analysis Summary")
        print(f"{'='*60}")
        print(f"Status: {stats['status']} | Objective: {stats['objective']}")
        print(f"Vars: {stats['num_vars']} ({stats['num_binary']} bin, {stats['num_integer']} int, {stats['num_continuous']} cont)")
        print(f"Constraints: {stats['num_constraints']} | GenConstrs: {stats['num_gen_constraints']}")
        print(f"Solve time: {stats['solve_time']:.2f}s | MIP gap: {stats['mip_gap']}")

        tiles = self.get_tile_selections()
        if tiles:
            print(f"\nTile Selections:")
            for ts in tiles:
                print(f"  {ts.dim}: {ts.selected_tile} (from {ts.candidates})")

        print(f"\nTotal latency: {self.get_var('total_latency')}")

    def print_latency_breakdown(self) -> None:
        """Print per-slot latency breakdown."""
        print(f"\n{'='*60}")
        print(f"Slot Latency Breakdown")
        print(f"{'='*60}")
        for entry in self.get_slot_latency_breakdown():
            print(f"\nSlot {entry.slot_index}: latency={entry.slot_latency_value:.0f} (binding: {entry.binding_constraint})")
            for cn in entry.compute_nodes:
                print(f"  Compute: {cn['name']} lat={cn['latency']}")
            for tr in entry.transfers:
                print(f"  Transfer: {tr['name']} (y={tr['y_value']:.1f})")

    def print_latency_comparison(self) -> None:
        """Print CO vs estimator latency comparison (diagnostic)."""
        print(f"\n{'='*60}")
        print(f"CO vs Estimator Latency Comparison")
        print(f"{'='*60}")
        for entry in self.compare_latency_with_estimator():
            if "error" in entry:
                print(f"  {entry['node']}: ERROR - {entry['error']}")
            else:
                match = "OK" if entry["match"] else "MISMATCH"
                print(f"  {entry['node']}: CO={entry['co_node_latency']} est={entry['estimator_latency']} [{match}]")
                if not entry["match"]:
                    print(f"    tiling: {entry['tiling_used']}")

    def print_reuse_decisions(self) -> None:
        """Print reuse/fire decisions per transfer."""
        print(f"\n{'='*60}")
        print(f"Reuse Decisions")
        print(f"{'='*60}")
        for rd in self.get_reuse_decisions():
            print(f"  {rd.transfer_name} [{rd.tensor_name}]: rf={rd.reuse_factor} stop={rd.stop_level} fires={rd.fire_count}")
