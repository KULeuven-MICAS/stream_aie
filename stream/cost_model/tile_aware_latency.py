"""Tile-aware latency estimator for AIE computation nodes.

Provides TileAwareLatencyEstimator as the replacement for CoreCostLUT-based
latency estimation in the constraint optimizer. Computes latency on demand
from node dimensions, kernel utilization, and explicit inter_core_tiling.

Decision references: D-01, D-06, D-07, D-08 (Phase 6 context).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor, prod

from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.stages.estimation.aie_cost_estimator import AIECostEstimator
from stream.workload.workload import ComputationNode, Workload


@dataclass
class LatencyEstimate:
    """Lightweight result of a tile-aware latency computation.

    Fields:
        latency_total: Actual cycles accounting for kernel utilization.
        ideal_cycle:   Ideal cycles at 100% utilization (ceil(MACs / ideal_ops)).
        energy_total:  Energy estimate (placeholder, 0.0 for now).
    """

    latency_total: int
    ideal_cycle: int
    energy_total: float = 0.0


class TileAwareLatencyEstimator:
    """Compute latency for a ComputationNode given explicit inter_core_tiling.

    Unlike AIECostEstimator, this class accepts the inter_core_tiling as an
    explicit argument instead of reading it from the mapping. This allows the
    constraint optimizer to call it once per tile-size candidate during model
    construction.

    Formula (D-01):
        macs = prod(dim_sizes) // tiling_factor
        ideal_ops = AIECostEstimator.ops_per_cycle(node, core)
        ideal_cycles = ceil(macs / ideal_ops)
        ops_per_cycle = floor(ideal_ops * utilization / 100.0)
        latency_total = ceil(macs / ops_per_cycle)
    """

    def __init__(self, workload: Workload, mapping: Mapping) -> None:
        self.workload = workload
        self.mapping = mapping
        # Internal AIECostEstimator for ops_per_cycle dispatch
        self._aie = AIECostEstimator(workload=workload, mapping=mapping)

    def estimate(
        self,
        node: ComputationNode,
        core: Core,
        inter_core_tiling: tuple[tuple, ...],
    ) -> LatencyEstimate:
        """Compute latency estimate for node on core with the given tiling.

        Args:
            node: The computation node to estimate.
            core: The target AIE core (determines ops_per_cycle).
            inter_core_tiling: Sequence of (LayerDim, factor) pairs. The
                product of all factors is the inter-core tiling factor used
                to reduce the MAC count. Pass an empty tuple for no tiling.

        Returns:
            LatencyEstimate with latency_total, ideal_cycle, and energy_total.
        """
        # Compute total MACs from workload dimension sizes
        dims = self.workload.get_dims(node)
        dim_sizes = [self.workload.get_dimension_size(d) for d in dims]
        tiling_factor = prod(f for _, f in inter_core_tiling) if inter_core_tiling else 1
        macs = prod(dim_sizes) // tiling_factor

        # Get kernel utilization from mapping
        kernel = self.mapping.get(node).kernel
        assert kernel is not None, "Kernel must be defined in mapping for TileAwareLatencyEstimator."
        utilization = kernel.utilization

        # Compute cycle counts using the AIE formula (D-01)
        ideal_ops = self._aie.ops_per_cycle(node, core)
        ideal_cycles = ceil(macs / ideal_ops)
        ops_per_cycle = floor(ideal_ops * utilization / 100.0)
        cycles = ceil(macs / ops_per_cycle)

        return LatencyEstimate(
            latency_total=cycles,
            ideal_cycle=ideal_cycles,
            energy_total=0.0,
        )
