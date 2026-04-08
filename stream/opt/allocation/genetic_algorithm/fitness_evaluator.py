from zigzag.datatypes import LayerOperand, MemoryOperand
from zigzag.mapping.data_movement import FourWayDataMoving
from zigzag.utils import pickle_deepcopy

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.cost_model.tile_aware_latency import TileAwareLatencyEstimator
from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import get_too_large_operands
from stream.workload.workload import ComputationNode, Workload


class FitnessEvaluator:
    def __init__(
        self,
        workload: Workload,
        accelerator: Accelerator,
        latency_estimator: TileAwareLatencyEstimator | None,
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.latency_estimator = latency_estimator
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError


class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload: Workload,
        accelerator: Accelerator,
        latency_estimator: TileAwareLatencyEstimator | None,
        layer_groups_flexible,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        latency_attr: str,
    ) -> None:
        super().__init__(workload, accelerator, latency_estimator)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.layer_groups_flexible = layer_groups_flexible
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order
        self.latency_attr = latency_attr

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.operands_to_prefetch,
            self.scheduling_order,
        )
        scme.evaluate()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            return energy, latency
        return energy, latency, scme

    def set_node_core_allocations(self, core_allocations: list[int]):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core
        allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.node_list
                if isinstance(node, ComputationNode) and node.id == layer_id and node.group == group_id
            )
            for node in nodes:
                # Use latency_estimator if available, else fall back to zero-cost placeholder
                if self.latency_estimator is not None:
                    inter_core_tiling = tuple(node.inter_core_tiling) if node.inter_core_tiling else ()
                    lat_est = self.latency_estimator.estimate(node, core, inter_core_tiling)
                    latency = lat_est.latency_total
                    onchip_energy = lat_est.energy_total
                else:
                    latency = 0
                    onchip_energy = 0.0
                offchip_energy = 0.0
                too_large_operands: list[MemoryOperand] = []
                offchip_bandwidth_per_op: dict[MemoryOperand, FourWayDataMoving] = {}
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(latency)
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(offchip_bandwidth_per_op)
