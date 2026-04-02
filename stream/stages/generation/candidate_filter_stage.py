import logging

from stream.datatypes import LayerDim
from stream.opt.search_space import SearchSpace, TileSizeOption
from stream.opt.tile_size_utils import is_divisible_candidate, passes_single_tensor_memory_check
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import ComputationNode

logger = logging.getLogger(__name__)


class CandidateFilterStage(Stage):
    """Filter candidate tile sizes for divisibility and single-tensor memory feasibility.

    Reads tile_options_raw from ctx, resolves dim names to unique LayerDim groups,
    filters candidates, builds a SearchSpace, and sets it into ctx.
    """

    REQUIRED_FIELDS = ("workload", "mapping", "accelerator", "tile_options_raw")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.mapping = self.ctx.require_value("mapping", self.__class__.__name__)
        self.accelerator = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.tile_options_raw: dict[str, list[int]] = self.ctx.require_value(
            "tile_options_raw", self.__class__.__name__
        )

    def run(self):
        search_space = self._build_search_space()
        self.ctx.set(search_space=search_space)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _build_search_space(self) -> SearchSpace:
        search_space = SearchSpace()
        seen_unique_dims: set[LayerDim] = set()

        for raw_dim_str, candidates in self.tile_options_raw.items():
            # Resolve "NodeName.D0" -> unique LayerDim (same logic as MappingFactory)
            node_name, dim_name = raw_dim_str.split(".")
            dim_idx = int(dim_name[1:])  # "D1" -> 1
            node = self.workload.get_node_by_name(node_name)
            assert isinstance(node, ComputationNode), f"Node {node_name} is not a ComputationNode."
            unique_dim = self.workload.get_dims(node)[dim_idx]

            # Skip if already processed this unique dim (multiple LayerDims may map to same one)
            if unique_dim in seen_unique_dims:
                continue
            seen_unique_dims.add(unique_dim)

            workload_size = self.workload.get_dimension_size(unique_dim)

            for candidate in candidates:
                if not is_divisible_candidate(self.workload, self.mapping, unique_dim, candidate):
                    logger.info(
                        "Candidate tile %d for dim %s rejected: not divisible (workload_size=%d)",
                        candidate,
                        unique_dim,
                        workload_size,
                    )
                    continue

                option = TileSizeOption(dim=unique_dim, tile=candidate, workload_size=workload_size)
                search_space.add(unique_dim, option)
                logger.info("Candidate tile %d for dim %s accepted", candidate, unique_dim)

        # Single-tensor memory check per compute node
        for node in self.workload.get_computation_nodes():
            for core in self.accelerator.core_list:
                if not passes_single_tensor_memory_check(self.workload, self.mapping, node, core):
                    logger.warning(
                        "Node %s has a tensor exceeding memory capacity of core %s",
                        node.name,
                        core,
                    )
                    break

        # Assert non-empty SearchSpace
        if search_space.is_empty():
            raise ValueError(
                "SearchSpace is empty — no valid tile candidates for any dimension. "
                "Ensure tile_options in the mapping YAML contains at least one valid divisor "
                "per workload dimension."
            )

        for dim in search_space.dims():
            if not search_space.get(dim):
                raise ValueError(f"All candidate tiles filtered out for dimension {dim}. No valid tile sizes remain.")

        return search_space
