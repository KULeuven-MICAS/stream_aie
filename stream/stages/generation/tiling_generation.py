import logging
import os
from math import prod

from stream.datatypes import LayerDim
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
from stream.workload.utils import (
    determine_fusion_splits,
)
from stream.workload.workload import ComputationNode, Workload

logger = logging.getLogger(__name__)


class TilingGenerationStage(Stage):
    """
    This stage:
    - Determines the best dimension to fuse the layers on (pre-solve mode), OR
      accepts CO-solved tile sizes (post-solve mode, per D-01/D-07).
    - Substitutes the loop ranges with the smaller tiled ranges.
    - Generates the steady state iteration space for all tensors and computation nodes.

    Post-solve mode is activated when ``selected_tiles`` is present in ctx
    (set by ConstraintOptimizationAllocationStage after the CO solves).
    In post-solve mode, ``determine_fusion_splits`` is NOT called; the pre-computed
    ``fusion_splits`` from ctx is used as-is.

    TODO: Add support for multiple layer stacks. Curently it assumes all layers are fused together.
    """

    REQUIRED_FIELDS = ("workload", "mapping", "output_path")

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.output_path: str = self.ctx.get("output_path")
        self.fusion_splits: dict[LayerDim, int] = {}
        self.tiled_sizes: dict[int, int] = {}
        self.steady_state_iteration_spaces: dict[ComputationNode, SteadyStateIterationSpace] = {}
        self.unique_dims, self.dim_expressions = self.workload.unique_dimensions()
        # Post-solve mode: selected_tiles set by ConstraintOptimizationAllocationStage
        self.selected_tiles: dict[LayerDim, int] | None = self.ctx.get("selected_tiles")

    def run(self):
        if self.selected_tiles is not None:
            # Post-solve mode (per D-01, D-07): tile sizes come from the CO solver.
            # fusion_splits was pre-computed before the CO (by _FusionSplitsStage) and is in ctx.
            self.fusion_splits = self.ctx.get("fusion_splits")
            self.tiled_sizes = self.substitute_loop_sizes_with_selected_tiles()
        else:
            # Pre-solve / legacy mode: derive fusion_splits and tiled_sizes from mapping.
            self.fusion_splits = determine_fusion_splits(self.workload, self.mapping)
            self.tiled_sizes = self.substitute_loop_sizes_with_tiled_sizes()

        self.tiled_workload = self.workload.with_modified_dimension_sizes(self.tiled_sizes)
        self.tiled_mapping = self.mapping.with_updated_workload(self.tiled_workload, self.workload)

        self.tiled_workload.visualize(os.path.join(self.output_path, "tiled_workload.png"))
        self.ctx.set(
            workload=self.tiled_workload,
            mapping=self.tiled_mapping,
            fusion_splits=self.fusion_splits,
        )
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def substitute_loop_sizes_with_selected_tiles(self) -> dict[LayerDim, int]:
        """Build tiled_sizes dict from CO-solved selected tile sizes (post-solve mode).

        In post-solve mode, ``self.selected_tiles[dim]`` is the tile size chosen by the
        CO for each tiled dimension.  Non-tiled dimensions keep their original workload size.
        """
        assert self.selected_tiles is not None, "selected_tiles must be set in post-solve mode"
        unique_dims, _ = self.workload.unique_dimensions()
        result: dict[LayerDim, int] = {}
        for dim, tile_size in self.selected_tiles.items():
            workload_size = self.workload.get_dimension_size(dim)
            assert workload_size % tile_size == 0, (
                f"Workload dimension size {workload_size} is not divisible by "
                f"selected tile size {tile_size} for dim {dim}"
            )
            result[dim] = tile_size
        # Non-tiled dimensions keep their original size
        for dim in set(unique_dims) - set(self.selected_tiles.keys()):
            result[dim] = self.workload.get_dimension_size(dim)
        return result

    def substitute_loop_sizes_with_tiled_sizes(self):
        """
        The returned dict maps from dimension to its new total tiled size across the spatial unrollings.
        As such, it can differ from the defined fusion split factor in the mapping input as that is per core.
        """
        unique_dims, _ = self.workload.unique_dimensions()
        result = {}
        # Size for the new tiled dimensions
        for dim, split_factor in self.fusion_splits.items():
            wanted_tile_size, rem = divmod(self.workload.get_dimension_size(dim), split_factor)
            assert rem == 0, (
                f"Dimension size {self.workload.get_dimension_size(dim)} not divisible by "
                f"desired tile size {split_factor}"
            )
            result[dim] = wanted_tile_size
        # Size for non-tiled dimensions
        for dim in set(unique_dims) - set(self.fusion_splits.keys()):
            size = self.workload.get_dimension_size(dim)
            result[dim] = size
        return result

    def _get_total_spatial_unrolling_for_dim(
        self,
        dim: LayerDim,
        spatial_unrollings: set[tuple[LayerDim, int]],
    ) -> int:
        total_unrolling = prod(su[1] for su in spatial_unrollings if su[0] == dim)
        return total_unrolling
