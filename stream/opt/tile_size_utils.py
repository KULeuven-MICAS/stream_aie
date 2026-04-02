"""Pure utility functions for tile-dependent quantity computation.

These functions compute values on demand from first principles.
No global cache or memoization — call-site helper variables provide
natural memoization at the CO model construction scope.
"""

from __future__ import annotations

from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.workload.tensor import Tensor
from stream.workload.utils import collect_spatial_unrollings
from stream.workload.workload import ComputationNode, Workload


def tensor_size_bits(
    workload: Workload,
    tensor: Tensor,
    inter_core_tiling: InterCoreTiling,
) -> int:
    """Bits occupied by one per-core tensor slice given inter-core tiling."""
    shape = workload.get_tensor_shape_with_tiling(tensor, inter_core_tiling)
    return tensor.size_bits(shape=shape)


def is_divisible_candidate(
    workload: Workload,
    mapping: Mapping,
    dim: LayerDim,
    candidate_tile: int,
) -> bool:
    """Check that candidate_tile * spatial_unrolling divides workload dimension size.

    Mirrors the divisibility check in determine_fusion_splits() (utils.py):
    divmod(workload_size, int(tile_size * spatial_unrolling)).
    """
    _, unique_spatial_unrollings = collect_spatial_unrollings(workload, mapping)
    unrollings_dict = dict(unique_spatial_unrollings)
    spatial_unrolling = unrollings_dict.get(dim, 1)
    workload_size = workload.get_dimension_size(dim)
    _, rem = divmod(workload_size, int(candidate_tile * spatial_unrolling))
    return rem == 0


def passes_single_tensor_memory_check(
    workload: Workload,
    mapping: Mapping,
    node: ComputationNode,
    core: Core,
) -> bool:
    """True if no single per-core tensor slice of this node exceeds core memory capacity.

    Uses post-inter-core-split tensor size (per-core slice). The intra-core
    tile candidate does NOT affect per-core slice size.
    """
    capacity = core.get_memory_capacity()
    inter_core_tiling = workload.get_unique_dims_inter_core_tiling(node, mapping)
    for tensor in node.tensors:
        size = tensor_size_bits(workload, tensor, inter_core_tiling)
        if size > capacity:
            return False
    return True
