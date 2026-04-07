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


def tensor_size_bits_for_candidate(
    workload: Workload,
    tensor: Tensor,
    base_inter_core_tiling: InterCoreTiling,
    dim: LayerDim,
    candidate_tile: int,
) -> int:
    """Tensor size in bits when `dim` has intra-core tile size `candidate_tile`.

    Replaces the factor for `dim` in inter_core_tiling with
    (workload_dim_size / candidate_tile) to simulate the per-core tensor
    shape under this candidate. All other tiling factors are unchanged.
    """
    workload_dim_size = workload.get_dimension_size(dim)
    new_factor = workload_dim_size // candidate_tile
    tiling_list = list(base_inter_core_tiling)
    replaced = False
    for i, (d, _f) in enumerate(tiling_list):
        if d == dim:
            tiling_list[i] = (d, new_factor)
            replaced = True
            break
    if not replaced:
        tiling_list.append((dim, new_factor))
    updated_tiling: InterCoreTiling = tuple(tiling_list)
    shape = workload.get_tensor_shape_with_tiling(tensor, updated_tiling)
    return tensor.size_bits(shape=shape)


def max_tensor_size_bits(
    workload: Workload,
    tensor: Tensor,
    base_inter_core_tiling: InterCoreTiling,
    candidates_per_dim: dict[LayerDim, list[int]],
) -> int:
    """Maximum tensor size in bits over all joint candidate tile combinations.

    Uses itertools.product to enumerate all combinations across tiled dims.
    Returns the base tensor_size_bits if candidates_per_dim is empty.
    """
    from itertools import product as iproduct

    if not candidates_per_dim:
        return tensor_size_bits(workload, tensor, base_inter_core_tiling)

    dims = list(candidates_per_dim.keys())
    all_candidates = [candidates_per_dim[d] for d in dims]
    max_size = 0
    for combo in iproduct(*all_candidates):
        current_tiling = list(base_inter_core_tiling)
        for d, tile_val in zip(dims, combo):
            wdim_size = workload.get_dimension_size(d)
            new_factor = wdim_size // tile_val
            replaced = False
            for i, (td, _tf) in enumerate(current_tiling):
                if td == d:
                    current_tiling[i] = (d, new_factor)
                    replaced = True
                    break
            if not replaced:
                current_tiling.append((d, new_factor))
        updated_tiling: InterCoreTiling = tuple(current_tiling)
        shape = workload.get_tensor_shape_with_tiling(tensor, updated_tiling)
        size = tensor.size_bits(shape=shape)
        max_size = max(max_size, size)
    return max_size


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
