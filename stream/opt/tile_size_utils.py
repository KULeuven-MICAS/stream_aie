"""Pure utility functions for tile-dependent quantity computation.

These functions compute values on demand from first principles.
No global cache or memoization — call-site helper variables provide
natural memoization at the CO model construction scope.
"""

from __future__ import annotations

import math

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


def ssis_loop_sizes_for_candidate(
    dim: LayerDim,
    candidate_tile: int,
    workload: Workload,
    mapping: Mapping,
) -> tuple[int, int]:
    """Return (K, T) for a candidate tile on dim.

    Per D-08/D-09: delegates to the SSIS K × S × T = workload_size decomposition.
    K = candidate_tile (kernel/intra-core tile size).
    T = workload_size / (S × K) where S is the spatial unrolling factor.

    This is a convenience wrapper that does not require a pre-built SSIS object.
    For use at CO model construction time when the SSIS is not yet available.

    Parameters
    ----------
    dim : LayerDim
        The dimension being tiled.
    candidate_tile : int
        The candidate intra-core tile size for this dimension.
    workload : Workload
    mapping : Mapping

    Returns
    -------
    (K, T) where K = candidate_tile and T = workload_size / (S * K).
    """
    _, unique_spatial_unrollings = collect_spatial_unrollings(workload, mapping)
    unrollings_dict = dict(unique_spatial_unrollings)
    S = unrollings_dict.get(dim, 1)
    workload_size = workload.get_dimension_size(dim)
    K = candidate_tile
    T, rem = divmod(workload_size, S * K)
    assert rem == 0, (
        f"workload_size {workload_size} not divisible by S*K = {S}*{K} = {S * K}"
    )
    return K, T


def reuse_coefficients_for_sizes(
    sizes: list[int],
    relevancies: list[bool],
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Compute per-stop-level reuse coefficients from temporal loop sizes and relevancies.

    Returns four dicts keyed by stop level (−1 = outermost / no stop, 0..N-1 = innermost
    to outermost stop positions):

    - fires_per_stop:       product of all loop sizes above this stop level
    - size_factor_per_stop: product of relevant loop sizes above this stop level
    - tiles_needed_per_stop: same as size_factor (distinct tile slots needed)
    - bds_needed_per_stop:  product of irrelevant loop sizes in the innermost
                            contiguous irrelevant block above this stop level

    Parameters
    ----------
    sizes : list[int]
        Temporal loop trip-counts, innermost first.
    relevancies : list[bool]
        Corresponding relevancy flags (True = relevant / varying).
    """
    fires_out: dict[int, int] = {}
    size_factor_out: dict[int, int] = {}
    tiles_needed_out: dict[int, int] = {}
    bds_needed_out: dict[int, int] = {}

    fires = math.prod(sizes)
    size_factor = 1
    tiles_needed = 1
    bds_needed = 1

    fires_out[-1] = fires
    size_factor_out[-1] = size_factor
    tiles_needed_out[-1] = tiles_needed
    bds_needed_out[-1] = bds_needed

    for i, (Nl, relevancy) in enumerate(zip(sizes, relevancies, strict=True)):
        size_factor *= Nl if relevancy else 1
        tiles_needed *= Nl if relevancy else 1
        fires //= Nl
        if relevancy:
            bds_needed = 1
        else:
            bds_needed *= Nl
        fires_out[i] = fires
        size_factor_out[i] = size_factor
        tiles_needed_out[i] = tiles_needed
        bds_needed_out[i] = bds_needed

    return fires_out, size_factor_out, tiles_needed_out, bds_needed_out
