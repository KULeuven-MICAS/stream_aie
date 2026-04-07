from unittest.mock import MagicMock, patch

import pytest

from stream.datatypes import LayerDim
from stream.opt.search_space import SearchSpace, TileSizeOption
from stream.opt.tile_size_utils import (
    is_divisible_candidate,
    max_tensor_size_bits,
    passes_single_tensor_memory_check,
    reuse_coefficients_for_sizes,
    ssis_loop_sizes_for_candidate,
    tensor_size_bits,
    tensor_size_bits_for_candidate,
)

# --- SearchSpace and TileSizeOption tests ---


def _dim(pos: int) -> LayerDim:
    return LayerDim(position=pos, prefix="z")


def test_tile_size_option_is_frozen():
    opt = TileSizeOption(dim=_dim(0), tile=16, workload_size=256)
    with pytest.raises(AttributeError):
        opt.tile = 32  # type: ignore[misc]


def test_tile_size_option_stores_fields():
    d = _dim(1)
    tile = 128
    ws = 2048
    opt = TileSizeOption(dim=d, tile=tile, workload_size=ws)
    assert opt.dim == d
    assert opt.tile == tile
    assert opt.workload_size == ws


def test_search_space_add_and_get():
    ss = SearchSpace()
    d = _dim(0)
    opt = TileSizeOption(dim=d, tile=16, workload_size=256)
    ss.add(d, opt)
    assert ss.get(d) == [opt]


def test_search_space_get_unknown_dim_returns_empty():
    ss = SearchSpace()
    assert ss.get(_dim(99)) == []


def test_search_space_dims():
    ss = SearchSpace()
    d0 = _dim(0)
    d1 = _dim(1)
    ss.add(d0, TileSizeOption(dim=d0, tile=16, workload_size=256))
    ss.add(d1, TileSizeOption(dim=d1, tile=32, workload_size=8192))
    assert set(ss.dims()) == {d0, d1}


def test_search_space_is_empty():
    ss = SearchSpace()
    assert ss.is_empty()
    d = _dim(0)
    ss.add(d, TileSizeOption(dim=d, tile=16, workload_size=256))
    assert not ss.is_empty()


def test_search_space_multiple_options_per_dim():
    ss = SearchSpace()
    d = _dim(0)
    opt1 = TileSizeOption(dim=d, tile=16, workload_size=256)
    opt2 = TileSizeOption(dim=d, tile=32, workload_size=256)
    opt3 = TileSizeOption(dim=d, tile=64, workload_size=256)
    ss.add(d, opt1)
    ss.add(d, opt2)
    ss.add(d, opt3)
    assert ss.get(d) == [opt1, opt2, opt3]


# --- Utility function tests ---


def test_tensor_size_bits_delegates_to_workload_and_tensor():
    workload = MagicMock()
    tensor = MagicMock()
    tiling = ((_dim(0), 4),)
    expected_shape = (64, 128)
    expected_bits = 131072

    workload.get_tensor_shape_with_tiling.return_value = expected_shape
    tensor.size_bits.return_value = expected_bits

    result = tensor_size_bits(workload, tensor, tiling)

    workload.get_tensor_shape_with_tiling.assert_called_once_with(tensor, tiling)
    tensor.size_bits.assert_called_once_with(shape=expected_shape)
    assert result == expected_bits


@patch("stream.opt.tile_size_utils.collect_spatial_unrollings")
def test_is_divisible_candidate_true_with_unrolling(mock_collect):
    workload = MagicMock()
    mapping = MagicMock()
    d = _dim(0)
    # workload_size=256, candidate=16, spatial_unrolling=4 → 256 % (16*4) = 256 % 64 = 0
    mock_collect.return_value = ({}, [(d, 4)])
    workload.get_dimension_size.return_value = 256

    assert is_divisible_candidate(workload, mapping, d, 16) is True


@patch("stream.opt.tile_size_utils.collect_spatial_unrollings")
def test_is_divisible_candidate_false_with_unrolling(mock_collect):
    workload = MagicMock()
    mapping = MagicMock()
    d = _dim(0)
    # workload_size=256, candidate=33, spatial_unrolling=1 → 256 % 33 != 0
    mock_collect.return_value = ({}, [(d, 1)])
    workload.get_dimension_size.return_value = 256

    assert is_divisible_candidate(workload, mapping, d, 33) is False


@patch("stream.opt.tile_size_utils.collect_spatial_unrollings")
def test_is_divisible_candidate_no_unrolling_for_dim(mock_collect):
    workload = MagicMock()
    mapping = MagicMock()
    d = _dim(0)
    other_dim = _dim(1)
    # No unrolling for d → defaults to 1; 256 % 16 = 0
    mock_collect.return_value = ({}, [(other_dim, 4)])
    workload.get_dimension_size.return_value = 256

    assert is_divisible_candidate(workload, mapping, d, 16) is True


def test_passes_single_tensor_memory_check_all_fit():
    workload = MagicMock()
    mapping = MagicMock()
    node = MagicMock()
    core = MagicMock()

    core.get_memory_capacity.return_value = 1_000_000
    tiling = ((_dim(0), 4),)
    workload.get_unique_dims_inter_core_tiling.return_value = tiling

    t1 = MagicMock()
    t2 = MagicMock()
    node.tensors = [t1, t2]

    workload.get_tensor_shape_with_tiling.return_value = (64, 128)
    t1.size_bits.return_value = 500_000
    t2.size_bits.return_value = 400_000

    assert passes_single_tensor_memory_check(workload, mapping, node, core) is True


def test_passes_single_tensor_memory_check_one_exceeds():
    workload = MagicMock()
    mapping = MagicMock()
    node = MagicMock()
    core = MagicMock()

    capacity = 1_000_000
    core.get_memory_capacity.return_value = capacity
    tiling = ((_dim(0), 4),)
    workload.get_unique_dims_inter_core_tiling.return_value = tiling

    t1 = MagicMock()
    t2 = MagicMock()
    node.tensors = [t1, t2]

    workload.get_tensor_shape_with_tiling.return_value = (64, 128)
    t1.size_bits.return_value = 500_000
    t2.size_bits.return_value = 1_500_000  # exceeds capacity

    assert passes_single_tensor_memory_check(workload, mapping, node, core) is False


# --- tensor_size_bits_for_candidate tests ---


def test_tensor_size_bits_for_candidate_single_dim():
    """candidate_tile replaces the factor for dim in base_inter_core_tiling."""
    d = _dim(0)
    workload = MagicMock()
    tensor = MagicMock()

    # workload_size=256, candidate_tile=16 → new_factor = 256 // 16 = 16
    workload.get_dimension_size.return_value = 256
    expected_shape = (16, 64)
    expected_bits = 16 * 16 * 64  # bitwidth * prod(shape)
    workload.get_tensor_shape_with_tiling.return_value = expected_shape
    tensor.size_bits.return_value = expected_bits

    base_tiling = ((d, 4),)  # original factor 4, will be replaced with 16
    result = tensor_size_bits_for_candidate(workload, tensor, base_tiling, d, 16)

    # Verify new factor was computed: 256 // 16 = 16
    workload.get_dimension_size.assert_called_once_with(d)
    # Verify get_tensor_shape_with_tiling was called with updated tiling
    call_args = workload.get_tensor_shape_with_tiling.call_args
    called_tiling = call_args[0][1]
    # The tiling should have d mapped to factor 16
    tiling_dict = dict(called_tiling)
    assert tiling_dict[d] == 16
    assert result == expected_bits


def test_tensor_size_bits_for_candidate_dim_not_in_base():
    """When dim is NOT in base_inter_core_tiling, the function appends it."""
    d0 = _dim(0)
    d1 = _dim(1)
    workload = MagicMock()
    tensor = MagicMock()

    workload.get_dimension_size.return_value = 128
    workload.get_tensor_shape_with_tiling.return_value = (8, 32)
    tensor.size_bits.return_value = 8 * 32 * 16

    # base_tiling does NOT contain d1
    base_tiling = ((d0, 4),)
    tensor_size_bits_for_candidate(workload, tensor, base_tiling, d1, 16)

    call_args = workload.get_tensor_shape_with_tiling.call_args
    called_tiling = call_args[0][1]
    tiling_dict = dict(called_tiling)
    # d1 should have been appended with factor 128 // 16 = 8
    assert d1 in tiling_dict
    assert tiling_dict[d1] == 8
    # d0 should remain unchanged
    assert tiling_dict[d0] == 4


def test_max_tensor_size_bits_single_dim():
    """Returns the maximum size across candidates [16, 32, 64] for one dim."""
    d = _dim(0)
    workload = MagicMock()
    tensor = MagicMock()

    # Size grows as tile shrinks (more splits → smaller slices), but here we want
    # to verify max is returned regardless of ordering.
    # Tile 16 → factor 16 → large shape → size 3000
    # Tile 32 → factor 8  → medium shape → size 2000
    # Tile 64 → factor 4  → small shape  → size 1000
    workload.get_dimension_size.return_value = 256

    def shape_side_effect(t, tiling):
        tiling_dict = dict(tiling)
        factor = tiling_dict.get(d, 1)
        # factor 16 → size 3000, factor 8 → 2000, factor 4 → 1000
        return (factor * 100,)

    workload.get_tensor_shape_with_tiling.side_effect = shape_side_effect

    def size_bits_side_effect(shape):
        return shape[0] * 10

    tensor.size_bits.side_effect = lambda shape: shape[0] * 10

    candidates_per_dim = {d: [16, 32, 64]}
    result = max_tensor_size_bits(workload, tensor, (), candidates_per_dim)

    # Tile 16 → factor 16 → shape (1600,) → size 16000
    # Tile 32 → factor 8  → shape (800,)  → size 8000
    # Tile 64 → factor 4  → shape (400,)  → size 4000
    assert result == 16000


def test_max_tensor_size_bits_multi_dim():
    """Returns the max over Cartesian product for 2 dims."""
    d0 = _dim(0)
    d1 = _dim(1)
    workload = MagicMock()
    tensor = MagicMock()

    workload.get_dimension_size.return_value = 256

    def shape_side_effect(t, tiling):
        tiling_dict = dict(tiling)
        f0 = tiling_dict.get(d0, 1)
        f1 = tiling_dict.get(d1, 1)
        return (f0, f1)

    workload.get_tensor_shape_with_tiling.side_effect = shape_side_effect
    tensor.size_bits.side_effect = lambda shape: shape[0] * shape[1]

    # d0 candidates: [16, 32] → factors [16, 8]
    # d1 candidates: [16, 64] → factors [16, 4]
    # max product of factors: 16*16 = 256
    candidates_per_dim = {d0: [16, 32], d1: [16, 64]}
    result = max_tensor_size_bits(workload, tensor, (), candidates_per_dim)
    assert result == 256


def test_max_tensor_size_bits_empty_candidates():
    """When candidates_per_dim is empty, returns the base tensor_size_bits."""
    d = _dim(0)
    workload = MagicMock()
    tensor = MagicMock()

    base_tiling = ((d, 4),)
    workload.get_tensor_shape_with_tiling.return_value = (64, 128)
    tensor.size_bits.return_value = 8192

    result = max_tensor_size_bits(workload, tensor, base_tiling, {})

    # Should fall back to tensor_size_bits(workload, tensor, base_tiling)
    workload.get_tensor_shape_with_tiling.assert_called_once_with(tensor, base_tiling)
    tensor.size_bits.assert_called_once_with(shape=(64, 128))
    assert result == 8192


# ---------------------------------------------------------------------------
# Tests for ssis_loop_sizes_for_candidate (D-08/D-09)
# ---------------------------------------------------------------------------


@patch("stream.opt.tile_size_utils.collect_spatial_unrollings")
def test_ssis_loop_sizes_for_candidate_basic(mock_collect):
    """Basic: K=candidate, T=workload_size/(S*K) with S=1."""
    workload = MagicMock()
    mapping = MagicMock()
    d = _dim(0)

    # S=1 (no spatial unrolling for dim), workload_size=256, candidate=16
    # K=16, T=256/(1*16)=16
    mock_collect.return_value = ({}, [])
    workload.get_dimension_size.return_value = 256

    K, T = ssis_loop_sizes_for_candidate(d, 16, workload, mapping)
    assert K == 16
    assert T == 16


@patch("stream.opt.tile_size_utils.collect_spatial_unrollings")
def test_ssis_loop_sizes_for_candidate_with_spatial_unrolling(mock_collect):
    """S > 1: T = workload_size / (S * K)."""
    workload = MagicMock()
    mapping = MagicMock()
    d = _dim(1)

    # S=4, workload_size=2048, candidate=128 → K=128, T=2048/(4*128)=4
    mock_collect.return_value = ({}, [(d, 4)])
    workload.get_dimension_size.return_value = 2048

    K, T = ssis_loop_sizes_for_candidate(d, 128, workload, mapping)
    assert K == 128
    assert T == 4


@patch("stream.opt.tile_size_utils.collect_spatial_unrollings")
def test_ssis_loop_sizes_for_candidate_indivisible_raises(mock_collect):
    """Indivisible workload_size raises AssertionError."""
    workload = MagicMock()
    mapping = MagicMock()
    d = _dim(0)

    mock_collect.return_value = ({}, [])
    workload.get_dimension_size.return_value = 256

    with pytest.raises(AssertionError):
        ssis_loop_sizes_for_candidate(d, 33, workload, mapping)


# ---------------------------------------------------------------------------
# Tests for reuse_coefficients_for_sizes (D-08/D-09)
# ---------------------------------------------------------------------------


def test_reuse_coefficients_for_sizes_all_relevant():
    """All loops relevant: fires decreases, size_factor grows, bds_needed resets to 1."""
    sizes = [4, 8]
    relevancies = [True, True]
    fires, sf, tn, bn = reuse_coefficients_for_sizes(sizes, relevancies)

    # stop=-1: fires=32, sf=1, tn=1, bn=1
    assert fires[-1] == 32
    assert sf[-1] == 1
    assert tn[-1] == 1
    assert bn[-1] == 1

    # stop=0 (after first loop size=4, relevant=True):
    # fires=32//4=8, sf=4, tn=4, bds_needed=1 (relevant resets)
    assert fires[0] == 8
    assert sf[0] == 4
    assert tn[0] == 4
    assert bn[0] == 1

    # stop=1 (after second loop size=8, relevant=True):
    # fires=8//8=1, sf=4*8=32, tn=32, bds_needed=1
    assert fires[1] == 1
    assert sf[1] == 32
    assert tn[1] == 32
    assert bn[1] == 1


def test_reuse_coefficients_for_sizes_mixed_relevancies():
    """Mixed: irrelevant loop accumulates bds_needed, resets on relevant."""
    sizes = [4, 8, 2]
    relevancies = [False, True, False]
    fires, sf, tn, bn = reuse_coefficients_for_sizes(sizes, relevancies)

    # stop=-1: fires=64, sf=1, tn=1, bn=1
    assert fires[-1] == 64

    # stop=0 (size=4, irrelevant): fires=16, sf=1, tn=1, bds=4
    assert fires[0] == 16
    assert sf[0] == 1
    assert tn[0] == 1
    assert bn[0] == 4

    # stop=1 (size=8, relevant): fires=2, sf=8, tn=8, bds=1 (reset)
    assert fires[1] == 2
    assert sf[1] == 8
    assert tn[1] == 8
    assert bn[1] == 1

    # stop=2 (size=2, irrelevant): fires=1, sf=8, tn=8, bds=2
    assert fires[2] == 1
    assert sf[2] == 8
    assert tn[2] == 8
    assert bn[2] == 2


def test_reuse_coefficients_for_sizes_single_loop():
    """Single loop: basic sanity check."""
    fires, sf, tn, bn = reuse_coefficients_for_sizes([16], [True])
    assert fires[-1] == 16
    assert fires[0] == 1
    assert sf[0] == 16
    assert bn[0] == 1
