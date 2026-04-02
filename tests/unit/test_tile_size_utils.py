from unittest.mock import MagicMock, patch

import pytest

from stream.datatypes import LayerDim
from stream.opt.search_space import SearchSpace, TileSizeOption
from stream.opt.tile_size_utils import (
    is_divisible_candidate,
    passes_single_tensor_memory_check,
    tensor_size_bits,
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
