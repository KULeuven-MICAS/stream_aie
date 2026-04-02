from unittest.mock import MagicMock, patch

import pytest

from stream.datatypes import LayerDim
from stream.stages.generation.candidate_filter_stage import CandidateFilterStage


def _dim(pos: int) -> LayerDim:
    return LayerDim(position=pos, prefix="z")


def _make_stage(tile_options_raw, dims_map=None, dim_sizes=None):
    """Create a CandidateFilterStage with mocked context.

    Args:
        tile_options_raw: dict[str, list[int]] mapping "NodeName.Dx" to candidate lists
        dims_map: dict mapping (node_name, dim_idx) to LayerDim (unique dims)
        dim_sizes: dict mapping LayerDim to workload dimension size
    """
    if dims_map is None:
        dims_map = {}
    if dim_sizes is None:
        dim_sizes = {}

    workload = MagicMock()
    mapping = MagicMock()
    accelerator = MagicMock()
    accelerator.core_list = []

    # Set up get_node_by_name and get_dims
    def get_node_by_name(name):
        node = MagicMock()
        node.name = name
        node.__class__.__name__ = "ComputationNode"
        # Make isinstance check pass
        return node

    workload.get_node_by_name = get_node_by_name
    workload.get_computation_nodes.return_value = []

    def get_dims(node):
        # Return a list where index dim_idx -> LayerDim
        max_idx = max((idx for (nn, idx) in dims_map if nn == node.name), default=0)
        result = [None] * (max_idx + 1)
        for (nn, idx), ld in dims_map.items():
            if nn == node.name:
                result[idx] = ld
        return result

    workload.get_dims = get_dims

    def get_dimension_size(dim):
        return dim_sizes.get(dim, 256)

    workload.get_dimension_size = get_dimension_size

    # Build a mock ctx
    ctx = MagicMock()
    ctx.require_value = MagicMock(
        side_effect=lambda key, _: {
            "workload": workload,
            "mapping": mapping,
            "accelerator": accelerator,
            "tile_options_raw": tile_options_raw,
        }[key]
    )
    ctx.require_fields = MagicMock()

    # Create stage with a mock next-stage callable
    next_stage = MagicMock()
    next_stage.return_value = MagicMock()
    next_stage.return_value.run.return_value = iter([ctx])

    stage = CandidateFilterStage([next_stage], ctx)
    return stage


@patch("stream.stages.generation.candidate_filter_stage.is_divisible_candidate")
def test_filter_keeps_divisible_candidates(mock_divisible):
    d0 = _dim(0)
    mock_divisible.return_value = True

    stage = _make_stage(
        tile_options_raw={"Gemm.D0": [16, 32]},
        dims_map={("Gemm", 0): d0},
        dim_sizes={d0: 256},
    )

    ss = stage._build_search_space()
    options = ss.get(d0)
    tiles = [o.tile for o in options]
    assert tiles == [16, 32]


@patch("stream.stages.generation.candidate_filter_stage.is_divisible_candidate")
def test_filter_removes_non_divisible(mock_divisible):
    d0 = _dim(0)
    # 16 is divisible, bad_tile is not
    bad_tile = 33
    mock_divisible.side_effect = lambda _w, _m, _d, tile: tile != bad_tile

    stage = _make_stage(
        tile_options_raw={"Gemm.D0": [16, bad_tile, 64]},
        dims_map={("Gemm", 0): d0},
        dim_sizes={d0: 256},
    )

    ss = stage._build_search_space()
    tiles = [o.tile for o in ss.get(d0)]
    assert tiles == [16, 64]
    assert bad_tile not in tiles


@patch("stream.stages.generation.candidate_filter_stage.is_divisible_candidate")
def test_empty_search_space_raises(mock_divisible):
    d0 = _dim(0)
    mock_divisible.return_value = False  # All candidates rejected

    stage = _make_stage(
        tile_options_raw={"Gemm.D0": [33, 37]},
        dims_map={("Gemm", 0): d0},
        dim_sizes={d0: 256},
    )

    with pytest.raises(ValueError, match="SearchSpace is empty"):
        stage._build_search_space()


@patch("stream.stages.generation.candidate_filter_stage.is_divisible_candidate")
def test_duplicate_unique_dim_skipped(mock_divisible):
    d0 = _dim(0)
    mock_divisible.return_value = True

    # Two raw dim strings map to the same unique dim
    stage = _make_stage(
        tile_options_raw={"Gemm_Left.D1": [128], "Gemm_Down.D2": [128]},
        dims_map={("Gemm_Left", 1): d0, ("Gemm_Down", 2): d0},
        dim_sizes={d0: 2048},
    )

    ss = stage._build_search_space()
    # Should only have one entry for d0, not duplicated
    options = ss.get(d0)
    assert len(options) == 1
    assert options[0].tile == 128  # noqa: PLR2004


@patch("stream.stages.generation.candidate_filter_stage.is_divisible_candidate")
def test_multiple_dims_tracked_separately(mock_divisible):
    d0 = _dim(0)
    d1 = _dim(1)
    mock_divisible.return_value = True

    tile_d0 = 16
    tile_d1 = 128
    stage = _make_stage(
        tile_options_raw={"Gemm.D0": [tile_d0], "Gemm.D1": [tile_d1]},
        dims_map={("Gemm", 0): d0, ("Gemm", 1): d1},
        dim_sizes={d0: 256, d1: 2048},
    )

    ss = stage._build_search_space()
    assert set(ss.dims()) == {d0, d1}
    assert ss.get(d0)[0].tile == tile_d0
    assert ss.get(d1)[0].tile == tile_d1
