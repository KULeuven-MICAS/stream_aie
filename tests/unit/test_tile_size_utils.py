import pytest

from stream.datatypes import LayerDim
from stream.opt.search_space import SearchSpace, TileSizeOption

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
