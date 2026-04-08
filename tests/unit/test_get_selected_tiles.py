"""Unit tests for TransferAndTensorAllocator.get_selected_tiles().

Tests that get_selected_tiles() correctly reads w[dim,k] solved values
and returns the right {dim: tile_size} dict.  Follows the established
types.SimpleNamespace stub pattern from test_co_tile_variables.py.
"""
from __future__ import annotations

import types

import pytest

from stream.datatypes import LayerDim
from stream.opt.search_space import SearchSpace, TileSizeOption
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    TransferAndTensorAllocator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dim(name: str) -> LayerDim:
    return LayerDim(position=0, prefix=name)


def _mock_var(value: float):
    """Create a simple namespace that mimics a solved Gurobi binary variable."""
    v = types.SimpleNamespace()
    v.X = value
    return v


def _build_stub(w: dict, search_space: SearchSpace):
    """Build a minimal allocator stub with only the attributes used by get_selected_tiles()."""
    stub = types.SimpleNamespace()
    stub.w = w
    stub.search_space = search_space
    stub.VAR_THRESHOLD = TransferAndTensorAllocator.VAR_THRESHOLD
    # Bind get_selected_tiles as a bound method on the stub
    stub.get_selected_tiles = TransferAndTensorAllocator.get_selected_tiles.__get__(stub)
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetSelectedTiles:
    """Tests for TransferAndTensorAllocator.get_selected_tiles()."""

    def test_empty_w_returns_empty_dict(self):
        """When self.w is empty (no tile variables), return {}."""
        ss = SearchSpace()
        stub = _build_stub(w={}, search_space=ss)
        result = stub.get_selected_tiles()
        assert result == {}

    def test_single_candidate_returns_that_tile(self):
        """Single candidate per dim: w[dim,0].X = 1.0 => returns that tile."""
        dim_d = _dim("D")
        ss = SearchSpace()
        ss.add(dim_d, TileSizeOption(dim=dim_d, tile=16, workload_size=256))
        w = {(dim_d, 0): _mock_var(1.0)}
        stub = _build_stub(w=w, search_space=ss)
        result = stub.get_selected_tiles()
        assert result == {dim_d: 16}

    def test_multi_candidate_selects_correct_tile(self):
        """Three candidates: w[dim,1].X = 1.0, others 0.0 => returns second tile (32)."""
        dim_e = _dim("E")
        ss = SearchSpace()
        ss.add(dim_e, TileSizeOption(dim=dim_e, tile=16, workload_size=256))
        ss.add(dim_e, TileSizeOption(dim=dim_e, tile=32, workload_size=256))
        ss.add(dim_e, TileSizeOption(dim=dim_e, tile=64, workload_size=256))
        w = {
            (dim_e, 0): _mock_var(0.0),
            (dim_e, 1): _mock_var(1.0),
            (dim_e, 2): _mock_var(0.0),
        }
        stub = _build_stub(w=w, search_space=ss)
        result = stub.get_selected_tiles()
        assert result == {dim_e: 32}

    def test_multi_dim_selects_correct_tiles_per_dim(self):
        """Two dims with different selected candidates."""
        dim_s = _dim("S")
        dim_h = _dim("H")
        ss = SearchSpace()
        ss.add(dim_s, TileSizeOption(dim=dim_s, tile=8, workload_size=128))
        ss.add(dim_s, TileSizeOption(dim=dim_s, tile=16, workload_size=128))
        ss.add(dim_h, TileSizeOption(dim=dim_h, tile=32, workload_size=512))
        ss.add(dim_h, TileSizeOption(dim=dim_h, tile=64, workload_size=512))
        w = {
            (dim_s, 0): _mock_var(0.0),
            (dim_s, 1): _mock_var(1.0),
            (dim_h, 0): _mock_var(1.0),
            (dim_h, 1): _mock_var(0.0),
        }
        stub = _build_stub(w=w, search_space=ss)
        result = stub.get_selected_tiles()
        assert result == {dim_s: 16, dim_h: 32}

    def test_var_threshold_respected(self):
        """Values just above threshold (0.51) count as selected; 0.49 do not."""
        dim_k = _dim("K")
        ss = SearchSpace()
        ss.add(dim_k, TileSizeOption(dim=dim_k, tile=8, workload_size=64))
        ss.add(dim_k, TileSizeOption(dim=dim_k, tile=16, workload_size=64))
        # k=0 is 0.49 (below threshold), k=1 is 0.51 (above threshold)
        w = {
            (dim_k, 0): _mock_var(0.49),
            (dim_k, 1): _mock_var(0.51),
        }
        stub = _build_stub(w=w, search_space=ss)
        result = stub.get_selected_tiles()
        assert result == {dim_k: 16}
