"""Unit tests for tile-aware SteadyStateIterationSpace extensions.

Tests for:
- SteadyStateIterationSpace.candidate_loop_sizes (D-01)
- generate_steady_state_iteration_spaces with optional search_space (D-03)
- ssis_loop_sizes_for_candidate utility
- reuse_coefficients_for_sizes utility
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stream.datatypes import LayerDim
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    LoopEffect,
    SteadyStateIterationSpace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dim(pos: int) -> LayerDim:
    return LayerDim(position=pos, prefix="z")


def _make_temporal_var(dim: LayerDim, size: int, effect: LoopEffect = LoopEffect.VARYING) -> IterationVariable:
    return IterationVariable(dimension=dim, size=size, effect=effect, type=IterationVariableType.TEMPORAL)


def _make_ssis(*dims_sizes_effects: tuple[LayerDim, int, LoopEffect]) -> SteadyStateIterationSpace:
    """Build an SSIS with temporal variables from (dim, size, effect) triples."""
    variables = [_make_temporal_var(d, s, e) for d, s, e in dims_sizes_effects]
    return SteadyStateIterationSpace(variables)


# ---------------------------------------------------------------------------
# Tests for SteadyStateIterationSpace.candidate_loop_sizes (D-01)
# ---------------------------------------------------------------------------


def test_candidate_loop_sizes_single_candidate():
    """Single candidate: verify (K, T) = (tile, workload_size / (S * tile))."""
    dim_m = _dim(0)
    dim_k = _dim(1)
    ssis = _make_ssis(
        (dim_m, 16, LoopEffect.VARYING),
        (dim_k, 8, LoopEffect.VARYING),
    )
    # workload_size=256, S=1, candidate=16 → K=16, T=256/(1*16)=16
    result = ssis.candidate_loop_sizes(dim=dim_m, candidates=[16], workload_size=256, S=1)
    assert result == {16: (16, 16)}


def test_candidate_loop_sizes_multiple_candidates():
    """Multiple candidates: verify correct {K: (K, T)} dict entries."""
    dim_m = _dim(0)
    ssis = _make_ssis(
        (dim_m, 16, LoopEffect.VARYING),
    )
    # workload_size=256, S=1
    # tile=16: K=16, T=256/(1*16)=16
    # tile=32: K=32, T=256/(1*32)=8
    result = ssis.candidate_loop_sizes(dim=dim_m, candidates=[16, 32], workload_size=256, S=1)
    assert result == {16: (16, 16), 32: (32, 8)}


def test_candidate_loop_sizes_with_spatial_unrolling():
    """S > 1: T = workload_size / (S * K)."""
    dim_k = _dim(1)
    ssis = _make_ssis(
        (dim_k, 4, LoopEffect.VARYING),
    )
    # workload_size=2048, S=4, candidate=128 → K=128, T=2048/(4*128)=4
    result = ssis.candidate_loop_sizes(dim=dim_k, candidates=[128], workload_size=2048, S=4)
    assert result == {128: (128, 4)}


def test_candidate_loop_sizes_non_temporal_dim():
    """Dim not in SSIS temporal variables: return empty dict."""
    dim_m = _dim(0)
    dim_other = _dim(99)  # not in SSIS
    ssis = _make_ssis(
        (dim_m, 16, LoopEffect.VARYING),
    )
    result = ssis.candidate_loop_sizes(dim=dim_other, candidates=[16, 32], workload_size=256, S=1)
    assert result == {}


def test_candidate_loop_sizes_indivisible_raises():
    """workload_size % (S*K) != 0: raises AssertionError."""
    dim_m = _dim(0)
    ssis = _make_ssis(
        (dim_m, 16, LoopEffect.VARYING),
    )
    with pytest.raises(AssertionError):
        ssis.candidate_loop_sizes(dim=dim_m, candidates=[33], workload_size=256, S=1)


def test_candidate_loop_sizes_absent_dim_not_counted():
    """Dim with LoopEffect.ABSENT is not an applicable temporal dim; returns empty dict."""
    dim_m = _dim(0)
    dim_absent = _dim(5)
    ssis = SteadyStateIterationSpace([
        _make_temporal_var(dim_m, 16, LoopEffect.VARYING),
        _make_temporal_var(dim_absent, 4, LoopEffect.ABSENT),
    ])
    # dim_absent should NOT be in applicable_temporal_dims (it's ABSENT)
    result = ssis.candidate_loop_sizes(dim=dim_absent, candidates=[4], workload_size=256, S=1)
    assert result == {}


def test_candidate_loop_sizes_empty_candidates():
    """Empty candidate list: return empty dict."""
    dim_m = _dim(0)
    ssis = _make_ssis(
        (dim_m, 16, LoopEffect.VARYING),
    )
    result = ssis.candidate_loop_sizes(dim=dim_m, candidates=[], workload_size=256, S=1)
    assert result == {}


# ---------------------------------------------------------------------------
# Tests for generate_steady_state_iteration_spaces with search_space (D-03)
# ---------------------------------------------------------------------------


@patch("stream.workload.utils.collect_spatial_unrollings")
@patch("stream.workload.utils._create_steady_state_iteration_spaces")
@patch("stream.workload.utils._add_temporal_iteration_variables")
@patch("stream.workload.utils._insert_kernel_iteration_variables")
@patch("stream.workload.utils._create_spatial_iteration_variables")
def test_generate_ssis_backward_compat(
    mock_create_spatial,
    mock_insert_kernel,
    mock_add_temporal,
    mock_create_ssis,
    mock_collect,
):
    """search_space=None: function still accepts the call and doesn't error."""
    from stream.workload.utils import generate_steady_state_iteration_spaces

    mock_collect.return_value = ({}, [])
    mock_create_spatial.return_value = {}
    mock_add_temporal.return_value = {}
    mock_insert_kernel.return_value = {}
    expected_result = {MagicMock(): MagicMock()}
    mock_create_ssis.return_value = expected_result

    workload = MagicMock()
    mapping = MagicMock()
    fusion_splits = {}

    result = generate_steady_state_iteration_spaces(workload, mapping, fusion_splits, search_space=None)
    assert result is expected_result


def test_generate_ssis_stores_search_space_on_ssis_objects():
    """When search_space is passed, it is forwarded to _create_steady_state_iteration_spaces."""
    from stream.opt.search_space import SearchSpace
    from stream.workload.utils import generate_steady_state_iteration_spaces

    ss = SearchSpace()
    dim_m = _dim(0)

    # Build a minimal SSIS object
    ssis_obj = _make_ssis((dim_m, 4, LoopEffect.VARYING))

    with patch("stream.workload.utils.collect_spatial_unrollings") as mock_collect, \
         patch("stream.workload.utils._create_spatial_iteration_variables") as mock_spatial, \
         patch("stream.workload.utils._add_temporal_iteration_variables") as mock_add, \
         patch("stream.workload.utils._insert_kernel_iteration_variables") as mock_insert, \
         patch("stream.workload.utils._create_steady_state_iteration_spaces") as mock_create:

        mock_collect.return_value = ({}, [])
        mock_spatial.return_value = {}
        mock_add.return_value = {}
        mock_insert.return_value = {}

        node = MagicMock()
        mock_create.return_value = {node: ssis_obj}

        workload = MagicMock()
        mapping = MagicMock()
        generate_steady_state_iteration_spaces(workload, mapping, {}, search_space=ss)

    # Verify search_space was forwarded to _create_steady_state_iteration_spaces
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert "search_space" in call_kwargs, (
        f"Expected search_space keyword arg, got: {call_kwargs}"
    )
    assert call_kwargs["search_space"] is ss
