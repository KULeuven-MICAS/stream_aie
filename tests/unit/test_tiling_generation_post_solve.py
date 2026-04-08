"""Unit tests for TilingGenerationStage post-solve mode.

Tests that TilingGenerationStage:
- In post-solve mode (selected_tiles in ctx): uses CO-solved tiles, does NOT
  call determine_fusion_splits(), asserts divisibility.
- In legacy mode (no selected_tiles in ctx): preserves existing behaviour
  (calls determine_fusion_splits()).

Mocking approach: use unittest.mock.MagicMock / patch for workload + mapping so
we don't need a real ONNX model or hardware description.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from stream.datatypes import LayerDim
from stream.stages.context import StageContext
from stream.stages.generation.tiling_generation import TilingGenerationStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dim(name: str) -> LayerDim:
    return LayerDim(position=0, prefix=name)


def _build_ctx(**kwargs) -> StageContext:
    defaults = dict(output_path="/tmp/test_tiling_gen")
    defaults.update(kwargs)
    return StageContext.from_kwargs(**defaults)


def _build_stage(ctx: StageContext, workload, mapping) -> TilingGenerationStage:
    """Construct a TilingGenerationStage with a mock leaf callable."""
    from stream.stages.stage import LeafStage

    ctx.set(workload=workload, mapping=mapping)
    return TilingGenerationStage([LeafStage], ctx)


def _make_mock_workload(dim_sizes: dict[LayerDim, int]):
    """Create a lightweight workload mock that returns the given dimension sizes."""
    wl = MagicMock()
    unique_dims = list(dim_sizes.keys())
    wl.unique_dimensions.return_value = (unique_dims, [])
    wl.get_dimension_size.side_effect = lambda d: dim_sizes[d]
    # with_modified_dimension_sizes returns a new mock workload
    tiled_wl = MagicMock()
    tiled_wl.visualize = MagicMock()
    wl.with_modified_dimension_sizes.return_value = tiled_wl
    return wl, tiled_wl


def _make_mock_mapping(tiled_wl, wl):
    """Create a mapping mock."""
    mapping = MagicMock()
    tiled_mapping = MagicMock()
    mapping.with_updated_workload.return_value = tiled_mapping
    return mapping, tiled_mapping


# ---------------------------------------------------------------------------
# Tests: Post-solve mode (selected_tiles present)
# ---------------------------------------------------------------------------


class TestPostSolveMode:
    """TilingGenerationStage with selected_tiles in ctx (post-solve mode)."""

    def test_does_not_call_determine_fusion_splits_in_post_solve_mode(self, tmp_path):
        """When selected_tiles is in ctx, determine_fusion_splits must NOT be called."""
        dim_d = _dim("D")
        dim_e = _dim("E")
        dim_sizes = {dim_d: 256, dim_e: 512}
        wl, tiled_wl = _make_mock_workload(dim_sizes)
        mapping, tiled_mapping = _make_mock_mapping(tiled_wl, wl)

        fusion_splits = {dim_d: 16}
        selected_tiles = {dim_d: 16}
        ctx = _build_ctx(
            output_path=str(tmp_path),
            fusion_splits=fusion_splits,
            selected_tiles=selected_tiles,
        )
        stage = _build_stage(ctx, wl, mapping)

        with patch("stream.stages.generation.tiling_generation.determine_fusion_splits") as mock_dfs:
            # Run the generator partially (just the first ctx.set call)
            list(stage.run())
            mock_dfs.assert_not_called()

    def test_uses_selected_tiles_for_tiled_dimensions(self, tmp_path):
        """Post-solve mode passes selected_tiles as the tiled size for tiled dims."""
        dim_d = _dim("D")
        dim_e = _dim("E")
        dim_sizes = {dim_d: 256, dim_e: 512}
        wl, tiled_wl = _make_mock_workload(dim_sizes)
        mapping, tiled_mapping = _make_mock_mapping(tiled_wl, wl)

        selected_tiles = {dim_d: 32}
        fusion_splits = {dim_d: 8}  # 256 / 32 = 8 splits; fusion_splits value is split count
        ctx = _build_ctx(
            output_path=str(tmp_path),
            fusion_splits=fusion_splits,
            selected_tiles=selected_tiles,
        )
        stage = _build_stage(ctx, wl, mapping)
        list(stage.run())

        # with_modified_dimension_sizes should have been called with {dim_d: 32, dim_e: 512}
        call_args = wl.with_modified_dimension_sizes.call_args[0][0]
        assert call_args[dim_d] == 32, f"Expected tiled dim_d=32, got {call_args[dim_d]}"
        assert call_args[dim_e] == 512, f"Expected non-tiled dim_e=512, got {call_args[dim_e]}"

    def test_non_tiled_dims_keep_original_size_in_post_solve_mode(self, tmp_path):
        """Dimensions not in selected_tiles should keep their original workload size."""
        dim_s = _dim("S")
        dim_h = _dim("H")
        dim_sizes = {dim_s: 128, dim_h: 1024}
        wl, tiled_wl = _make_mock_workload(dim_sizes)
        mapping, tiled_mapping = _make_mock_mapping(tiled_wl, wl)

        # Only dim_s is tiled; dim_h keeps its size
        selected_tiles = {dim_s: 16}
        fusion_splits = {dim_s: 8}
        ctx = _build_ctx(
            output_path=str(tmp_path),
            fusion_splits=fusion_splits,
            selected_tiles=selected_tiles,
        )
        stage = _build_stage(ctx, wl, mapping)
        list(stage.run())

        call_args = wl.with_modified_dimension_sizes.call_args[0][0]
        assert call_args[dim_s] == 16
        assert call_args[dim_h] == 1024

    def test_raises_if_tile_does_not_divide_workload_size(self, tmp_path):
        """Post-solve mode asserts tile_size divides workload_size."""
        dim_d = _dim("D")
        dim_sizes = {dim_d: 100}
        wl, tiled_wl = _make_mock_workload(dim_sizes)
        mapping, tiled_mapping = _make_mock_mapping(tiled_wl, wl)

        # 100 is not divisible by 32
        selected_tiles = {dim_d: 32}
        fusion_splits = {}
        ctx = _build_ctx(
            output_path=str(tmp_path),
            fusion_splits=fusion_splits,
            selected_tiles=selected_tiles,
        )
        stage = _build_stage(ctx, wl, mapping)
        with pytest.raises(AssertionError, match="not divisible"):
            list(stage.run())


# ---------------------------------------------------------------------------
# Tests: Legacy mode (no selected_tiles in ctx)
# ---------------------------------------------------------------------------


class TestLegacyMode:
    """TilingGenerationStage without selected_tiles in ctx (legacy / pre-solve mode)."""

    def test_calls_determine_fusion_splits_in_legacy_mode(self, tmp_path):
        """When selected_tiles is NOT in ctx, determine_fusion_splits should be called."""
        dim_d = _dim("D")
        dim_sizes = {dim_d: 256}
        wl, tiled_wl = _make_mock_workload(dim_sizes)
        mapping, tiled_mapping = _make_mock_mapping(tiled_wl, wl)

        ctx = _build_ctx(output_path=str(tmp_path))
        stage = _build_stage(ctx, wl, mapping)

        fake_fusion_splits = {dim_d: 16}
        with patch(
            "stream.stages.generation.tiling_generation.determine_fusion_splits",
            return_value=fake_fusion_splits,
        ) as mock_dfs:
            list(stage.run())
            mock_dfs.assert_called_once()

    def test_selected_tiles_attribute_is_none_when_not_in_ctx(self, tmp_path):
        """stage.selected_tiles should be None when not set in ctx."""
        dim_d = _dim("D")
        dim_sizes = {dim_d: 256}
        wl, _ = _make_mock_workload(dim_sizes)
        mapping = MagicMock()

        ctx = _build_ctx(output_path=str(tmp_path))
        ctx.set(workload=wl, mapping=mapping)
        from stream.stages.stage import LeafStage

        stage = TilingGenerationStage([LeafStage], ctx)
        assert stage.selected_tiles is None
