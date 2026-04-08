"""Unit tests for TileAwareLatencyEstimator.

Tests the estimate() method formula: ceil(MACs / floor(ideal_ops * util/100)).
Uses mocks for workload, mapping, node, core, and kernel.
"""
from __future__ import annotations

from math import ceil, floor
from unittest.mock import MagicMock, patch

import pytest

from stream.datatypes import LayerDim


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _dim(pos: int) -> LayerDim:
    return LayerDim(position=pos, prefix="z")


def _make_node_mock(dim_sizes: list[int]) -> MagicMock:
    """Create a ComputationNode mock with the given dimension sizes."""
    node = MagicMock()
    node.name = "test_node"
    return node


def _make_core_mock(core_type: str = "aie2.compute") -> MagicMock:
    """Create a Core mock with the given core_type."""
    core = MagicMock()
    core.core_type = core_type
    return core


def _make_kernel_mock(utilization: float) -> MagicMock:
    """Create a kernel mock with the given utilization."""
    kernel = MagicMock()
    kernel.utilization = utilization
    return kernel


def _make_workload_mock(dims: list[LayerDim], dim_sizes: list[int]) -> MagicMock:
    """Create a Workload mock that returns given dims and sizes."""
    workload = MagicMock()
    workload.get_dims.return_value = dims
    # get_dimension_size returns the right size for each dim
    size_map = dict(zip(dims, dim_sizes))
    workload.get_dimension_size.side_effect = lambda d: size_map[d]
    return workload


def _make_mapping_mock(node: MagicMock, kernel: MagicMock) -> MagicMock:
    """Create a Mapping mock that returns the kernel for the given node."""
    mapping = MagicMock()
    node_mapping = MagicMock()
    node_mapping.kernel = kernel
    mapping.get.return_value = node_mapping
    return mapping


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_estimate_known_macs_and_utilization():
    """Test 1: estimate() with known MACs and utilization returns ceil(MACs / floor(ideal_ops * util/100)).

    Setup: dims [z0, z1] with sizes [16, 32] -> MACs = 512 (no tiling)
    ideal_ops = 32, utilization = 80%
    ops_per_cycle = floor(32 * 80 / 100) = floor(25.6) = 25
    cycles = ceil(512 / 25) = ceil(20.48) = 21
    """
    from stream.cost_model.tile_aware_latency import LatencyEstimate, TileAwareLatencyEstimator

    dims = [_dim(0), _dim(1)]
    node = _make_node_mock([16, 32])
    core = _make_core_mock("aie2.compute")
    kernel = _make_kernel_mock(utilization=80.0)
    workload = _make_workload_mock(dims, [16, 32])
    mapping = _make_mapping_mock(node, kernel)

    estimator = TileAwareLatencyEstimator(workload=workload, mapping=mapping)

    # Patch AIECostEstimator.ops_per_cycle to return 32
    with patch.object(estimator._aie, "ops_per_cycle", return_value=32):
        result = estimator.estimate(node, core, inter_core_tiling=())

    macs = 16 * 32  # = 512
    ideal_ops = 32
    utilization = 80.0
    ops_per_cycle = floor(ideal_ops * utilization / 100.0)  # floor(25.6) = 25
    expected_cycles = ceil(macs / ops_per_cycle)  # ceil(20.48) = 21

    assert isinstance(result, LatencyEstimate)
    assert result.latency_total == expected_cycles, f"Expected {expected_cycles}, got {result.latency_total}"


def test_estimate_perfect_utilization():
    """Test 2: estimate() with perfect utilization (100%) returns ceil(MACs / ideal_ops).

    Setup: 1 dim with size 64 -> MACs = 64
    ideal_ops = 32, utilization = 100%
    ops_per_cycle = floor(32 * 100 / 100) = 32
    cycles = ceil(64 / 32) = 2
    """
    from stream.cost_model.tile_aware_latency import LatencyEstimate, TileAwareLatencyEstimator

    dims = [_dim(0)]
    node = _make_node_mock([64])
    core = _make_core_mock("aie2.compute")
    kernel = _make_kernel_mock(utilization=100.0)
    workload = _make_workload_mock(dims, [64])
    mapping = _make_mapping_mock(node, kernel)

    estimator = TileAwareLatencyEstimator(workload=workload, mapping=mapping)

    with patch.object(estimator._aie, "ops_per_cycle", return_value=32):
        result = estimator.estimate(node, core, inter_core_tiling=())

    expected_cycles = ceil(64 / 32)  # = 2

    assert result.latency_total == expected_cycles, f"Expected {expected_cycles}, got {result.latency_total}"


def test_estimate_returns_ideal_cycle_independently():
    """Test 3: estimate() returns ideal_cycle = ceil(MACs / ideal_ops) regardless of utilization.

    Setup: dims [z0] with size 100, ideal_ops=32, utilization=50%
    ideal_cycles = ceil(100 / 32) = 4  (ignores utilization)
    ops_per_cycle = floor(32 * 50 / 100) = 16
    cycles = ceil(100 / 16) = 7
    """
    from stream.cost_model.tile_aware_latency import TileAwareLatencyEstimator

    dims = [_dim(0)]
    node = _make_node_mock([100])
    core = _make_core_mock("aie2.compute")
    kernel = _make_kernel_mock(utilization=50.0)
    workload = _make_workload_mock(dims, [100])
    mapping = _make_mapping_mock(node, kernel)

    estimator = TileAwareLatencyEstimator(workload=workload, mapping=mapping)

    with patch.object(estimator._aie, "ops_per_cycle", return_value=32):
        result = estimator.estimate(node, core, inter_core_tiling=())

    expected_ideal = ceil(100 / 32)  # = 4
    assert result.ideal_cycle == expected_ideal, f"Expected ideal_cycle={expected_ideal}, got {result.ideal_cycle}"

    # Also verify the actual cycles are different (not same as ideal)
    expected_cycles = ceil(100 / floor(32 * 50 / 100))  # ceil(100/16) = 7
    assert result.latency_total == expected_cycles


def test_estimate_inter_core_tiling_reduces_macs():
    """Test 4: estimate() with inter_core_tiling reduces MACs proportionally.

    Setup: dims [z0, z1] with sizes [16, 32] -> total = 512
    inter_core_tiling = ((z0, 2),) -> tiling_factor = 2 -> MACs = 512 // 2 = 256
    ideal_ops = 32, utilization = 100%
    cycles = ceil(256 / 32) = 8
    """
    from stream.cost_model.tile_aware_latency import TileAwareLatencyEstimator

    dims = [_dim(0), _dim(1)]
    node = _make_node_mock([16, 32])
    core = _make_core_mock("aie2.compute")
    kernel = _make_kernel_mock(utilization=100.0)
    workload = _make_workload_mock(dims, [16, 32])
    mapping = _make_mapping_mock(node, kernel)

    estimator = TileAwareLatencyEstimator(workload=workload, mapping=mapping)

    tiling = ((_dim(0), 2),)
    with patch.object(estimator._aie, "ops_per_cycle", return_value=32):
        result = estimator.estimate(node, core, inter_core_tiling=tiling)

    expected_cycles = ceil(256 / 32)  # = 8
    assert result.latency_total == expected_cycles, f"Expected {expected_cycles}, got {result.latency_total}"


def test_ops_per_cycle_bf16_aie2():
    """Test 5: ops_per_cycle dispatches correctly for bf16 on aie2.compute (returns 32).

    This delegates to AIECostEstimator.ops_per_cycle, so we verify the real call
    returns 32 for BFloat16/BFloat16 on aie2.compute.
    """
    from xdsl.dialects.builtin import BFloat16Type

    from stream.cost_model.tile_aware_latency import TileAwareLatencyEstimator

    workload = MagicMock()
    mapping = MagicMock()
    estimator = TileAwareLatencyEstimator(workload=workload, mapping=mapping)

    node = MagicMock()
    bf16 = BFloat16Type()
    inp = MagicMock()
    inp.operand_type = bf16
    node.inputs = [inp]
    out = MagicMock()
    out.operand_type = bf16
    node.outputs = [out]

    core = _make_core_mock("aie2.compute")

    result = estimator._aie.ops_per_cycle(node, core)
    assert result == 32, f"Expected 32 ops/cycle for bf16 on aie2.compute, got {result}"


def test_estimate_energy_is_zero():
    """estimate() returns energy_total == 0.0 (placeholder)."""
    from stream.cost_model.tile_aware_latency import TileAwareLatencyEstimator

    dims = [_dim(0)]
    node = _make_node_mock([64])
    core = _make_core_mock("aie2.compute")
    kernel = _make_kernel_mock(utilization=100.0)
    workload = _make_workload_mock(dims, [64])
    mapping = _make_mapping_mock(node, kernel)

    estimator = TileAwareLatencyEstimator(workload=workload, mapping=mapping)

    with patch.object(estimator._aie, "ops_per_cycle", return_value=32):
        result = estimator.estimate(node, core, inter_core_tiling=())

    assert result.energy_total == 0.0
