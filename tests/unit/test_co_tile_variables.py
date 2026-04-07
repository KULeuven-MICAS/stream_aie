"""Unit tests for TransferAndTensorAllocator tile selection variables.

Tests w[dim,k] binary variables, tile_var[dim] INTEGER variables, one-hot
constraints, and joint candidate enumeration methods (D-01/D-02/D-03/D-04).
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import gurobipy as gp
import pytest
from gurobipy import GRB

from stream.datatypes import LayerDim
from stream.opt.search_space import SearchSpace, TileSizeOption


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dim(pos: int) -> LayerDim:
    return LayerDim(position=pos, prefix="z")


def _build_search_space(candidates_per_dim: dict[int, list[int]]) -> SearchSpace:
    ss = SearchSpace()
    for pos, tiles in candidates_per_dim.items():
        dim = _dim(pos)
        for tile in tiles:
            ss.add(dim, TileSizeOption(dim=dim, tile=tile, workload_size=256))
    return ss


def _make_allocator_stub(model: gp.Model, search_space: SearchSpace | None = None):
    """Build a minimal stub object that only has the attributes needed by
    __create_tile_selection_vars and the joint-candidate helpers.

    We avoid constructing a real TransferAndTensorAllocator (which requires
    a full workload/accelerator/etc.) by building a plain namespace object
    and binding the methods manually.
    """
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub = types.SimpleNamespace()
    stub.model = model
    stub.search_space = search_space
    stub.w = {}
    stub.tile_var = {}
    stub._tensor_max_size = {}
    stub._tensor_joint_candidates = {}

    # Bind the private methods as bound methods on the stub
    stub._TransferAndTensorAllocator__create_tile_selection_vars = (
        TransferAndTensorAllocator._TransferAndTensorAllocator__create_tile_selection_vars.__get__(stub)
    )
    stub._tiled_dims_for_tensor = (
        TransferAndTensorAllocator._tiled_dims_for_tensor.__get__(stub)
    )
    stub._joint_candidates_for_tensor = (
        TransferAndTensorAllocator._joint_candidates_for_tensor.__get__(stub)
    )
    stub._joint_binary_for_combo = (
        TransferAndTensorAllocator._joint_binary_for_combo.__get__(stub)
    )
    stub._add_binary_product = (
        TransferAndTensorAllocator._add_binary_product.__get__(stub)
    )
    stub._safe_name = (
        TransferAndTensorAllocator._safe_name.__get__(stub)
    )
    return stub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    m = gp.Model()
    m.setParam("OutputFlag", 0)
    return m


@pytest.fixture
def ss_2dims():
    """SearchSpace with dim(0)=[16,32,64] and dim(1)=[8,16]."""
    return _build_search_space({0: [16, 32, 64], 1: [8, 16]})


@pytest.fixture
def ss_1dim():
    """SearchSpace with dim(0)=[16,32]."""
    return _build_search_space({0: [16, 32]})


@pytest.fixture
def ss_single_candidate():
    """SearchSpace with one candidate per dim (degenerate case)."""
    return _build_search_space({0: [16], 1: [32]})


# ---------------------------------------------------------------------------
# Tests for __create_tile_selection_vars
# ---------------------------------------------------------------------------


def test_w_vars_created(model, ss_2dims):
    """5 w[dim,k] binary variables for 2 dims with 3 and 2 candidates."""
    stub = _make_allocator_stub(model, ss_2dims)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    # dim(0) has 3 candidates, dim(1) has 2 candidates → 5 w vars
    assert len(stub.w) == 5
    d0 = _dim(0)
    d1 = _dim(1)
    assert (d0, 0) in stub.w
    assert (d0, 1) in stub.w
    assert (d0, 2) in stub.w
    assert (d1, 0) in stub.w
    assert (d1, 1) in stub.w


def test_w_vars_are_binary(model, ss_2dims):
    """w[dim,k] variables must have BINARY variable type."""
    stub = _make_allocator_stub(model, ss_2dims)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    for var in stub.w.values():
        assert var.VType == GRB.BINARY


def test_one_hot_constraint_exists(model, ss_2dims):
    """Model contains w_one_hot_{dim} constraint for each dim."""
    stub = _make_allocator_stub(model, ss_2dims)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    constraint_names = {c.ConstrName for c in model.getConstrs()}
    for dim in ss_2dims.dims():
        assert f"w_one_hot_{dim}" in constraint_names


def test_tile_var_created(model, ss_2dims):
    """tile_var[dim] INTEGER variable exists for each dim."""
    stub = _make_allocator_stub(model, ss_2dims)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    d0 = _dim(0)
    d1 = _dim(1)
    assert d0 in stub.tile_var
    assert d1 in stub.tile_var
    assert stub.tile_var[d0].VType == GRB.INTEGER
    assert stub.tile_var[d1].VType == GRB.INTEGER


def test_tile_var_equality_constraint_exists(model, ss_1dim):
    """tile_var_def_{dim} constraint exists for each dim."""
    stub = _make_allocator_stub(model, ss_1dim)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    constraint_names = {c.ConstrName for c in model.getConstrs()}
    d0 = _dim(0)
    assert f"tile_var_def_{d0}" in constraint_names


def test_no_vars_when_search_space_none(model):
    """When search_space=None, w and tile_var remain empty."""
    stub = _make_allocator_stub(model, None)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    assert stub.w == {}
    assert stub.tile_var == {}


def test_no_vars_when_search_space_empty(model):
    """When search_space is empty, w and tile_var remain empty."""
    stub = _make_allocator_stub(model, SearchSpace())
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    assert stub.w == {}
    assert stub.tile_var == {}


# ---------------------------------------------------------------------------
# Tests for _joint_binary_for_combo
# ---------------------------------------------------------------------------


def _make_stub_with_w_vars(model: gp.Model, candidates_per_dim: dict[int, list[int]]):
    """Build stub with w vars pre-created for joint binary tests."""
    ss = _build_search_space(candidates_per_dim)
    stub = _make_allocator_stub(model, ss)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()
    return stub, ss


def test_joint_binary_single_dim(model):
    """For 1-dim, joint binary is just the w[dim,k] variable itself."""
    stub, ss = _make_stub_with_w_vars(model, {0: [16, 32]})
    d0 = _dim(0)
    options = ss.get(d0)
    per_dim_options = [(d0, options)]
    opt = options[0]  # tile=16, k=0
    combo = (opt,)
    result = stub._joint_binary_for_combo(per_dim_options, combo)
    model.update()
    # Single dim: must return w[(d0, 0)] directly, no auxiliary variable
    assert result is stub.w[(d0, 0)]


def test_joint_binary_multi_dim_uses_add_binary_product(model):
    """For 2-dim, joint binary uses _add_binary_product → creates auxiliary AND var."""
    stub, ss = _make_stub_with_w_vars(model, {0: [16, 32], 1: [8, 16]})
    d0 = _dim(0)
    d1 = _dim(1)
    opts0 = ss.get(d0)
    opts1 = ss.get(d1)
    per_dim_options = [(d0, opts0), (d1, opts1)]
    combo = (opts0[0], opts1[0])  # k=0 for both dims
    n_vars_before = model.NumVars
    result = stub._joint_binary_for_combo(per_dim_options, combo)
    model.update()
    # An auxiliary __and var should have been created
    assert model.NumVars > n_vars_before
    assert result.VType == GRB.BINARY


# ---------------------------------------------------------------------------
# Tests for _joint_candidates_for_tensor
# ---------------------------------------------------------------------------


def _make_stub_for_joint_candidates(
    model: gp.Model,
    candidates_per_dim: dict[int, list[int]],
    tiled_dims_positions: list[int],
):
    """Build stub ready for joint candidate tests.

    Sets up:
    - w vars for the search space
    - stub.workload with mocked methods
    - stub.mapping (MagicMock)
    """
    from stream.workload.workload import InEdge, OutEdge

    ss = _build_search_space(candidates_per_dim)
    stub = _make_allocator_stub(model, ss)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    tiled_dims = [_dim(p) for p in tiled_dims_positions]

    # Mock workload
    workload = MagicMock()
    stub.workload = workload
    stub.mapping = MagicMock()

    # Successor is a ComputationNode (not InEdge/OutEdge)
    succ_node = MagicMock()
    succ_node.__class__ = MagicMock  # not InEdge/OutEdge
    # Make isinstance checks work: succ_node is NOT an InEdge or OutEdge
    from unittest.mock import patch
    workload.successors.return_value = [succ_node]

    # get_unique_dims_inter_core_tiling returns tiling for tiled dims
    base_tiling = tuple((_dim(p), 4) for p in tiled_dims_positions)
    workload.get_unique_dims_inter_core_tiling.return_value = base_tiling

    # get_dimension_size returns 256 for all dims
    workload.get_dimension_size.return_value = 256

    # get_tensor_shape_with_tiling returns a simple shape
    workload.get_tensor_shape_with_tiling.return_value = (64, 128)

    return stub, ss, succ_node


def test_joint_candidates_single_dim_returns_pairs(model):
    """For 1-dim SearchSpace, returns one (size, var) pair per candidate."""
    stub, ss, succ_node = _make_stub_for_joint_candidates(
        model, {0: [16, 32]}, [0]
    )
    tensor = MagicMock()
    tensor.size_bits.return_value = 1024
    tr = MagicMock()
    tr.outputs = [tensor]

    results = stub._joint_candidates_for_tensor(tensor, tr)

    # 2 candidates → 2 pairs
    assert len(results) == 2
    for size, var in results:
        assert size == 1024
        assert hasattr(var, "VType")  # it's a gp.Var


def test_joint_candidates_populates_tensor_max_size(model):
    """_joint_candidates_for_tensor sets _tensor_max_size[tensor] as side-effect."""
    stub, ss, succ_node = _make_stub_for_joint_candidates(
        model, {0: [16, 32]}, [0]
    )
    tensor = MagicMock()
    tensor.size_bits.return_value = 2048
    tr = MagicMock()
    tr.outputs = [tensor]

    stub._joint_candidates_for_tensor(tensor, tr)
    assert tensor in stub._tensor_max_size
    assert stub._tensor_max_size[tensor] == 2048


def test_joint_candidates_caches_result(model):
    """Calling _joint_candidates_for_tensor twice returns same list (cached)."""
    stub, ss, succ_node = _make_stub_for_joint_candidates(
        model, {0: [16, 32]}, [0]
    )
    tensor = MagicMock()
    tensor.size_bits.return_value = 512
    tr = MagicMock()
    tr.outputs = [tensor]

    result1 = stub._joint_candidates_for_tensor(tensor, tr)
    result2 = stub._joint_candidates_for_tensor(tensor, tr)
    assert result1 is result2


def test_single_candidate_degenerate(model):
    """With 1 candidate per dim, exactly 1 joint combination exists (Pitfall 5 regression)."""
    stub, ss, succ_node = _make_stub_for_joint_candidates(
        model, {0: [16], 1: [32]}, [0, 1]
    )
    tensor = MagicMock()
    tensor.size_bits.return_value = 512
    tr = MagicMock()
    tr.outputs = [tensor]

    results = stub._joint_candidates_for_tensor(tensor, tr)
    # 1 candidate per dim → 1 * 1 = 1 joint combination
    assert len(results) == 1


def test_joint_candidates_no_tiled_dims_returns_empty(model):
    """When no SearchSpace dims appear in the successor's tiling, returns empty list."""
    from stream.workload.workload import InEdge, OutEdge

    ss = _build_search_space({0: [16, 32]})
    stub = _make_allocator_stub(model, ss)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    workload = MagicMock()
    stub.workload = workload
    stub.mapping = MagicMock()

    succ_node = MagicMock()
    workload.successors.return_value = [succ_node]
    # Successor tiling uses a DIFFERENT dim (pos=99) not in search_space
    workload.get_unique_dims_inter_core_tiling.return_value = ((_dim(99), 4),)
    workload.get_dimension_size.return_value = 256
    workload.get_tensor_shape_with_tiling.return_value = (64,)

    tensor = MagicMock()
    tensor.size_bits.return_value = 1024
    tr = MagicMock()
    tr.outputs = [tensor]

    results = stub._joint_candidates_for_tensor(tensor, tr)
    assert results == []
    # _tensor_max_size should still be set via fallback scalar path
    assert tensor in stub._tensor_max_size


# ---------------------------------------------------------------------------
# Helpers for memory capacity constraint tests
# ---------------------------------------------------------------------------


def _make_mem_constraint_stub(
    model: gp.Model,
    search_space: SearchSpace | None,
    tensor_size: int,
    size_factor: float,
    n_stops: int = 1,
    joint_candidates: list[tuple[int, gp.Var]] | None = None,
    tensor_max_size: int | None = None,
):
    """Build a stub ready for _memory_capacity_constraints tests.

    Pre-wires a single transfer node, single output tensor, and single core.
    joint_candidates overrides _joint_candidates_for_tensor when provided.
    """
    from collections import defaultdict
    from math import ceil

    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub = types.SimpleNamespace()
    stub.model = model
    stub.search_space = search_space
    stub.w = {}
    stub.tile_var = {}
    stub._tensor_max_size = {}
    stub._tensor_joint_candidates = {}

    # Bind helpers
    stub._add_binary_product = TransferAndTensorAllocator._add_binary_product.__get__(stub)
    stub._safe_name = TransferAndTensorAllocator._safe_name.__get__(stub)
    stub._joint_candidates_for_tensor = TransferAndTensorAllocator._joint_candidates_for_tensor.__get__(stub)
    stub._tiled_dims_for_tensor = TransferAndTensorAllocator._tiled_dims_for_tensor.__get__(stub)
    stub._joint_binary_for_combo = TransferAndTensorAllocator._joint_binary_for_combo.__get__(stub)

    # Build a Core mock using spec=Core so isinstance(core, Core) passes
    from stream.hardware.architecture.core import Core
    core = MagicMock(spec=Core)
    core.id = 42  # needed by _resource_key(core) -> f"Core {res.id}"
    core.get_memory_capacity.return_value = 2**30  # large cap, no infeasibility
    stub._core = core

    # Build a Tensor mock
    from stream.workload.workload import Tensor
    tensor = MagicMock(spec=Tensor)
    tensor.name = "test_tensor"
    tensor.size_bits.return_value = tensor_size
    stub._tensor = tensor

    # Workload mock for get_tensor_of_transfer_to_single_core
    scalar_tensor = MagicMock()
    scalar_tensor.size_bits.return_value = tensor_size
    workload = MagicMock()
    workload.get_tensor_of_transfer_to_single_core.return_value = scalar_tensor
    stub.workload = workload

    # Transfer node mock
    tr = MagicMock()
    tr.outputs = [tensor]
    stub.transfer_nodes = [tr]
    stub._tr = tr

    # SSIS mock for get_applicable_temporal_variables
    ssis_mock = MagicMock()
    temporal_vars = [MagicMock()] * n_stops
    ssis_mock.get_applicable_temporal_variables.return_value = temporal_vars
    stub.ssis = {tr: ssis_mock}

    # reuse_levels: stop==-1 always present + n_stops more stops
    stub.reuse_levels = {}
    for s in range(-1, n_stops):
        stub.reuse_levels[(tensor, s)] = (1, size_factor)

    # z_stop binary variables for (tensor, stop)
    stub.z_stop = {}
    for s in range(-1, n_stops):
        z = model.addVar(vtype=GRB.BINARY, name=f"z_stop_{s}")
        stub.z_stop[(tensor, s)] = z
    model.update()

    # _candidate_cores_for_tensor returns {core}
    stub._candidate_cores_for_tensor = lambda t: {core}

    # _tensor_uses_core_var: returns a fixed binary var
    u = model.addVar(vtype=GRB.BINARY, name="u_tensor_core")
    model.addConstr(u == 1, name="u_fixed")  # always uses this core for tests
    model.update()
    stub.tensor_core_indicator = {}
    stub._tensor_uses_core_var = lambda t, c: u
    stub._u = u

    # Override joint candidates if provided
    if joint_candidates is not None:
        stub._tensor_joint_candidates[tensor] = joint_candidates
        if tensor_max_size is not None:
            stub._tensor_max_size[tensor] = tensor_max_size
        elif joint_candidates:
            stub._tensor_max_size[tensor] = max(sz for sz, _ in joint_candidates)

    # Bind _memory_capacity_constraints
    stub._memory_capacity_constraints = (
        TransferAndTensorAllocator._memory_capacity_constraints.__get__(stub)
    )

    stub.mapping = MagicMock()

    return stub


# ---------------------------------------------------------------------------
# Tests for _memory_capacity_constraints rewrite
# ---------------------------------------------------------------------------


def test_memory_constraint_uses_load_contrib(model):
    """With search_space set and 2 candidates, verify lc_* vars and constraints exist."""
    ss = _build_search_space({0: [16, 32]})

    # Build 2 joint candidate (size, binary_var) pairs
    jw0 = model.addVar(vtype=GRB.BINARY, name="jw0")
    jw1 = model.addVar(vtype=GRB.BINARY, name="jw1")
    model.update()
    joint_candidates = [(1024, jw0), (512, jw1)]

    stub = _make_mem_constraint_stub(
        model,
        search_space=ss,
        tensor_size=1024,
        size_factor=1.0,
        n_stops=1,
        joint_candidates=joint_candidates,
        tensor_max_size=1024,
    )
    stub._memory_capacity_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    constr_names = {c.ConstrName for c in model.getConstrs()}

    # Verify load_contrib variable exists
    lc_vars = [n for n in var_names if n.startswith("lc_")]
    assert len(lc_vars) >= 1, f"Expected lc_* vars, got: {var_names}"

    # Verify big-M activation constraints
    lc_ub_expr = [n for n in constr_names if n.startswith("lc_ub_expr_")]
    lc_ub_m = [n for n in constr_names if n.startswith("lc_ub_m_")]
    lc_lb = [n for n in constr_names if n.startswith("lc_lb_")]
    assert len(lc_ub_expr) >= 1, f"Missing lc_ub_expr constraints, got: {constr_names}"
    assert len(lc_ub_m) >= 1, f"Missing lc_ub_m constraints, got: {constr_names}"
    assert len(lc_lb) >= 1, f"Missing lc_lb constraints, got: {constr_names}"


def test_memory_constraint_scalar_fallback(model):
    """With search_space=None, no lc_* variables are created."""
    stub = _make_mem_constraint_stub(
        model,
        search_space=None,
        tensor_size=1024,
        size_factor=1.0,
        n_stops=1,
    )
    stub._memory_capacity_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("lc_")]
    assert len(lc_vars) == 0, f"Expected no lc_* vars in scalar fallback, got: {lc_vars}"

    # mem_cap constraint must still exist
    constr_names = {c.ConstrName for c in model.getConstrs()}
    mem_cap = [n for n in constr_names if n.startswith("mem_cap_")]
    assert len(mem_cap) >= 1, "Expected mem_cap_ constraint in scalar fallback"


def test_tight_bigm_not_legacy(model):
    """Big-M for load_contrib upper bound equals ceil(size_factor * max_tensor_size)."""
    from math import ceil

    ss = _build_search_space({0: [16, 32]})
    size_factor = 2.0

    jw0 = model.addVar(vtype=GRB.BINARY, name="jw0_bm")
    jw1 = model.addVar(vtype=GRB.BINARY, name="jw1_bm")
    model.update()
    # Two candidate sizes: 1024 and 512; max is 1024
    joint_candidates = [(1024, jw0), (512, jw1)]
    max_size = 1024
    legacy_bigm = 10  # simulated len(nodes)+5 value, smaller than tight M

    stub = _make_mem_constraint_stub(
        model,
        search_space=ss,
        tensor_size=1024,
        size_factor=size_factor,
        n_stops=0,  # only stop=-1
        joint_candidates=joint_candidates,
        tensor_max_size=max_size,
    )
    stub._memory_capacity_constraints()
    model.update()

    # Find lc_* variables and check their upper bound
    lc_vars = [v for v in model.getVars() if v.VarName.startswith("lc_")]
    assert len(lc_vars) >= 1, "No lc_* variable found"

    expected_M = ceil(size_factor * max_size)  # = ceil(2.0 * 1024) = 2048
    assert expected_M != legacy_bigm, "Test setup: M must differ from legacy value"
    for lc in lc_vars:
        assert lc.UB == expected_M, (
            f"Expected lc.UB == {expected_M} (tight), got {lc.UB} (legacy would be {legacy_bigm})"
        )


def test_single_candidate_regression_compat(model):
    """With 1 candidate per dim, model is feasible and w[dim,0].X == 1."""
    ss = _build_search_space({0: [16]})
    stub = _make_allocator_stub(model, ss)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    # Verify w[(dim,0)] exists and has the right type
    d0 = _dim(0)
    assert (d0, 0) in stub.w
    w_var = stub.w[(d0, 0)]
    assert w_var.VType == GRB.BINARY

    # Force the one-hot: with 1 candidate, w[d0,0] must be 1
    model.optimize()
    assert model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL
    model.update()
    # The one-hot constraint forces w[d0,0] == 1
    assert abs(w_var.X - 1.0) < 1e-6, f"Expected w[dim,0].X == 1.0, got {w_var.X}"

    # tile_var[d0] should equal the single tile size (16)
    tile_v = stub.tile_var[d0]
    assert abs(tile_v.X - 16.0) < 1e-6, f"Expected tile_var[dim].X == 16.0, got {tile_v.X}"
