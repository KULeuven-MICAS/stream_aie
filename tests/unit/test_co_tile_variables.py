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
    stub._tiled_dims_for_tensor = TransferAndTensorAllocator._tiled_dims_for_tensor.__get__(stub)
    stub._joint_candidates_for_tensor = TransferAndTensorAllocator._joint_candidates_for_tensor.__get__(stub)
    stub._joint_binary_for_combo = TransferAndTensorAllocator._joint_binary_for_combo.__get__(stub)
    stub._add_binary_product = TransferAndTensorAllocator._add_binary_product.__get__(stub)
    stub._safe_name = TransferAndTensorAllocator._safe_name.__get__(stub)
    # Phase 7: methods now access _orig_workload/_orig_mapping for stable dim resolution.
    # In test stubs, orig == current (no SSW translation needed).
    stub._orig_workload = MagicMock()
    stub._orig_workload.get_node_by_name.side_effect = KeyError("stub")  # force fallback to SSW workload
    stub._orig_workload.get_iteration_space_nodes.return_value = []
    stub._orig_mapping = MagicMock()
    stub._ssw_to_orig = {}  # identity mapping
    stub._orig_to_ssw = {}  # identity mapping
    stub._ssw_dim_to_orig = lambda d: d  # identity: no SSW translation in test stubs
    stub._orig_dim_to_ssw = lambda d: d
    stub._resolve_orig_tensor = lambda tr, fallback: fallback  # always use SSW tensor in stubs
    stub._orig_workload_dim_size = lambda d: stub._orig_workload.get_dimension_size(d)
    stub._translate_tiling_to_ssw = lambda tiling: tiling  # identity in test stubs
    stub._translate_tiling_to_orig = lambda tiling: tiling  # identity in test stubs
    stub._base_orig_dim_sizes = {}  # no iteration scaling in test stubs
    stub._iter_scale_by_jw = {}
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

    ss = _build_search_space(candidates_per_dim)
    stub = _make_allocator_stub(model, ss)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    # Mock workload
    workload = MagicMock()
    stub.workload = workload
    stub.mapping = MagicMock()

    # Successor is a ComputationNode (not InEdge/OutEdge)
    succ_node = MagicMock()
    succ_node.__class__ = MagicMock  # not InEdge/OutEdge
    # Make isinstance checks work: succ_node is NOT an InEdge or OutEdge
    workload.successors.return_value = [succ_node]

    # get_unique_dims_inter_core_tiling returns tiling for tiled dims
    base_tiling = tuple((_dim(p), 4) for p in tiled_dims_positions)
    workload.get_unique_dims_inter_core_tiling.return_value = base_tiling

    # get_dims returns the tiled dims (used by _tiled_dims_for_tensor)
    workload.get_dims.return_value = [_dim(p) for p in tiled_dims_positions]

    # get_dimension_size returns 256 for all dims
    workload.get_dimension_size.return_value = 256

    # get_tensor_shape_with_tiling returns a simple shape
    workload.get_tensor_shape_with_tiling.return_value = (64, 128)

    return stub, ss, succ_node


def test_joint_candidates_single_dim_returns_pairs(model):
    """For 1-dim SearchSpace, returns one (size, var) pair per candidate."""
    stub, ss, succ_node = _make_stub_for_joint_candidates(model, {0: [16, 32]}, [0])
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
    stub, ss, succ_node = _make_stub_for_joint_candidates(model, {0: [16, 32]}, [0])
    tensor = MagicMock()
    tensor.size_bits.return_value = 2048
    tr = MagicMock()
    tr.outputs = [tensor]

    stub._joint_candidates_for_tensor(tensor, tr)
    assert tensor in stub._tensor_max_size
    assert stub._tensor_max_size[tensor] == 2048


def test_joint_candidates_caches_result(model):
    """Calling _joint_candidates_for_tensor twice returns same list (cached)."""
    stub, ss, succ_node = _make_stub_for_joint_candidates(model, {0: [16, 32]}, [0])
    tensor = MagicMock()
    tensor.size_bits.return_value = 512
    tr = MagicMock()
    tr.outputs = [tensor]

    result1 = stub._joint_candidates_for_tensor(tensor, tr)
    result2 = stub._joint_candidates_for_tensor(tensor, tr)
    assert result1 is result2


def test_single_candidate_degenerate(model):
    """With 1 candidate per dim, exactly 1 joint combination exists (Pitfall 5 regression)."""
    stub, ss, succ_node = _make_stub_for_joint_candidates(model, {0: [16], 1: [32]}, [0, 1])
    tensor = MagicMock()
    tensor.size_bits.return_value = 512
    tr = MagicMock()
    tr.outputs = [tensor]

    results = stub._joint_candidates_for_tensor(tensor, tr)
    # 1 candidate per dim → 1 * 1 = 1 joint combination
    assert len(results) == 1


def test_joint_candidates_no_tiled_dims_returns_empty(model):
    """When no SearchSpace dims appear in the successor's tiling, returns empty list."""

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

    # Phase 7 attributes for _orig_workload resolution
    stub._orig_workload = MagicMock()
    stub._orig_workload.get_node_by_name.side_effect = KeyError("stub")
    stub._orig_workload.get_iteration_space_nodes.return_value = []
    stub._orig_mapping = MagicMock()
    stub._ssw_to_orig = {}
    stub._orig_to_ssw = {}
    stub._ssw_dim_to_orig = lambda d: d  # identity: no SSW translation in test stubs
    stub._orig_dim_to_ssw = lambda d: d
    stub._resolve_orig_tensor = lambda tr, fallback: fallback  # always use SSW tensor in stubs
    stub._orig_workload_dim_size = lambda d: stub._orig_workload.get_dimension_size(d)
    stub._translate_tiling_to_ssw = lambda tiling: tiling  # identity in test stubs
    stub._translate_tiling_to_orig = lambda tiling: tiling  # identity in test stubs
    stub._base_orig_dim_sizes = {}
    stub._iter_scale_by_jw = {}

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
    stub._memory_capacity_constraints = TransferAndTensorAllocator._memory_capacity_constraints.__get__(stub)

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


# ---------------------------------------------------------------------------
# Helpers for Task 2: _init_transfer_fire_helpers + _ssis_coefficients tests
# ---------------------------------------------------------------------------


def _build_ssis_stub(dim_sizes_relevancies: list[tuple, ...]):
    """Build a mock SSIS with get_applicable_temporal_variables returning mocks."""
    from stream.workload.steady_state.iteration_space import (
        IterationVariable,
        IterationVariableType,
        LoopEffect,
        SteadyStateIterationSpace,
    )

    variables = []
    for dim, size, relevant in dim_sizes_relevancies:
        effect = LoopEffect.VARYING if relevant else LoopEffect.INVARIANT
        iv = IterationVariable(dimension=dim, size=size, effect=effect, type=IterationVariableType.TEMPORAL)
        variables.append(iv)
    return SteadyStateIterationSpace(variables)


def _make_fire_helpers_stub(
    model: gp.Model,
    search_space,
    ssis_per_tr,
    tensors_per_tr: list[list],
):
    """Build a stub with _init_transfer_fire_helpers bound.

    Parameters
    ----------
    model : gp.Model
    search_space : SearchSpace | None
    ssis_per_tr : list of SteadyStateIterationSpace  (one per transfer node)
    tensors_per_tr : list of lists of Tensor mocks (one list per transfer node)
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

    # Build transfer node mocks
    trs = []
    for i, (ssis, tensors) in enumerate(zip(ssis_per_tr, tensors_per_tr, strict=False)):
        tr = MagicMock()
        tr.name = f"tr_{i}"
        tr.tensors = tensors
        trs.append(tr)
    stub.transfer_nodes = tuple(trs)
    stub.ssis = {tr: ssis for tr, ssis in zip(trs, ssis_per_tr, strict=False)}

    # Fire helpers state
    stub.reuse_levels = {}
    stub.tiles_needed_levels = {}
    stub.bds_needed_levels = {}
    stub.transfer_nodes_to_optimize_firings_for = []
    stub._ssis_max_coefficients = {}

    # Bind all required methods
    stub._ssis_tiled_dims_for_transfer = TransferAndTensorAllocator._ssis_tiled_dims_for_transfer.__get__(stub)
    stub._ssis_coefficients_for_transfer = TransferAndTensorAllocator._ssis_coefficients_for_transfer.__get__(stub)
    stub._joint_binary_for_combo = TransferAndTensorAllocator._joint_binary_for_combo.__get__(stub)
    stub._add_binary_product = TransferAndTensorAllocator._add_binary_product.__get__(stub)
    stub._safe_name = TransferAndTensorAllocator._safe_name.__get__(stub)
    stub._init_transfer_fire_helpers = TransferAndTensorAllocator._init_transfer_fire_helpers.__get__(stub)
    stub._classify_transfer_nodes_for_firing_optimization = (
        TransferAndTensorAllocator._classify_transfer_nodes_for_firing_optimization.__get__(stub)
    )
    # Phase 7: methods access _orig_workload/_orig_mapping for stable dim resolution.
    stub._orig_workload = MagicMock()
    stub._orig_workload.get_node_by_name.side_effect = KeyError("stub")
    stub._orig_workload.get_iteration_space_nodes.return_value = []
    stub._orig_mapping = MagicMock()
    stub._ssw_to_orig = {}
    stub._orig_to_ssw = {}
    stub._ssw_dim_to_orig = lambda d: d  # identity: no SSW translation in test stubs
    stub._orig_dim_to_ssw = lambda d: d
    stub._resolve_orig_tensor = lambda tr, fallback: fallback  # always use SSW tensor in stubs
    stub._orig_workload_dim_size = lambda d: stub._orig_workload.get_dimension_size(d)
    stub._translate_tiling_to_ssw = lambda tiling: tiling  # identity in test stubs
    stub._translate_tiling_to_orig = lambda tiling: tiling  # identity in test stubs
    stub._base_orig_dim_sizes = {}
    stub._iter_scale_by_jw = {}
    # Pre-populate transfer_nodes_to_optimize_firings_for (normally done in __init__)
    stub._classify_transfer_nodes_for_firing_optimization()
    return stub, trs


# ---------------------------------------------------------------------------
# Tests for _ssis_tiled_dims_for_transfer
# ---------------------------------------------------------------------------


def test_ssis_tiled_dims_for_transfer_no_search_space(model):
    """When search_space is None, returns empty list."""
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub = types.SimpleNamespace()
    stub.search_space = None

    # Bind method
    stub._ssis_tiled_dims_for_transfer = TransferAndTensorAllocator._ssis_tiled_dims_for_transfer.__get__(stub)

    tr = MagicMock()
    result = stub._ssis_tiled_dims_for_transfer(tr)
    assert result == []


def test_ssis_tiled_dims_for_transfer_intersection(model):
    """Returns only dims present in both search_space and SSIS temporal dims."""
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )
    from stream.workload.steady_state.iteration_space import (
        IterationVariable,
        IterationVariableType,
        LoopEffect,
        SteadyStateIterationSpace,
    )

    d0 = _dim(0)
    d1 = _dim(1)
    d99 = _dim(99)  # in search_space but NOT in SSIS

    ss = _build_search_space({0: [16, 32], 99: [4, 8]})

    stub = types.SimpleNamespace()
    stub.search_space = ss

    # SSIS with d0 and d1 but NOT d99
    ssis = SteadyStateIterationSpace(
        [
            IterationVariable(d0, 16, LoopEffect.VARYING, IterationVariableType.TEMPORAL),
            IterationVariable(d1, 8, LoopEffect.VARYING, IterationVariableType.TEMPORAL),
        ]
    )
    tr = MagicMock()
    stub.ssis = {tr: ssis}

    stub._ssis_tiled_dims_for_transfer = TransferAndTensorAllocator._ssis_tiled_dims_for_transfer.__get__(stub)
    result = stub._ssis_tiled_dims_for_transfer(tr)
    # Only d0 is in both search_space (pos=0) and SSIS temporal dims
    assert d0 in result
    assert d99 not in result


# ---------------------------------------------------------------------------
# Tests for _init_transfer_fire_helpers scalar fallback (search_space=None)
# ---------------------------------------------------------------------------


def test_init_fire_helpers_scalar_fallback(model):
    """search_space=None: reuse_levels[(t,-1)] is a plain (int, int) tuple."""
    d0 = _dim(0)
    ssis = _build_ssis_stub([(d0, 4, True)])
    tensor = MagicMock()
    tensor.name = "t0"
    stub, trs = _make_fire_helpers_stub(model, None, [ssis], [[tensor]])

    stub._init_transfer_fire_helpers()

    # Scalar path: reuse_levels[(tensor, -1)] == (fires, size_factor)
    assert (tensor, -1) in stub.reuse_levels
    val = stub.reuse_levels[(tensor, -1)]
    assert isinstance(val, tuple)
    assert len(val) == 2
    fires, sf = val
    assert isinstance(fires, int)
    assert isinstance(sf, int)
    # With 1 temporal loop of size 4 (relevant): fires=4, sf=1 at stop=-1
    assert fires == 4
    assert sf == 1


def test_init_fire_helpers_scalar_tiles_needed(model):
    """search_space=None: tiles_needed_levels[(t,stop)] is a plain int."""
    d0 = _dim(0)
    ssis = _build_ssis_stub([(d0, 4, True)])
    tensor = MagicMock()
    tensor.name = "t0"
    stub, trs = _make_fire_helpers_stub(model, None, [ssis], [[tensor]])

    stub._init_transfer_fire_helpers()

    assert (tensor, -1) in stub.tiles_needed_levels
    assert isinstance(stub.tiles_needed_levels[(tensor, -1)], int)


def test_init_fire_helpers_variable_tile_coefficients(model):
    """search_space with 2 candidates: reuse_levels[(t,-1)] is a list of (fires,sf,jw) tuples."""
    d0 = _dim(0)
    ss = _build_search_space({0: [16, 32]})

    # Create w vars
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    # SSIS for this transfer: d0 with size 16 (will be replaced by candidate sizes)
    ssis = _build_ssis_stub([(d0, 16, True)])

    tensor = MagicMock()
    tensor.name = "t0"
    stub, trs = _make_fire_helpers_stub(model, ss, [ssis], [[tensor]])

    # Create tile selection vars
    stub._TransferAndTensorAllocator__create_tile_selection_vars = (
        TransferAndTensorAllocator._TransferAndTensorAllocator__create_tile_selection_vars.__get__(stub)
    )
    stub._tiled_dims_for_tensor = TransferAndTensorAllocator._tiled_dims_for_tensor.__get__(stub)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    # We need workload for the ssis_tiled_dims check
    stub.workload = MagicMock()
    stub.workload.get_dimension_size.return_value = 256  # workload_size for d0

    stub._init_transfer_fire_helpers()
    model.update()

    # With 2 candidates, reuse_levels[(tensor,-1)] should be a list with 2 entries
    assert (tensor, -1) in stub.reuse_levels
    val = stub.reuse_levels[(tensor, -1)]
    assert isinstance(val, list), f"Expected list, got {type(val)}"
    assert len(val) == 2
    # Each entry: (fires_coeff, size_factor_coeff, joint_binary_var)
    for fires_c, sf_c, jw in val:
        assert isinstance(fires_c, int)
        assert isinstance(sf_c, int)
        assert hasattr(jw, "VType")  # is a gp.Var


def test_init_fire_helpers_degenerate_single_candidate(model):
    """search_space with 1 candidate: coefficient lists have exactly 1 entry."""
    d0 = _dim(0)
    ss = _build_search_space({0: [16]})

    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    ssis = _build_ssis_stub([(d0, 16, True)])
    tensor = MagicMock()
    tensor.name = "t0"
    stub, trs = _make_fire_helpers_stub(model, ss, [ssis], [[tensor]])

    stub._TransferAndTensorAllocator__create_tile_selection_vars = (
        TransferAndTensorAllocator._TransferAndTensorAllocator__create_tile_selection_vars.__get__(stub)
    )
    stub._tiled_dims_for_tensor = TransferAndTensorAllocator._tiled_dims_for_tensor.__get__(stub)
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    stub.workload = MagicMock()
    stub.workload.get_dimension_size.return_value = 256

    stub._init_transfer_fire_helpers()
    model.update()

    assert (tensor, -1) in stub.reuse_levels
    val = stub.reuse_levels[(tensor, -1)]
    assert isinstance(val, list)
    assert len(val) == 1


def test_ensure_same_ssis_not_called_at_init():
    """_ensure_same_ssis_for_all_transfers is NOT called in __init__ block.

    Structural test: the __init__ source must not invoke _ensure_same_ssis_for_all_transfers.
    """
    import inspect

    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    source = inspect.getsource(TransferAndTensorAllocator.__init__)
    # The call _ensure_same_ssis_for_all_transfers() must not appear in __init__
    assert "_ensure_same_ssis_for_all_transfers()" not in source, (
        "_ensure_same_ssis_for_all_transfers() is still called in __init__ (should be post-solve only)"
    )


# ---------------------------------------------------------------------------
# Helpers for Task 1: _transfer_fire_rate_constraints and
# _reuse_factor_rate_constraints tests
# ---------------------------------------------------------------------------


def _make_fire_rate_stub(model: gp.Model, n_stops: int = 1):
    """Build a minimal stub for fire rate / reuse factor constraint tests.

    Pre-wires:
    - A single transfer node with one input tensor
    - scalar reuse_levels (search_space=None baseline)
    - z_stop binary vars for (tensor, stop) for stop in range(-1, n_stops)
    - SSIS mock returning n_stops temporal variables
    - fires dict (empty, to be populated by method)
    - reuse_factors dict
    """
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )
    from stream.workload.workload import Tensor

    stub = types.SimpleNamespace()
    stub.model = model

    # Single transfer node with one input tensor
    tensor = MagicMock(spec=Tensor)
    tensor.name = "tr_tensor"
    tr = MagicMock()
    tr.name = "tr0"
    tr.inputs = [tensor]
    stub.transfer_nodes = [tr]

    # SSIS mock
    ssis_mock = MagicMock()
    temporal_vars = [MagicMock()] * n_stops
    ssis_mock.get_applicable_temporal_variables.return_value = temporal_vars
    stub.ssis = {tr: ssis_mock}

    # z_stop binary variables
    stub.z_stop = {}
    for s in range(-1, n_stops):
        z = model.addVar(vtype=GRB.BINARY, name=f"z_{s}")
        stub.z_stop[(tensor, s)] = z
    model.update()

    # Scalar reuse_levels: (fires_coeff, size_factor_coeff)
    stub.reuse_levels = {}
    for s in range(-1, n_stops):
        stub.reuse_levels[(tensor, s)] = (3, 2)  # fires=3, sf=2

    # Bind constraint methods
    stub.fires = {}
    stub.reuse_factors = {}
    stub._transfer_fire_rate_constraints = TransferAndTensorAllocator._transfer_fire_rate_constraints.__get__(stub)
    stub._reuse_factor_rate_constraints = TransferAndTensorAllocator._reuse_factor_rate_constraints.__get__(stub)
    return stub, tr, tensor


def _make_fire_rate_stub_variable(model: gp.Model, n_candidates: int = 2, n_stops: int = 1):
    """Build stub with candidate-indexed (variable tile) reuse_levels.

    reuse_levels[(tensor, stop)] = list of (fires_k, sf_k, jw_k) tuples
    """
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )
    from stream.workload.workload import Tensor

    stub = types.SimpleNamespace()
    stub.model = model

    tensor = MagicMock(spec=Tensor)
    tensor.name = "tr_tensor_var"
    tr = MagicMock()
    tr.name = "tr_var"
    tr.inputs = [tensor]
    stub.transfer_nodes = [tr]

    ssis_mock = MagicMock()
    temporal_vars = [MagicMock()] * n_stops
    ssis_mock.get_applicable_temporal_variables.return_value = temporal_vars
    stub.ssis = {tr: ssis_mock}

    # z_stop binary variables
    stub.z_stop = {}
    for s in range(-1, n_stops):
        z = model.addVar(vtype=GRB.BINARY, name=f"z_var_{s}")
        stub.z_stop[(tensor, s)] = z

    # joint binary variables (one per candidate)
    jw_vars = [model.addVar(vtype=GRB.BINARY, name=f"jw_{k}") for k in range(n_candidates)]
    model.update()

    # candidate coefficients: fires_k = (k+1)*4, sf_k = (k+1)*2
    stub.reuse_levels = {}
    stub._ssis_max_coefficients = {}
    for s in range(-1, n_stops):
        stub.reuse_levels[(tensor, s)] = [((k + 1) * 4, (k + 1) * 2, jw_vars[k]) for k in range(n_candidates)]
        max_fires = n_candidates * 4
        max_sf = n_candidates * 2
        stub._ssis_max_coefficients[(tensor, s)] = {
            "fires": max_fires,
            "size_factor": max_sf,
        }

    # Bind constraint methods
    stub.fires = {}
    stub.reuse_factors = {}
    stub._transfer_fire_rate_constraints = TransferAndTensorAllocator._transfer_fire_rate_constraints.__get__(stub)
    stub._reuse_factor_rate_constraints = TransferAndTensorAllocator._reuse_factor_rate_constraints.__get__(stub)
    return stub, tr, tensor, jw_vars


# ---------------------------------------------------------------------------
# Tests for _transfer_fire_rate_constraints (Task 1 RED)
# ---------------------------------------------------------------------------


def test_fire_rate_scalar_creates_fires_def_constraint(model):
    """With scalar reuse_levels, fires_def_{tr.name} constraint exists (unchanged behavior)."""
    stub, tr, tensor = _make_fire_rate_stub(model, n_stops=1)
    stub._transfer_fire_rate_constraints()
    model.update()

    constr_names = {c.ConstrName for c in model.getConstrs()}
    assert f"fires_def_{tr.name}" in constr_names, f"Expected fires_def_{tr.name}, got: {constr_names}"


def test_fire_rate_scalar_no_lc_vars(model):
    """With scalar reuse_levels, no fire_lc_* auxiliary variables are created."""
    stub, tr, tensor = _make_fire_rate_stub(model, n_stops=1)
    stub._transfer_fire_rate_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("fire_lc_")]
    assert len(lc_vars) == 0, f"Expected no fire_lc_ vars in scalar mode, got: {lc_vars}"


def test_fire_rate_variable_creates_lc_vars(model):
    """With candidate-indexed reuse_levels (2 candidates), fire_lc_ vars are created."""
    stub, tr, tensor, jw_vars = _make_fire_rate_stub_variable(model, n_candidates=2, n_stops=1)
    stub._transfer_fire_rate_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("fire_lc_")]
    assert len(lc_vars) >= 1, f"Expected fire_lc_ vars in variable mode, got: {var_names}"


def test_fire_rate_variable_creates_bigm_constraints(model):
    """With candidate-indexed reuse_levels, fire_lc_ub_expr_, fire_lc_ub_m_, fire_lc_lb_ constraints exist."""
    stub, tr, tensor, jw_vars = _make_fire_rate_stub_variable(model, n_candidates=2, n_stops=1)
    stub._transfer_fire_rate_constraints()
    model.update()

    constr_names = {c.ConstrName for c in model.getConstrs()}
    lc_ub_expr = [n for n in constr_names if n.startswith("fire_lc_ub_expr_")]
    lc_ub_m = [n for n in constr_names if n.startswith("fire_lc_ub_m_")]
    lc_lb = [n for n in constr_names if n.startswith("fire_lc_lb_")]

    assert len(lc_ub_expr) >= 1, f"Missing fire_lc_ub_expr_ constraints: {constr_names}"
    assert len(lc_ub_m) >= 1, f"Missing fire_lc_ub_m_ constraints: {constr_names}"
    assert len(lc_lb) >= 1, f"Missing fire_lc_lb_ constraints: {constr_names}"


# ---------------------------------------------------------------------------
# Tests for _reuse_factor_rate_constraints (Task 1 RED)
# ---------------------------------------------------------------------------


def test_reuse_factor_scalar_creates_reuse_factor_def_constraint(model):
    """With scalar reuse_levels, reuse_factor_def_{tr.name} constraint exists (unchanged)."""
    stub, tr, tensor = _make_fire_rate_stub(model, n_stops=1)
    stub._reuse_factor_rate_constraints()
    model.update()

    constr_names = {c.ConstrName for c in model.getConstrs()}
    assert f"reuse_factor_def_{tr.name}" in constr_names, f"Expected reuse_factor_def_{tr.name}, got: {constr_names}"


def test_reuse_factor_scalar_no_lc_vars(model):
    """With scalar reuse_levels, no rf_lc_* auxiliary variables are created."""
    stub, tr, tensor = _make_fire_rate_stub(model, n_stops=1)
    stub._reuse_factor_rate_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("rf_lc_")]
    assert len(lc_vars) == 0, f"Expected no rf_lc_ vars in scalar mode, got: {lc_vars}"


def test_reuse_factor_variable_creates_lc_vars(model):
    """With candidate-indexed reuse_levels (2 candidates), rf_lc_ vars are created."""
    stub, tr, tensor, jw_vars = _make_fire_rate_stub_variable(model, n_candidates=2, n_stops=1)
    stub._reuse_factor_rate_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("rf_lc_")]
    assert len(lc_vars) >= 1, f"Expected rf_lc_ vars in variable mode, got: {var_names}"


def test_reuse_factor_variable_creates_bigm_constraints(model):
    """With candidate-indexed reuse_levels, rf_lc_ub_expr_, rf_lc_ub_m_, rf_lc_lb_ constraints exist."""
    stub, tr, tensor, jw_vars = _make_fire_rate_stub_variable(model, n_candidates=2, n_stops=1)
    stub._reuse_factor_rate_constraints()
    model.update()

    constr_names = {c.ConstrName for c in model.getConstrs()}
    lc_ub_expr = [n for n in constr_names if n.startswith("rf_lc_ub_expr_")]
    lc_ub_m = [n for n in constr_names if n.startswith("rf_lc_ub_m_")]
    lc_lb = [n for n in constr_names if n.startswith("rf_lc_lb_")]

    assert len(lc_ub_expr) >= 1, f"Missing rf_lc_ub_expr_ constraints: {constr_names}"
    assert len(lc_ub_m) >= 1, f"Missing rf_lc_ub_m_ constraints: {constr_names}"
    assert len(lc_lb) >= 1, f"Missing rf_lc_lb_ constraints: {constr_names}"


# ---------------------------------------------------------------------------
# Helpers for Task 2: _object_fifo_depth_constraints and
# _buffer_descriptor_constraints tests
# ---------------------------------------------------------------------------


def _make_fifo_bd_stub(
    model: gp.Model,
    n_stops: int = 1,
    tiles_needed_scalar: int = 2,
    bds_needed_scalar: int = 3,
    n_candidates: int = 0,  # 0 = scalar mode
    force_double_buffering: bool = False,
    core_type: str = "compute",
):
    """Build a minimal stub for fifo depth / BD constraint tests.

    When n_candidates > 0: tiles_needed_levels / bds_needed_levels are lists of (coeff, jw) tuples.
    When n_candidates == 0: they are plain ints (scalar mode).
    """
    from collections import defaultdict

    from stream.hardware.architecture.core import Core
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )
    from stream.workload.workload import Tensor

    stub = types.SimpleNamespace()
    stub.model = model
    stub.offchip_core_id = 999  # different from core.id below
    stub.force_double_buffering = force_double_buffering

    # Build a Core mock
    core = MagicMock(spec=Core)
    core.id = 1
    core.type = core_type
    stub._core = core

    # Build a Tensor mock
    tensor = MagicMock(spec=Tensor)
    tensor.name = "fifo_tensor"
    stub._tensor = tensor

    # Transfer node mock with outputs
    tr = MagicMock()
    tr.name = "tr_fifo"
    tr.outputs = [tensor]
    stub.transfer_nodes = [tr]

    # SSIS mock
    ssis_mock = MagicMock()
    temporal_vars = [MagicMock()] * n_stops
    ssis_mock.get_applicable_temporal_variables.return_value = temporal_vars
    stub.ssis = {tr: ssis_mock}

    # z_stop binary variables
    stub.z_stop = {}
    for s in range(-1, n_stops):
        z = model.addVar(vtype=GRB.BINARY, name=f"z_fifo_{s}")
        stub.z_stop[(tensor, s)] = z

    # u (tensor uses core) binary var — fix to 1 so constraints are active
    u = model.addVar(vtype=GRB.BINARY, name="u_fifo_core")
    model.addConstr(u == 1, name="u_fifo_fixed")
    model.update()

    stub.tensor_core_indicator = {}
    stub._tensor_uses_core_var = lambda t, c: u
    stub._u = u

    stub._candidate_cores_for_tensor = lambda t: {core}

    # Build coefficient levels
    stub._ssis_max_coefficients = {}
    if n_candidates > 0:
        # Variable tile mode: list of (coeff, jw) tuples
        jw_vars = [model.addVar(vtype=GRB.BINARY, name=f"jw_fifo_{k}") for k in range(n_candidates)]
        model.update()
        stub.tiles_needed_levels = {}
        stub.bds_needed_levels = {}
        for s in range(-1, n_stops):
            stub.tiles_needed_levels[(tensor, s)] = [
                ((k + 1) * tiles_needed_scalar, jw_vars[k]) for k in range(n_candidates)
            ]
            stub.bds_needed_levels[(tensor, s)] = [
                ((k + 1) * bds_needed_scalar, jw_vars[k]) for k in range(n_candidates)
            ]
            stub._ssis_max_coefficients[(tensor, s)] = {
                "tiles_needed": n_candidates * tiles_needed_scalar,
                "bds_needed": n_candidates * bds_needed_scalar,
            }
        stub._jw_vars = jw_vars
    else:
        # Scalar mode
        stub.tiles_needed_levels = {}
        stub.bds_needed_levels = {}
        for s in range(-1, n_stops):
            stub.tiles_needed_levels[(tensor, s)] = tiles_needed_scalar
            stub.bds_needed_levels[(tensor, s)] = bds_needed_scalar

    # Context mock — just captures calls
    stub.context = MagicMock()

    # Bind helpers
    stub._add_binary_product = TransferAndTensorAllocator._add_binary_product.__get__(stub)
    stub._safe_name = TransferAndTensorAllocator._safe_name.__get__(stub)

    # Bind constraint methods
    stub.object_fifo_depth = defaultdict(gp.LinExpr)
    stub.bd_depth = defaultdict(gp.LinExpr)
    stub._object_fifo_depth_constraints = TransferAndTensorAllocator._object_fifo_depth_constraints.__get__(stub)
    stub._buffer_descriptor_constraints = TransferAndTensorAllocator._buffer_descriptor_constraints.__get__(stub)
    return stub, tr, tensor, core


# ---------------------------------------------------------------------------
# Tests for _object_fifo_depth_constraints (Task 2 RED)
# ---------------------------------------------------------------------------


def test_fifo_depth_scalar_no_lc_vars(model):
    """With scalar tiles_needed_levels, no fifo_lc_* auxiliary variables are created."""
    stub, tr, tensor, core = _make_fifo_bd_stub(model, n_stops=1, tiles_needed_scalar=2)
    stub._object_fifo_depth_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("fifo_lc_")]
    assert len(lc_vars) == 0, f"Expected no fifo_lc_ vars in scalar mode, got: {lc_vars}"


def test_fifo_depth_variable_creates_lc_vars(model):
    """With candidate-indexed tiles_needed_levels (2 candidates), fifo_lc_ vars are created."""
    stub, tr, tensor, core = _make_fifo_bd_stub(model, n_stops=1, n_candidates=2)
    stub._object_fifo_depth_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("fifo_lc_")]
    assert len(lc_vars) >= 1, f"Expected fifo_lc_ vars in variable mode, got: {var_names}"


def test_fifo_depth_variable_creates_bigm_constraints(model):
    """With candidate-indexed tiles_needed_levels, fifo_lc_ big-M constraints exist."""
    stub, tr, tensor, core = _make_fifo_bd_stub(model, n_stops=1, n_candidates=2)
    stub._object_fifo_depth_constraints()
    model.update()

    constr_names = {c.ConstrName for c in model.getConstrs()}
    lc_ub_expr = [n for n in constr_names if n.startswith("fifo_lc_ub_expr_")]
    lc_ub_m = [n for n in constr_names if n.startswith("fifo_lc_ub_m_")]
    lc_lb = [n for n in constr_names if n.startswith("fifo_lc_lb_")]

    assert len(lc_ub_expr) >= 1, f"Missing fifo_lc_ub_expr_ constraints: {constr_names}"
    assert len(lc_ub_m) >= 1, f"Missing fifo_lc_ub_m_ constraints: {constr_names}"
    assert len(lc_lb) >= 1, f"Missing fifo_lc_lb_ constraints: {constr_names}"


def test_fifo_depth_double_buffering_increases_m(model):
    """force_double_buffering=True adds +1 to M bound, reflected in lc var upper bound."""
    stub_no_db, _, _, _ = _make_fifo_bd_stub(
        model, n_stops=1, n_candidates=2, tiles_needed_scalar=2, force_double_buffering=False
    )
    stub_no_db._object_fifo_depth_constraints()
    model.update()
    lc_no_db = [v for v in model.getVars() if v.VarName.startswith("fifo_lc_")]
    ub_no_db = max(v.UB for v in lc_no_db) if lc_no_db else None

    model2 = gp.Model()
    model2.setParam("OutputFlag", 0)
    stub_db, _, _, _ = _make_fifo_bd_stub(
        model2, n_stops=1, n_candidates=2, tiles_needed_scalar=2, force_double_buffering=True
    )
    stub_db._object_fifo_depth_constraints()
    model2.update()
    lc_db = [v for v in model2.getVars() if v.VarName.startswith("fifo_lc_")]
    ub_db = max(v.UB for v in lc_db) if lc_db else None

    assert ub_no_db is not None and ub_db is not None
    assert ub_db == ub_no_db + 1, f"Expected double-buffering M = no-db M + 1, got {ub_db} vs {ub_no_db}"


# ---------------------------------------------------------------------------
# Tests for _buffer_descriptor_constraints (Task 2 RED)
# ---------------------------------------------------------------------------


def test_bd_depth_scalar_no_lc_vars(model):
    """With scalar bds_needed_levels, no bd_lc_* auxiliary variables are created."""
    stub, tr, tensor, core = _make_fifo_bd_stub(model, n_stops=1, bds_needed_scalar=3, core_type="memory")
    stub._buffer_descriptor_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("bd_lc_")]
    assert len(lc_vars) == 0, f"Expected no bd_lc_ vars in scalar mode, got: {lc_vars}"


def test_bd_depth_variable_creates_lc_vars(model):
    """With candidate-indexed bds_needed_levels (2 candidates), bd_lc_ vars are created."""
    stub, tr, tensor, core = _make_fifo_bd_stub(model, n_stops=1, n_candidates=2, core_type="memory")
    stub._buffer_descriptor_constraints()
    model.update()

    var_names = {v.VarName for v in model.getVars()}
    lc_vars = [n for n in var_names if n.startswith("bd_lc_")]
    assert len(lc_vars) >= 1, f"Expected bd_lc_ vars in variable mode, got: {var_names}"


def test_bd_depth_variable_creates_bigm_constraints(model):
    """With candidate-indexed bds_needed_levels, bd_lc_ big-M constraints exist."""
    stub, tr, tensor, core = _make_fifo_bd_stub(model, n_stops=1, n_candidates=2, core_type="memory")
    stub._buffer_descriptor_constraints()
    model.update()

    constr_names = {c.ConstrName for c in model.getConstrs()}
    lc_ub_expr = [n for n in constr_names if n.startswith("bd_lc_ub_expr_")]
    lc_ub_m = [n for n in constr_names if n.startswith("bd_lc_ub_m_")]
    lc_lb = [n for n in constr_names if n.startswith("bd_lc_lb_")]

    assert len(lc_ub_expr) >= 1, f"Missing bd_lc_ub_expr_ constraints: {constr_names}"
    assert len(lc_ub_m) >= 1, f"Missing bd_lc_ub_m_ constraints: {constr_names}"
    assert len(lc_lb) >= 1, f"Missing bd_lc_lb_ constraints: {constr_names}"


# ---------------------------------------------------------------------------
# Degenerate single-candidate regression test (Task 2 RED)
# ---------------------------------------------------------------------------


def test_degenerate_ssis_single_candidate_feasible(model):
    """Single-candidate search space: model is feasible with correct fire/reuse values.

    Verifies that linearized constraints with 1 candidate degenerate correctly
    to the equivalent scalar baseline.
    """
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )
    from stream.workload.workload import Tensor

    # One-stop SSIS, one candidate, fires=4 sf=1, tiles_needed=2, bds_needed=3
    fires_k = 4
    sf_k = 1
    tn_k = 2
    bn_k = 3

    jw = model.addVar(vtype=GRB.BINARY, name="jw_deg")
    # One-hot: jw must be 1
    model.addConstr(jw == 1, name="jw_onehot")

    z = model.addVar(vtype=GRB.BINARY, name="z_deg")
    # Only one stop level (-1), force it to be selected
    model.addConstr(z == 1, name="z_selected")
    model.update()

    # Build stub manually for fire/reuse rate + fifo + bd
    stub = types.SimpleNamespace()
    stub.model = model

    from stream.hardware.architecture.core import Core

    tensor = MagicMock(spec=Tensor)
    tensor.name = "deg_tensor"
    tr = MagicMock()
    tr.name = "tr_deg"
    tr.inputs = [tensor]
    tr.outputs = [tensor]
    stub.transfer_nodes = [tr]

    ssis_mock = MagicMock()
    ssis_mock.get_applicable_temporal_variables.return_value = []  # n_stops=0, only stop=-1
    stub.ssis = {tr: ssis_mock}

    stub.z_stop = {(tensor, -1): z}

    # Variable tile mode: single candidate
    stub.reuse_levels = {(tensor, -1): [(fires_k, sf_k, jw)]}
    stub.tiles_needed_levels = {(tensor, -1): [(tn_k, jw)]}
    stub.bds_needed_levels = {(tensor, -1): [(bn_k, jw)]}
    stub._ssis_max_coefficients = {
        (tensor, -1): {
            "fires": fires_k,
            "size_factor": sf_k,
            "tiles_needed": tn_k,
            "bds_needed": bn_k,
        }
    }

    stub.offchip_core_id = 999
    stub.force_double_buffering = False
    stub.context = MagicMock()

    core = MagicMock(spec=Core)
    core.id = 1
    core.type = "memory"

    u = model.addVar(vtype=GRB.BINARY, name="u_deg")
    model.addConstr(u == 1, name="u_deg_fixed")
    model.update()
    stub.tensor_core_indicator = {}
    stub._tensor_uses_core_var = lambda t, c: u
    stub._candidate_cores_for_tensor = lambda t: {core}

    from collections import defaultdict

    stub.fires = {}
    stub.reuse_factors = {}
    stub.object_fifo_depth = defaultdict(gp.LinExpr)
    stub.bd_depth = defaultdict(gp.LinExpr)

    # Bind all methods
    stub._add_binary_product = TransferAndTensorAllocator._add_binary_product.__get__(stub)
    stub._safe_name = TransferAndTensorAllocator._safe_name.__get__(stub)
    stub._transfer_fire_rate_constraints = TransferAndTensorAllocator._transfer_fire_rate_constraints.__get__(stub)
    stub._reuse_factor_rate_constraints = TransferAndTensorAllocator._reuse_factor_rate_constraints.__get__(stub)
    stub._object_fifo_depth_constraints = TransferAndTensorAllocator._object_fifo_depth_constraints.__get__(stub)
    stub._buffer_descriptor_constraints = TransferAndTensorAllocator._buffer_descriptor_constraints.__get__(stub)

    # Execute all constraint methods
    stub._transfer_fire_rate_constraints()
    stub._reuse_factor_rate_constraints()
    stub._object_fifo_depth_constraints()
    stub._buffer_descriptor_constraints()
    model.update()

    # Add objective (minimize nothing — just feasibility)
    model.setObjective(0)
    model.optimize()

    assert model.Status == GRB.OPTIMAL, f"Model infeasible, status={model.Status}"

    # With jw=1 and z=1: fires should equal fires_k=4, reuse_factor should equal sf_k=1
    fires_var = stub.fires[tr]
    rf_var = stub.reuse_factors[tr]
    assert abs(fires_var.X - fires_k) < 1e-6, f"Expected fires={fires_k}, got {fires_var.X}"
    assert abs(rf_var.X - sf_k) < 1e-6, f"Expected reuse_factor={sf_k}, got {rf_var.X}"


# ---------------------------------------------------------------------------
# Phase 5 — Pure MILP transfer latency tests
# ---------------------------------------------------------------------------


def _make_latency_stub(model: gp.Model):
    """Build a minimal stub for _active_transfer_latency testing."""
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub = types.SimpleNamespace()
    stub.model = model
    stub._transfer_latency_cache = {}
    stub.z_stop = {}
    stub.reuse_levels = {}
    stub.ssis = {}
    stub.reuse_factors = {}
    stub.y_path_choice = {}
    stub._ssis_max_coefficients = {}
    stub._tensor_max_size = {}
    stub._tensor_joint_candidates = {}

    stub._add_binary_product = TransferAndTensorAllocator._add_binary_product.__get__(stub)
    stub._add_binary_scaled_continuous = TransferAndTensorAllocator._add_binary_scaled_continuous.__get__(stub)
    stub._safe_name = TransferAndTensorAllocator._safe_name.__get__(stub)
    stub._active_transfer_latency = TransferAndTensorAllocator._active_transfer_latency.__get__(stub)
    stub._iter_scale_by_jw = {}  # Phase 7: iteration scaling for transfer latency

    return stub


def test_active_transfer_latency_variable_mode(model):
    """Variable tile mode with 2 candidates: latency var == amortized_latency for selected candidate.

    Setup:
    - 2 joint candidates: size_bits=[1024, 2048], min_bw=256
    - latency_numerator = [ceil(1024/256)=4, ceil(2048/256)=8]
    - sf_coeff = [1, 2], so amortized = [4/1=4.0, 8/2=4.0]
    - Force jw0=1, z_stop=1, y=1
    - Expected: latency_var.X == 4.0
    """
    from math import ceil

    from stream.workload.workload import Tensor

    stub = _make_latency_stub(model)

    tensor = MagicMock(spec=Tensor)
    tensor.name = "lat_tensor_var"
    tr = MagicMock()
    tr.name = "tr_lat_var"
    tr.inputs = [tensor]

    # 2 joint candidates
    jw0 = model.addVar(vtype=GRB.BINARY, name="jw_lat_0")
    jw1 = model.addVar(vtype=GRB.BINARY, name="jw_lat_1")
    # One-hot: jw0 + jw1 = 1, force jw0=1
    model.addConstr(jw0 + jw1 == 1, name="jw_lat_onehot")
    model.addConstr(jw0 == 1, name="jw_lat_force0")

    # z_stop for stop level -1
    z = model.addVar(vtype=GRB.BINARY, name="z_lat")
    model.addConstr(z == 1, name="z_lat_fixed")

    # path choice y
    y = model.addVar(vtype=GRB.BINARY, name="y_lat")
    model.addConstr(y == 1, name="y_lat_fixed")

    model.update()

    # Stub data
    # reuse_levels[(tensor, -1)] = list of (fires_c, sf_c, jw) triples
    stub.reuse_levels[(tensor, -1)] = [(1, 1, jw0), (1, 2, jw1)]
    stub.z_stop[(tensor, -1)] = z

    ssis_mock = MagicMock()
    ssis_mock.get_applicable_temporal_variables.return_value = []  # only stop=-1
    stub.ssis[tr] = ssis_mock

    # Mock _joint_candidates_for_tensor to return [(1024, jw0), (2048, jw1)]
    stub._joint_candidates_for_tensor = lambda t, _tr: [(1024, jw0), (2048, jw1)]

    # Mock choice with min_bw=256
    link = MagicMock()
    link.bandwidth = 256
    choice = MagicMock()
    choice.links_used = [link]

    # Expected latency: amortized[0] = ceil(1024/256)/1 = 4/1 = 4.0
    expected_latency = ceil(1024 / 256) / 1  # 4.0

    active_latency = stub._active_transfer_latency(tr, choice, y)

    model.update()
    model.setObjective(0)
    model.optimize()

    assert model.Status == GRB.OPTIMAL, f"Model infeasible, status={model.Status}"
    assert abs(active_latency.X - expected_latency) < 1e-6, (
        f"Expected latency={expected_latency}, got {active_latency.X}"
    )


def test_active_transfer_latency_degenerate(model):
    """Degenerate case: 1 candidate, latency == ceil(size_bits/min_bw)/sf_coeff.

    Setup:
    - 1 joint candidate: size_bits=512, min_bw=128
    - latency_numerator = ceil(512/128) = 4
    - sf_coeff = 1, so amortized = 4.0
    - Force jw=1, z_stop=1, y=1
    - Expected: latency_var.X == 4.0
    """
    from math import ceil

    from stream.workload.workload import Tensor

    stub = _make_latency_stub(model)

    tensor = MagicMock(spec=Tensor)
    tensor.name = "lat_tensor_deg"
    tr = MagicMock()
    tr.name = "tr_lat_deg"
    tr.inputs = [tensor]

    jw = model.addVar(vtype=GRB.BINARY, name="jw_lat_deg")
    model.addConstr(jw == 1, name="jw_lat_deg_fixed")

    z = model.addVar(vtype=GRB.BINARY, name="z_lat_deg")
    model.addConstr(z == 1, name="z_lat_deg_fixed")

    y = model.addVar(vtype=GRB.BINARY, name="y_lat_deg")
    model.addConstr(y == 1, name="y_lat_deg_fixed")

    model.update()

    stub.reuse_levels[(tensor, -1)] = [(1, 1, jw)]
    stub.z_stop[(tensor, -1)] = z

    ssis_mock = MagicMock()
    ssis_mock.get_applicable_temporal_variables.return_value = []
    stub.ssis[tr] = ssis_mock

    stub._joint_candidates_for_tensor = lambda t, _tr: [(512, jw)]

    link = MagicMock()
    link.bandwidth = 128
    choice = MagicMock()
    choice.links_used = [link]

    expected_latency = ceil(512 / 128) / 1  # 4.0

    active_latency = stub._active_transfer_latency(tr, choice, y)

    model.update()
    model.setObjective(0)
    model.optimize()

    assert model.Status == GRB.OPTIMAL, f"Model infeasible, status={model.Status}"
    assert abs(active_latency.X - expected_latency) < 1e-6, (
        f"Expected latency={expected_latency}, got {active_latency.X}"
    )


def test_active_transfer_latency_scalar_fallback(model):
    """Scalar fallback: no tiled dims, latency is computed correctly without addGenConstrNL.

    Setup:
    - _joint_candidates_for_tensor returns [] (no tiled dims)
    - reuse_levels[(tensor, -1)] = (1, 1) plain tuple (scalar mode)
    - _transfer_latency_for_path returns 4
    - Force y=1
    - Expected: latency_var.X == 4.0, model.NumGenConstrs == 0
    """
    from stream.workload.workload import Tensor

    stub = _make_latency_stub(model)

    tensor = MagicMock(spec=Tensor)
    tensor.name = "lat_tensor_scalar"
    tr = MagicMock()
    tr.name = "tr_lat_scalar"
    tr.inputs = [tensor]

    z = model.addVar(vtype=GRB.BINARY, name="z_lat_scalar")
    model.addConstr(z == 1, name="z_lat_scalar_fixed")

    y = model.addVar(vtype=GRB.BINARY, name="y_lat_scalar")
    model.addConstr(y == 1, name="y_lat_scalar_fixed")

    reuse_factor_var = model.addVar(vtype=GRB.INTEGER, lb=1, ub=1, name="rf_scalar")
    model.addConstr(reuse_factor_var == 1, name="rf_scalar_fixed")

    model.update()

    # Scalar mode: plain (fires, sf) tuple
    stub.reuse_levels[(tensor, -1)] = (1, 1)
    stub.z_stop[(tensor, -1)] = z
    stub.reuse_factors[tr] = reuse_factor_var

    ssis_mock = MagicMock()
    ssis_mock.get_applicable_temporal_variables.return_value = []
    stub.ssis[tr] = ssis_mock

    # No tiled dims => empty candidates list
    stub._joint_candidates_for_tensor = lambda t, _tr: []

    # Mock _transfer_latency_for_path to return 4
    stub._transfer_latency_for_path = staticmethod(lambda _tr, _choice: 4)

    link = MagicMock()
    link.bandwidth = 256
    choice = MagicMock()
    choice.links_used = [link]

    active_latency = stub._active_transfer_latency(tr, choice, y)

    model.update()
    model.setObjective(0)
    model.optimize()

    assert model.Status == GRB.OPTIMAL, f"Model infeasible, status={model.Status}"
    assert abs(active_latency.X - 4.0) < 1e-6, f"Expected latency=4.0, got {active_latency.X}"
    assert model.NumGenConstrs == 0, f"Expected 0 general constraints (NL), got {model.NumGenConstrs}"


def test_no_genconstr_nl_in_model(model):
    """Variable mode: after calling _active_transfer_latency, model.NumGenConstrs == 0.

    Same setup as test_active_transfer_latency_variable_mode. Verifies the model
    is a pure MILP — no addGenConstrNL calls remain.
    """
    from stream.workload.workload import Tensor

    stub = _make_latency_stub(model)

    tensor = MagicMock(spec=Tensor)
    tensor.name = "lat_tensor_nl"
    tr = MagicMock()
    tr.name = "tr_lat_nl"
    tr.inputs = [tensor]

    jw0 = model.addVar(vtype=GRB.BINARY, name="jw_nl_0")
    jw1 = model.addVar(vtype=GRB.BINARY, name="jw_nl_1")
    model.addConstr(jw0 + jw1 == 1, name="jw_nl_onehot")
    model.addConstr(jw0 == 1, name="jw_nl_force0")

    z = model.addVar(vtype=GRB.BINARY, name="z_nl")
    model.addConstr(z == 1, name="z_nl_fixed")

    y = model.addVar(vtype=GRB.BINARY, name="y_nl")
    model.addConstr(y == 1, name="y_nl_fixed")

    model.update()

    stub.reuse_levels[(tensor, -1)] = [(1, 1, jw0), (1, 2, jw1)]
    stub.z_stop[(tensor, -1)] = z

    ssis_mock = MagicMock()
    ssis_mock.get_applicable_temporal_variables.return_value = []
    stub.ssis[tr] = ssis_mock

    stub._joint_candidates_for_tensor = lambda t, _tr: [(1024, jw0), (2048, jw1)]

    link = MagicMock()
    link.bandwidth = 256
    choice = MagicMock()
    choice.links_used = [link]

    stub._active_transfer_latency(tr, choice, y)

    model.update()

    assert model.NumGenConstrs == 0, f"Expected 0 general constraints (NL), got {model.NumGenConstrs}"


# ---------------------------------------------------------------------------
# Phase 6 — Slot latency linearization tests
# ---------------------------------------------------------------------------


def _make_slot_latency_stub(model: gp.Model, search_space, latency_estimate_per_combo=None):
    """Build a minimal stub for _slot_latency_constraints and _create_idle_latency_vars testing.

    Provides:
      - One ssc_node mapped to slot 0
      - slot_latency[0] = gp.Var
      - latency_estimator = mock returning known LatencyEstimate values
      - workload with mocked get_unique_dims_inter_core_tiling and get_dimension_size
      - mapping with mocked get(node).resource_allocation
      - search_space (passed in, may be None)
      - _ssc_node_lat_coeffs = {}
    """
    from stream.cost_model.tile_aware_latency import LatencyEstimate
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub = types.SimpleNamespace()
    stub.model = model
    stub.search_space = search_space
    stub._ssc_node_lat_coeffs = {}
    stub._tensor_joint_candidates = {}
    stub.w = {}
    stub.tile_var = {}
    stub.y_path_choice = {}  # no transfer nodes in this stub

    # Create a mock computation node
    mock_node = MagicMock()
    mock_node.name = "test_ssc_node"

    # Create slot latency var
    slot_lat_var = model.addVar(vtype=GRB.INTEGER, lb=0, name="slot_latency_0")
    model.update()

    stub.ssc_nodes = [mock_node]
    stub.slot_of = {mock_node: 0}
    stub.slot_latency = {0: slot_lat_var}

    # Mock core
    from stream.hardware.architecture.core import Core

    core = MagicMock(spec=Core)
    core.id = 1

    # Mock mapping: mapping.get(node).resource_allocation[0] = {core}
    mapping = MagicMock()
    node_mapping = MagicMock()
    node_mapping.resource_allocation = [{core}]
    mapping.get.return_value = node_mapping
    stub.mapping = mapping

    # Mock workload
    workload = MagicMock()
    # get_unique_dims_inter_core_tiling returns list of (dim, factor) pairs
    # Using dim(0) as the tiled dim with factor=4
    d0 = _dim(0)
    base_tiling = ((d0, 4),)
    workload.get_unique_dims_inter_core_tiling.return_value = base_tiling
    # get_dims returns the node's computation dimensions (used by _tiled_dims_for_tensor
    # and _slot_latency_constraints to find which search_space dims affect the node)
    workload.get_dims.return_value = [d0]
    workload.get_dimension_size.return_value = 64  # workload_size for dim(0)
    stub.workload = workload

    # Mock latency_estimator
    latency_estimator = MagicMock()
    if latency_estimate_per_combo is None:
        # Default: returns latency 10 per candidate
        latency_estimator.estimate.return_value = LatencyEstimate(latency_total=10, ideal_cycle=8, energy_total=0.0)
    else:
        latency_estimator.estimate.side_effect = latency_estimate_per_combo
    stub.latency_estimator = latency_estimator

    # Bind the methods under test
    stub._slot_latency_constraints = TransferAndTensorAllocator._slot_latency_constraints.__get__(stub)
    stub._create_idle_latency_vars = TransferAndTensorAllocator._create_idle_latency_vars.__get__(stub)
    stub._add_binary_product = TransferAndTensorAllocator._add_binary_product.__get__(stub)
    stub._safe_name = TransferAndTensorAllocator._safe_name.__get__(stub)
    stub._joint_binary_for_combo = TransferAndTensorAllocator._joint_binary_for_combo.__get__(stub)
    stub._add_binary_scaled_continuous = TransferAndTensorAllocator._add_binary_scaled_continuous.__get__(stub)

    # Phase 7: _slot_latency_constraints now accesses _orig_workload/_orig_mapping
    # for stable dimension resolution. In tests, orig == current (no SSW translation).
    stub._orig_workload = workload
    stub._orig_mapping = mapping
    stub._ssw_dim_to_orig = lambda d: d  # identity: no SSW translation in tests
    stub._orig_dim_to_ssw = lambda d: d  # identity: no SSW translation in tests
    # _base_orig_dim_sizes: base tile per orig dim (from tile_options[0]); used for iteration scaling
    if search_space is not None and not search_space.is_empty():
        stub._base_orig_dim_sizes = {d: search_space.get(d)[0].tile for d in search_space.dims()}
    else:
        stub._base_orig_dim_sizes = {}
    stub._translate_tiling_to_ssw = lambda tiling: tiling  # identity in tests
    stub._iter_scale_by_jw = {}  # populated during _slot_latency_constraints

    return stub, mock_node, slot_lat_var, core


def _build_ss_for_slot(candidates: list[int]) -> SearchSpace:
    """Build a SearchSpace with dim(0) having the given candidates."""
    return _build_search_space({0: candidates})


def test_slot_latency_variable_mode(model):
    """Test 1: With 2+ candidates, _slot_latency_constraints creates quicksum(lat*jw) expression.

    Setup:
    - search_space: dim(0) with candidates [16, 32]
    - workload: dim(0) tiled with factor=4, workload_size=64
    - latency_estimator returns lat=10 for tile=16, lat=8 for tile=32
    - After calling _slot_latency_constraints:
      * model should have constraint ssc_lat_{node.name}
      * 2 lat_coeffs cached in _ssc_node_lat_coeffs
    """
    ss = _build_ss_for_slot([16, 32])

    from stream.cost_model.tile_aware_latency import LatencyEstimate
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub, mock_node, slot_lat_var, core = _make_slot_latency_stub(model, ss)

    # Pre-create w vars via __create_tile_selection_vars
    stub._TransferAndTensorAllocator__create_tile_selection_vars = (
        TransferAndTensorAllocator._TransferAndTensorAllocator__create_tile_selection_vars.__get__(stub)
    )
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    # estimator returns different latency per candidate
    estimates = [
        LatencyEstimate(latency_total=10, ideal_cycle=8, energy_total=0.0),
        LatencyEstimate(latency_total=8, ideal_cycle=7, energy_total=0.0),
    ]
    stub.latency_estimator.estimate.side_effect = estimates

    n_constrs_before = model.NumConstrs
    stub._slot_latency_constraints()
    model.update()

    # Constraint was added
    assert model.NumConstrs > n_constrs_before, "Expected at least one constraint added"
    constraint_names = {c.ConstrName for c in model.getConstrs()}
    assert f"ssc_lat_{mock_node.name}" in constraint_names, (
        f"Expected constraint 'ssc_lat_{mock_node.name}', got: {constraint_names}"
    )

    # _ssc_node_lat_coeffs was populated with 2 entries (one per candidate)
    assert mock_node in stub._ssc_node_lat_coeffs, "Expected lat coeffs to be cached"
    lat_coeffs = stub._ssc_node_lat_coeffs[mock_node]
    assert len(lat_coeffs) == 2, f"Expected 2 lat coefficients, got {len(lat_coeffs)}"

    # Verify latency values are correct (Phase 7: iteration scaling applied)
    # base_tile=16, candidate 16: lat=10 * 16/16 = 10
    # base_tile=16, candidate 32: lat=8 * 16/32 = 4
    lats = sorted(lat for lat, _ in lat_coeffs)
    assert lats == [4, 10], f"Expected latencies [4, 10], got {lats}"


def test_slot_latency_degenerate_single_candidate(model):
    """Test 2: Single candidate produces same RHS as scalar mode.

    With exactly 1 candidate, the constraint is effectively: slot_latency >= lat_single.
    This is the degenerate case per D-02.
    """
    ss = _build_ss_for_slot([16])  # single candidate

    from stream.cost_model.tile_aware_latency import LatencyEstimate
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub, mock_node, slot_lat_var, core = _make_slot_latency_stub(model, ss)
    stub._TransferAndTensorAllocator__create_tile_selection_vars = (
        TransferAndTensorAllocator._TransferAndTensorAllocator__create_tile_selection_vars.__get__(stub)
    )
    stub._TransferAndTensorAllocator__create_tile_selection_vars()
    model.update()

    # Single candidate with latency 12
    stub.latency_estimator.estimate.return_value = LatencyEstimate(latency_total=12, ideal_cycle=10, energy_total=0.0)

    stub._slot_latency_constraints()
    model.update()

    # Constraint exists
    constraint_names = {c.ConstrName for c in model.getConstrs()}
    assert f"ssc_lat_{mock_node.name}" in constraint_names

    # Exactly 1 lat_coeff cached
    lat_coeffs = stub._ssc_node_lat_coeffs[mock_node]
    assert len(lat_coeffs) == 1
    assert lat_coeffs[0][0] == 12


def test_slot_latency_scalar_fallback(model):
    """Test 3: When search_space is None, _slot_latency_constraints uses latency_estimator scalar path.

    With no search_space, the constraint should use latency_estimator.estimate() with the node's
    inter_core_tiling. CoreCostLUT is no longer used.
    """
    from stream.cost_model.tile_aware_latency import LatencyEstimate

    stub, mock_node, slot_lat_var, core = _make_slot_latency_stub(model, search_space=None)

    # Configure latency_estimator to return latency=15
    stub.latency_estimator.estimate.return_value = LatencyEstimate(latency_total=15, ideal_cycle=12, energy_total=0.0)
    # Set inter_core_tiling on mock_node
    mock_node.inter_core_tiling = []

    stub._slot_latency_constraints()
    model.update()

    # latency_estimator.estimate WAS called (new scalar fallback)
    stub.latency_estimator.estimate.assert_called_once()

    # Constraint was added
    constraint_names = {c.ConstrName for c in model.getConstrs()}
    assert f"ssc_lat_{mock_node.name}" in constraint_names

    # lat_coeffs should contain the scalar value (15)
    lat_coeffs = stub._ssc_node_lat_coeffs[mock_node]
    assert len(lat_coeffs) == 1
    # scalar mode: (runtime, None)
    assert lat_coeffs[0][0] == 15
    assert lat_coeffs[0][1] is None


def test_slot_latency_ub_variable_mode(model):
    """Test 4: _create_idle_latency_vars uses max over candidate latencies for slot_latency_ub.

    Setup: populate _ssc_node_lat_coeffs with two candidates having lats 8 and 12.
    Call _create_idle_latency_vars with max_s=0.
    Verify that the slot_latency_ub used is >= max(8, 12) = 12.
    """
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    stub, mock_node, slot_lat_var, core = _make_slot_latency_stub(model, search_space=None)

    # Pre-populate _ssc_node_lat_coeffs with known coefficients
    jw1 = model.addVar(vtype=GRB.BINARY, name="jw_ub_1")
    jw2 = model.addVar(vtype=GRB.BINARY, name="jw_ub_2")
    model.update()
    stub._ssc_node_lat_coeffs[mock_node] = [(8, jw1), (12, jw2)]

    # No transfer nodes (simplify test)
    stub.transfer_nodes = []

    # Set up idleS to have one resource at slot 0
    from stream.hardware.architecture.noc.communication_link import CommunicationLink

    link = MagicMock(spec=CommunicationLink)
    link.id = "test_link"

    is_ = model.addVar(vtype=GRB.BINARY, name="idleS_link_0")
    model.update()
    stub.idleS = {(link, 0): is_}
    stub.idleE = {}

    # Track call to _add_binary_scaled_continuous to capture ub passed
    captured_ubs = []
    original_method = TransferAndTensorAllocator._add_binary_scaled_continuous

    def tracking_method(binary_var, continuous_var, continuous_ub, base_name):
        captured_ubs.append(continuous_ub)
        return original_method(
            stub, binary_var=binary_var, continuous_var=continuous_var, continuous_ub=continuous_ub, base_name=base_name
        )

    stub._add_binary_scaled_continuous = tracking_method

    stub._create_idle_latency_vars(max_s=0)
    model.update()

    # The slot_latency_ub used should be >= 12 (max of our coeffs)
    assert len(captured_ubs) > 0, "Expected _add_binary_scaled_continuous to be called"
    max_ub_used = max(captured_ubs)
    assert max_ub_used >= 12, f"Expected slot_latency_ub >= 12 (max of [8, 12]), got {max_ub_used}"
