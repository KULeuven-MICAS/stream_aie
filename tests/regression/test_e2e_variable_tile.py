"""E2E validation: variable tile CO on SwiGLU BIG BOY with multi-candidate tile selection.

Validates PIPE-01: the CO selects valid tile sizes and produces a feasible allocation
at least as good as the fixed-tile baseline.

Run with: .venv/bin/pytest tests/regression/test_e2e_variable_tile.py -m slow -x
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_PATH = PROJECT_ROOT / "tests" / "regression" / "fixtures" / "baseline_bigboy.json"

# BIG BOY config with MULTIPLE candidates per dimension (per D-08)
# Baseline tile is FIRST in each list (tile_options[0] sets the reference tile used by
# make_swiglu_mapping_v2 for the SSIS iteration count).  Putting the baseline first ensures
# the pre-solve iteration count matches the fixed-tile baseline so latency_total is
# comparable across runs.
MULTI_CANDIDATE_CONFIG = {
    "seq_len": 256,
    "embedding_dim": 2048,
    "hidden_dim": 8192,
    "in_dtype": "bf16",
    "out_dtype": "bf16",
    "trace_size": 1048576,
    "rows": 4,
    "cols": 8,
    "npu": "npu2",
    "seq_len_tile_options": [16, 8, 32],        # baseline 16 first, all divisors of 256
    "embedding_tile_options": [128, 64, 256],    # baseline 128 first, all divisors of 2048
    "hidden_tile_options": [32, 16, 64],         # baseline 32 first, all divisors of 8192
    "last_gemm_down": True,
}

# Workload dimensions for divisibility checks (per D-09)
WORKLOAD_DIMS = {
    "seq_len": 256,
    "embedding_dim": 2048,
    "hidden_dim": 8192,
}


@pytest.fixture(scope="module")
def baseline_fixture():
    assert FIXTURE_PATH.exists(), f"Fixture not found: {FIXTURE_PATH}"
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def multi_candidate_run():
    from main_swiglu_v2 import run_swiglu_v2

    ctx, results = run_swiglu_v2(**MULTI_CANDIDATE_CONFIG)
    return ctx, results


@pytest.mark.slow
def test_e2e_completes(multi_candidate_run):
    """PIPE-01: Multi-candidate run completes without error."""
    ctx, results = multi_candidate_run
    assert results["latency_total"] > 0
    assert results["latency_per_iteration"] > 0
    assert len(results["fire_counts"]) > 0


@pytest.mark.slow
def test_selected_tiles_exist(multi_candidate_run):
    """PIPE-01: CO reports selected tile sizes."""
    ctx, results = multi_candidate_run
    selected_tiles = ctx.get("selected_tiles")
    assert selected_tiles is not None, "selected_tiles must be set on ctx"
    assert len(selected_tiles) > 0, "selected_tiles must not be empty"


@pytest.mark.slow
def test_selected_tiles_are_valid_divisors(multi_candidate_run):
    """D-09: Selected tile sizes are valid divisors of their workload dimensions."""
    ctx, results = multi_candidate_run
    selected_tiles = ctx.get("selected_tiles")
    for dim, tile in selected_tiles.items():
        dim_name = str(dim)
        # Find matching workload dim
        workload_size = None
        for wl_name, wl_size in WORKLOAD_DIMS.items():
            if wl_name in dim_name.lower() or dim_name.lower() in wl_name:
                workload_size = wl_size
                break
        if workload_size is not None:
            assert workload_size % tile == 0, (
                f"Selected tile {tile} for {dim} is not a divisor of workload size {workload_size}"
            )


@pytest.mark.slow
def test_objective_at_least_as_good_as_baseline(multi_candidate_run, baseline_fixture):
    """D-08: CO objective with multiple candidates >= fixed-tile baseline."""
    _, results = multi_candidate_run
    baseline_latency = baseline_fixture["latency_total"]
    # Lower latency = better. Multi-candidate must be <= baseline (at least as good).
    assert results["latency_total"] <= baseline_latency * 1.001, (
        f"Multi-candidate latency {results['latency_total']} worse than "
        f"baseline {baseline_latency}"
    )
