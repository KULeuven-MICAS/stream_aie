"""Regression tests for the baseline BIG BOY CO pipeline.

These tests run the full CO pipeline with Gurobi and compare results
against a checked-in JSON fixture. They are marked as @pytest.mark.slow
because the CO solver takes several minutes.

Run with: .venv/bin/pytest tests/regression/test_baseline.py -m slow -x
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE_PATH = PROJECT_ROOT / "tests" / "regression" / "fixtures" / "baseline_bigboy.json"

# BIG BOY config (per D-02)
BIGBOY_CONFIG = {
    "seq_len": 256,
    "embedding_dim": 2048,
    "hidden_dim": 8192,
    "in_dtype": "bf16",
    "out_dtype": "bf16",
    "trace_size": 1048576,
    "rows": 4,
    "cols": 8,
    "npu": "npu2",
    "seq_len_tile_options": [16],
    "embedding_tile_options": [128],
    "hidden_tile_options": [32],
    "last_gemm_down": True,
}


@pytest.fixture(scope="module")
def baseline_fixture():
    """Load the checked-in baseline fixture."""
    assert FIXTURE_PATH.exists(), f"Fixture not found: {FIXTURE_PATH}"
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fresh_run():
    """Run the v2 pipeline fresh and return (ctx, results)."""
    from main_swiglu_v2 import run_swiglu_v2

    ctx, results = run_swiglu_v2(**BIGBOY_CONFIG)
    return ctx, results


@pytest.mark.slow
def test_baseline_runs(fresh_run):
    """BASE-01: main_swiglu_v2.py runs end-to-end without error and produces a CO objective."""
    ctx, results = fresh_run
    assert results["latency_total"] > 0, "CO must produce a positive latency_total"
    assert results["latency_per_iteration"] > 0, "CO must produce a positive latency_per_iteration"
    assert len(results["fire_counts"]) > 0, "CO must produce fire counts"
    assert len(results["z_stop"]) > 0, "CO must produce z_stop assignments"


@pytest.mark.slow
def test_baseline_regression_latency(fresh_run, baseline_fixture):
    """BASE-02: Latency metrics match fixture within tolerance (per D-06: ~1e0 cycles)."""
    _, results = fresh_run
    expected = baseline_fixture

    assert results["latency_total"] == pytest.approx(expected["latency_total"], abs=1.0), (
        f"latency_total regression: got {results['latency_total']}, expected {expected['latency_total']}"
    )
    assert results["latency_per_iteration"] == pytest.approx(expected["latency_per_iteration"], abs=1.0), (
        f"latency_per_iteration regression: got {results['latency_per_iteration']}, expected {expected['latency_per_iteration']}"
    )
    assert results["overlap"] == pytest.approx(expected["overlap"], abs=1.0), (
        f"overlap regression: got {results['overlap']}, expected {expected['overlap']}"
    )


@pytest.mark.slow
def test_baseline_regression_fire_counts(fresh_run, baseline_fixture):
    """BASE-02: Per-transfer fire counts match fixture exactly (integer values)."""
    _, results = fresh_run
    expected = baseline_fixture

    assert set(results["fire_counts"].keys()) == set(expected["fire_counts"].keys()), (
        f"Fire count transfer names differ. Got: {sorted(results['fire_counts'].keys())}, "
        f"Expected: {sorted(expected['fire_counts'].keys())}"
    )
    for transfer_name, expected_count in expected["fire_counts"].items():
        actual_count = results["fire_counts"][transfer_name]
        assert actual_count == expected_count, (
            f"Fire count regression for {transfer_name}: got {actual_count}, expected {expected_count}"
        )


@pytest.mark.slow
def test_baseline_regression_z_stop(fresh_run, baseline_fixture):
    """BASE-02: z_stop (reuse) assignments match fixture."""
    _, results = fresh_run
    expected = baseline_fixture

    assert set(results["z_stop"].keys()) == set(expected["z_stop"].keys()), (
        f"z_stop tensor keys differ. Got: {sorted(results['z_stop'].keys())}, "
        f"Expected: {sorted(expected['z_stop'].keys())}"
    )
    for tensor_key, expected_reuse in expected["z_stop"].items():
        actual_reuse = results["z_stop"][tensor_key]
        assert actual_reuse == expected_reuse, (
            f"z_stop regression for {tensor_key}: got {actual_reuse}, expected {expected_reuse}"
        )
