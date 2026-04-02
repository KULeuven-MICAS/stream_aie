import argparse
import json
import logging as _logging
import os
import re
from math import prod

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_swiglu_mapping_v2 import make_swiglu_mapping_v2
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"


def run_swiglu_v2(  # noqa: PLR0913
    seq_len,
    embedding_dim,
    hidden_dim,
    in_dtype,
    out_dtype,
    trace_size,
    rows,
    cols,
    npu,
    seq_len_tile_options: list[int],
    embedding_tile_options: list[int],
    hidden_tile_options: list[int],
    last_gemm_down: bool,
):  # noqa: N803
    """Run the SwiGLU v2 pipeline with tile_options format mapping.

    Args:
        seq_len: Sequence length dimension.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        in_dtype: Input data type (e.g. "bf16").
        out_dtype: Output data type (e.g. "bf16").
        trace_size: Trace buffer size in bytes.
        rows: Number of AIE rows to use.
        cols: Number of AIE columns to use.
        npu: NPU type to target (e.g. "npu2").
        seq_len_tile_options: List of possible tile sizes for seq_len dimension.
        embedding_tile_options: List of possible tile sizes for embedding dimension.
        hidden_tile_options: List of possible tile sizes for hidden dimension.
        last_gemm_down: Whether to include the final down projection Gemm.

    Returns:
        Tuple of (ctx, results) where ctx is the StageContext and results is the metrics dict.
    """
    accelerator = os.path.join(os.path.dirname(__file__), "stream/inputs/aie/hardware/whole_array_strix.yaml")

    # Create workload and mapping
    workload_path = make_swiglu_workload(seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype, last_gemm_down=last_gemm_down)
    mapping_path = make_swiglu_mapping_v2(seq_len, embedding_dim, hidden_dim, last_gemm_down, seq_len_tile_options, embedding_tile_options, hidden_tile_options)

    # Build experiment ID following same pattern as existing pipelines
    hw_name = accelerator.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"v2-{hw_name}-{wl_name}-{rows}_row_{cols}_col"

    # Set up logging to file
    log_path = os.path.join(os.getcwd(), f"outputs/{experiment_id}/stream.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = _logging.getLogger()
    logger.setLevel(_logging_level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = _logging.FileHandler(log_path)
    file_handler.setFormatter(_logging.Formatter(_logging_format))
    logger.addHandler(file_handler)
    logger.info(
        "Running SwiGLU v2 pipeline with "
        "seq_len=%d, embedding_dim=%d, hidden_dim=%d, "
        "seq_len_tile_options=%s, embedding_tile_options=%s, hidden_tile_options=%s",
        seq_len,
        embedding_dim,
        hidden_dim,
        seq_len_tile_options,
        embedding_tile_options,
        hidden_tile_options,
    )

    # Run the CO optimization pipeline
    ctx = optimize_allocation_co(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id=experiment_id,
        output_path="outputs",
        skip_if_exists=False,
        enable_codegen=False,
        trace_size=trace_size,
        nb_cols_to_use=cols,
        npu=npu,
    )

    # Extract metrics from context
    scheduler = ctx.get("scheduler")
    latency_total = scheduler.latency_total
    latency_per_iteration = scheduler.latency_per_iteration
    overlap = scheduler.overlap_between_iterations

    # Fire counts from transfer node SSIS
    fire_counts: dict[str, int] = {}
    try:
        for node in scheduler.steady_state_workload.get_transfer_nodes():
            ssis_for_node = scheduler.ssis.get(node, None)
            if ssis_for_node is not None:
                nb = prod(ssis_for_node.get_applicable_temporal_sizes())
                reuse = ssis_for_node.reuse_factor()
                fire_counts[node.name] = nb // reuse if reuse > 0 else nb
    except AttributeError:
        # Methods may not be available in this phase; fall back to temporal sizes
        try:
            for node in scheduler.steady_state_workload.get_transfer_nodes():
                ssis_for_node = scheduler.ssis.get(node, None)
                if ssis_for_node is not None:
                    nb = prod(ssis_for_node.get_temporal_sizes())
                    fire_counts[node.name] = nb
        except AttributeError:
            pass

    # z_stop (reuse) assignments via tensor-keyed SSIS
    z_stop: dict[str, object] = {}
    try:
        for node in scheduler.steady_state_workload.get_transfer_nodes():
            for i, tensor in enumerate(node.outputs):
                ssis_for_tensor = scheduler.ssis.get(tensor, None)
                if ssis_for_tensor is not None:
                    z_stop[f"{node.name}__out{i}"] = ssis_for_tensor.reuse_summary()
    except AttributeError:
        pass

    # Build results dict
    results = {
        "latency_total": latency_total,
        "latency_per_iteration": latency_per_iteration,
        "overlap": overlap,
        "fire_counts": fire_counts,
        "z_stop": z_stop,
        "config": {
            "seq_len": seq_len,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "seq_len_tile_options": seq_len_tile_options,
            "embedding_tile_options": embedding_tile_options,
            "hidden_tile_options": hidden_tile_options,
        },
    }

    # Save JSON results
    results_path = f"outputs/{experiment_id}/results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print key metrics to stdout
    print(f"Latency total: {latency_total}")
    print(f"Latency per iteration: {latency_per_iteration}")
    print(f"Overlap: {overlap}")
    print(f"Results saved to: outputs/{experiment_id}/results.json")

    return ctx, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SwiGLU v2 pipeline with tile_options mapping")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--embedding_dim", type=int, default=2048, help="Embedding dimension (default: 2048)")
    parser.add_argument("--hidden_dim", type=int, default=8192, help="Hidden dimension (default: 8192)")
    parser.add_argument("--in_dtype", type=str, default="bf16", help="Input data type (default: bf16)")
    parser.add_argument("--out_dtype", type=str, default="bf16", help="Output data type (default: bf16)")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Trace buffer size in bytes (default: 1048576)")
    parser.add_argument("--rows", type=int, default=4, help="Number of AIE rows to use (default: 4)")
    parser.add_argument("--cols", type=int, default=8, help="Number of AIE columns to use (default: 8)")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target (default: npu2)")
    parser.add_argument("--seq_len_tile_size", type=int, default=16, help="Tile size for seq_len dimension (default: 16)")
    parser.add_argument("--embedding_tile_size", type=int, default=128, help="Tile size for embedding dimension (default: 128)")
    parser.add_argument("--hidden_tile_size", type=int, default=32, help="Tile size for hidden dimension (default: 32)")
    parser.add_argument(
        "--no_last_gemm_down",
        dest="last_gemm_down",
        action="store_false",
        default=True,
        help="If set, the last down projection Gemm is skipped",
    )
    args = parser.parse_args()

    # Convert scalar tile sizes to single-element lists for tile_options format
    seq_len_tile_options = [args.seq_len_tile_size]
    embedding_tile_options = [args.embedding_tile_size]
    hidden_tile_options = [args.hidden_tile_size]

    run_swiglu_v2(
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        in_dtype=args.in_dtype,
        out_dtype=args.out_dtype,
        trace_size=args.trace_size,
        rows=args.rows,
        cols=args.cols,
        npu=args.npu,
        seq_len_tile_options=seq_len_tile_options,
        embedding_tile_options=embedding_tile_options,
        hidden_tile_options=hidden_tile_options,
        last_gemm_down=args.last_gemm_down,
    )
