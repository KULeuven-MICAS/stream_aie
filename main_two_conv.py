import argparse
import logging as _logging
import re

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_two_conv_mapping import make_two_conv_mapping
from stream.inputs.aie.workload.make_onnx_two_conv import make_two_conv_workload

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


def run_main_aie_codegen_two_conv(  # noqa: PLR0913
    input_size,
    in_channels,
    mid_channels,
    out_channels,
    kernel_size,
    stride,
    in_dtype,
    out_dtype,
    trace_size,
):  # noqa: N803, PLR0913
    ############################################INPUTS############################################
    # CREATE THE CONV ONNX MODEL
    workload_path = make_two_conv_workload(
        input_size, in_channels, mid_channels, out_channels, kernel_size, stride, in_dtype, out_dtype, pad="none"
    )
    accelerator = "stream/inputs/aie/hardware/whole_array.yaml"
    mapping_path = make_two_conv_mapping(
        input_size,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
    )
    mode = "fused"
    layer_stacks = [(0, 1)]
    ##############################################################################################

    ################################PARSING###############################
    hw_name = accelerator.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"{hw_name}-{wl_name}"
    ######################################################################

    ##############PLOTTING###############
    # section_start_percent = (0,)
    # percent_shown = (100,)
    #####################################

    ################################PATHS################################
    # memory_fig_path = f"outputs/{experiment_id}/memory.png"
    # json_path = f"outputs/{experiment_id}/scme.json"
    #####################################################################

    _ = optimize_allocation_co(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=layer_stacks,
        experiment_id=experiment_id,
        output_path="outputs",
        skip_if_exists=False,
        enable_codegen=True,
        trace_size=trace_size,
        nb_cols_to_use=1,
    )

    # #####################CostModelEvaluationLUT LOAD#############################
    # cost_lut_path = f"outputs/{experiment_id}/cost_lut_post_co.pickle"
    # cost_lut = CostModelEvaluationLUT(cost_lut_path)
    # #############################################################################

    # # Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
    # convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

    # # Plotting memory usage of best SCME
    # plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIE code generation for Two Conv layers")
    parser.add_argument("--input_size", type=int, required=True, help="Input size for the model")
    parser.add_argument("--in_channels", type=int, required=True, help="Number of input channels")
    parser.add_argument("--mid_channels", type=int, required=True, help="Number of intermediate channels")
    parser.add_argument("--out_channels", type=int, required=True, help="Number of output channels")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size (default: 3)")
    parser.add_argument("--stride", type=int, default=1, help="Stride (default: 1)")
    parser.add_argument("--in_dtype", type=str, default="i16", help="Input data type (default: i16)")
    parser.add_argument("--out_dtype", type=str, default="i32", help="Output data type (default: i32)")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Size of the trace buffer (default: 1048576)")
    args = parser.parse_args()

    run_main_aie_codegen_two_conv(
        args.input_size,
        args.in_channels,
        args.mid_channels,
        args.out_channels,
        args.kernel_size,
        args.stride,
        args.in_dtype,
        args.out_dtype,
        args.trace_size,
    )
