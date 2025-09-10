import numpy as np
import onnx
import onnx.shape_inference
from onnx import TensorProto, helper


def _sizes_from_dtypes(in_dtype: str, out_dtype: str):
    if "16" in in_dtype:
        ACT_SIZE = 16
        WEIGHT_SIZE = 16
    elif "32" in in_dtype:
        ACT_SIZE = 32
        WEIGHT_SIZE = 32
    else:
        raise ValueError(f"Unsupported input data type: {in_dtype}")

    if "16" in out_dtype:
        OUTPUT_SIZE = 16
    elif "32" in out_dtype:
        OUTPUT_SIZE = 32
    else:
        raise ValueError(f"Unsupported output data type: {out_dtype}")

    return ACT_SIZE, WEIGHT_SIZE, OUTPUT_SIZE


def make_two_conv_workload(
    input_size: int = 224,
    in_channels: int = 16,
    mid_channels: int = 32,
    out_channels: int = 48,
    kernel_size: int = 3,
    stride: int = 1,
    in_dtype: str = "fp16",
    out_dtype: str = "fp16",
):  # noqa: N803
    """
    Create an ONNX model with two back-to-back Conv layers:
      Input:  [1, in_channels, input_size, input_size]
      Conv1:  in_channels -> mid_channels, kxk, stride=1, SAME padding
      Conv2:  mid_channels -> out_channels, kxk, stride=1, SAME padding
      Output: [1, out_channels, input_size, input_size]

    Weights are included only for shape (zero arrays with data cleared).
    """

    if kernel_size % 2 != 1:
        raise ValueError("Only odd kernel sizes supported for SAME padding.")
    if stride != 1:
        raise ValueError("This helper currently assumes stride=1.")

    ACT_SIZE, WEIGHT_SIZE, OUTPUT_SIZE = _sizes_from_dtypes(in_dtype, out_dtype)
    name = f"two_conv_{input_size}_{in_channels}_{mid_channels}_{out_channels}_k{kernel_size}_s{stride}"

    # Shapes
    N, H, W = 1, input_size, input_size
    pads_2d = kernel_size // 2  # SAME padding for stride=1 with odd kernel
    pads = [pads_2d, pads_2d, pads_2d, pads_2d]  # [top, left, bottom, right]

    # IO tensors
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, in_channels, H, W])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)  # inferred

    # --- Conv1 ---
    W1_name = "W1"
    conv1_node = helper.make_node(
        "Conv",
        name="Conv0",
        inputs=["X", W1_name],  # no bias
        outputs=["Y1"],
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=pads,
        dilations=[1, 1],
        group=1,
        act_size=ACT_SIZE,
        weight_size=WEIGHT_SIZE,
        output_size=OUTPUT_SIZE,
    )
    W1 = helper.make_tensor(
        W1_name,
        TensorProto.FLOAT,
        [mid_channels, in_channels, kernel_size, kernel_size],
        np.zeros((mid_channels, in_channels, kernel_size, kernel_size), dtype=np.float32),
    )
    # Keep the initializer lightweight: remove float_data payload
    W1.ClearField("float_data")

    # --- Conv2 ---
    W2_name = "W2"
    conv2_node = helper.make_node(
        "Conv",
        name="Conv1",
        inputs=["Y1", W2_name],  # no bias
        outputs=["Y"],
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=pads,
        dilations=[1, 1],
        group=1,
        act_size=ACT_SIZE,
        weight_size=WEIGHT_SIZE,
        output_size=OUTPUT_SIZE,
    )
    W2 = helper.make_tensor(
        W2_name,
        TensorProto.FLOAT,
        [out_channels, mid_channels, kernel_size, kernel_size],
        np.zeros((out_channels, mid_channels, kernel_size, kernel_size), dtype=np.float32),
    )
    W2.ClearField("float_data")

    # Graph & model
    graph = helper.make_graph(
        nodes=[conv1_node, conv2_node],
        name=name,
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[W1, W2],
    )
    model = helper.make_model(graph, producer_name="stream-aie")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)

    # Save
    save_path = f"stream/inputs/aie/workload/{name}.onnx"
    onnx.save(inferred_model, save_path)
    print(f"{name} exported to {save_path}.")

    return save_path


if __name__ == "__main__":
    # Example usage (matches your requested sizes)
    make_two_conv_workload(
        input_size=224,
        in_channels=16,
        mid_channels=32,
        out_channels=48,
        kernel_size=3,
        stride=1,
        in_dtype="fp16",
        out_dtype="fp16",
    )
