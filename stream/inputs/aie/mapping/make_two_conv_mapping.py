import yaml


def make_two_conv_mapping(
    input_size: int = 224,
    in_channels: int = 16,
    mid_channels: int = 32,
    out_channels: int = 48,
    kernel_size: int = 3,
    stride: int = 1,
    kernel_name: str | None = None,
):
    """
    Create a YAML mapping for a two-conv workload with:
      - Conv #0 on core id 2
      - Conv #1 on core id 3
      - intra_core_tiling: [('OY', oy_size)]
      - inter_core_tiling: []

    Args:
        name: Base model/workload name; used in the output filename.
        oy_size: Tile size along OY (output height).
        kernel_name: Optional kernel name string; defaults to 'conv3x3'.
    Returns:
        Path to the generated YAML file.
    """
    name: str = f"two_conv_{input_size}_{in_channels}_{mid_channels}_{out_channels}_{kernel_size}_{stride}"
    output_file = f"stream/inputs/aie/mapping/{name}.yaml"
    intra_core_tiling = [f"OY, {input_size}"]
    inter_core_tiling = []

    kname = kernel_name or "conv3x3"
    kernel = {"name": kname, "utilization": 50.0}  # TODO: adjust kernel name and utilization

    mapping = [
        {
            "name": "Conv0",
            "core_allocation": [2],
            "intra_core_tiling": intra_core_tiling,
            "inter_core_tiling": inter_core_tiling,
            "kernel": kernel,
        },
        {
            "name": "Conv1",
            "core_allocation": [3],
            "intra_core_tiling": intra_core_tiling,
            "inter_core_tiling": inter_core_tiling,
            "kernel": kernel,
        },
        # Fallback rule (shouldn't get used in this case)
        {
            "name": "default",
            "core_allocation": [2, 3],
            "intra_core_tiling": intra_core_tiling,
            "inter_core_tiling": inter_core_tiling,
            "kernel": kernel,
        },
    ]

    with open(output_file, "w") as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    return output_file
