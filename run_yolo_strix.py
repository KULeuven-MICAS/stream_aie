import logging
import os
from stream.api import optimize_allocation_co

# Setup logging
logging.basicConfig(level=logging.INFO)

# Inputs
accelerator = "stream/inputs/aie/hardware/whole_array_strix.yaml"
workload_path = "yolo11n.onnx"
mapping_path = "yolo_mapping.yaml"
mode = "fused_topology"
experiment_id = "strix_yolo11n_fused"
output_path = "outputs"

# Run optimization
# We pass layer_stacks=None to let the tool automatically generate them based on weight capacity
optimize_allocation_co(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=None,  # Auto-generate
    experiment_id=experiment_id,
    output_path=output_path,
    skip_if_exists=False,
    nb_cols_to_use=8,
    enable_codegen=True, # Enable codegen as requested "get yolo11n working" usually implies running it
    npu="npu2" # Assuming Strix uses NPU2 or similar, default is npu2
)
