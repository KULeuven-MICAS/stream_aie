from zigzag.datatypes import Constants

from stream.node_tensor import NodeTensor
from stream.workload.dependency_propagation.propagation_node import PropagationNode
from stream.workload.node import Node


class ResizeNode(PropagationNode):
    """Class that represents an onnx Resize node."""

    def __init__(
        self,
        node_id: int,
        node_name: str,
        predecessor: int,
        output_shape: tuple[int, ...],
        input_names: list[str] | None = None,
    ) -> None:
        """Initialize the ResizeNode

        Args:
            predecessors: The id of this node's parent.
            output_shape: The shape of the output tensor.
        """
        if input_names is None:
            input_names = []
        op_type = "resize"
        super().__init__(node_id, node_name, op_type, input_names)

        self.output_shape = output_shape
        self.input_operand_source = {Constants.LAYER_OP_I: predecessor}

    def propagate(
        self,
        tensor: NodeTensor,
        previous_node: Node | None = None,
        next_node: Node | None = None,
        relevant_axes: list[bool] | None = None,
    ) -> tuple[NodeTensor, list[bool]]:
        """Resize the tensor to the output shape.
        We assume nearest neighbor upsampling for dependency propagation, which means repeating elements.
        """
        if relevant_axes is None:
            relevant_axes = [False] * len(tensor.tensor_shape)

        # Calculate scale factors
        input_shape = tensor.tensor_shape
        output_shape = self.output_shape
        
        # Handle case where input/output shapes might differ in length (e.g. broadcasting)
        # But Resize usually preserves rank.
        if len(input_shape) != len(output_shape):
             raise NotImplementedError("Resize with different rank not supported")

        # Perform resizing on NodeTensor
        # We use numpy repeat to simulate nearest neighbor upsampling
        resized_tensor = tensor
        for i, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
            if in_dim == out_dim:
                continue
            
            if out_dim % in_dim != 0:
                 # If not integer scale, we can't just repeat.
                 # But for dependency tracking, maybe we can just resize?
                 # NodeTensor is object array.
                 # We can use scipy.ndimage.zoom? No, it expects numbers.
                 # We can use a custom resize function.
                 # For now, assume integer scale (common in YOLO upsample).
                 raise NotImplementedError(f"Resize scale must be integer. {in_dim} -> {out_dim}")
            
            scale = out_dim // in_dim
            # NodeTensor has extra dimension at the end. We should not repeat that.
            # But np.repeat works on axis.
            # Axis i in tensor_shape corresponds to axis i in NodeTensor.
            resized_tensor = resized_tensor.repeat(scale, axis=i)

        return resized_tensor.view(NodeTensor), relevant_axes
