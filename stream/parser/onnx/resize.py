from stream.onnx_utils import get_onnx_output_shapes
from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.resize_node import ResizeNode


class ResizeParser(OnnxOperatorParser):
    """Parses an ONNX Resize operator into a ResizeNode."""

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        input_names = list(self.node.input)
        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)
        output_shape = output_shapes[0]

        # We assume the first input is the data tensor (X)
        # The second input (roi) or third (scales) or fourth (sizes) determine output shape.
        # But we already got output_shape from onnx helper.
        # So we just need to pass the predecessor that provides data.
        # Predecessors list contains node IDs.
        # Usually the first predecessor is the data producer.
        # But Resize might have constant inputs for roi, scales, sizes.
        # OnnxOperatorParser.get_node_predecessors returns predecessors for ALL inputs.
        # We need to identify which one is the data input.
        # Usually input[0] is X.
        
        # Find the predecessor corresponding to input[0]
        data_input_name = self.node.input[0]
        # Find which predecessor produces this input
        # self.nodes_outputs maps node_id -> output_names
        data_predecessor = None
        for pred_id in predecessors:
             # We don't have easy access to predecessor output names here directly from predecessors list
             # But we can check if pred_id is in self.nodes_outputs
             # Wait, OnnxOperatorParser has self.nodes_outputs.
             if data_input_name in self.nodes_outputs[pred_id]:
                 data_predecessor = pred_id
                 break
        
        if data_predecessor is None:
            # Maybe input is graph input?
            # If so, predecessor is None?
            # But PropagationNode expects an integer predecessor.
            # If it's a graph input, we might need a DummyNode for it?
            # But usually graph inputs are handled.
            # Let's assume there is a predecessor.
            # If not found, maybe it's because we are looking at constant inputs?
            # If input[0] is constant, Resize on constant?
            # Unlikely for main data flow.
            # Let's fallback to first predecessor if we can't find by name (risky but might work if only 1 data pred)
            if len(predecessors) > 0:
                data_predecessor = predecessors[0]
            else:
                raise ValueError(f"No predecessor found for Resize node {self.node.name}")

        return ResizeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=data_predecessor,
            output_shape=output_shape,
            input_names=input_names,
        )
