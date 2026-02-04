from zigzag.datatypes import Constants, LayerDim
from zigzag.parser.onnx.utils import OnnxTensorCategory, get_onnx_tensor_type

from stream.onnx_utils import get_onnx_input_shapes, get_onnx_output_shapes
from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.concat_node import ConcatConstantNode, ConcatNode


class ConcatParser(OnnxOperatorParser):
    """Parses an ONNX Concat operator with one constant input into a ConcatConstantNode.
    # TODO also parse concat nodes with non-constant inputs
    """

    def get_axis_value(self):
        """Find the value of the axis associated with this concat node in ONNX"""
        axis_attr = "axis"
        # `axis` is an attribute of the node
        try:
            axis_attr = next(filter(lambda x: x.name == axis_attr, self.node.attribute))
            return axis_attr.i
        except StopIteration as exc:
            raise ValueError("Axis attribute not found in ONNX node") from exc

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        axis = self.get_axis_value()
        input_names = list(self.node.input)
        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)
        output_shape = output_shapes[0]

        mapping = self.all_mappings.get(self.node.op_type)
        if mapping and mapping.layer_dimension_names and len(mapping.layer_dimension_names) == len(output_shape):
            loop_dims = mapping.layer_dimension_names
        else:
            loop_dims = ["B", "H", "D", "K"][: len(output_shape)]

        loop_dim_size = {}
        for dim_name, size in zip(loop_dims, output_shape):
            loop_dim_size[LayerDim(dim_name)] = size

        try:  # Try first one as constant input
            input_1, input_2 = self.node.input[0], self.node.input[1]
            constant_tensor = get_onnx_tensor_type(input_1, self.onnx_model)
            if constant_tensor.category != OnnxTensorCategory.HIDDEN or "constant" not in input_1.lower():
                raise ValueError

            constant_shape = tuple(constant_tensor.shape)
            variable_input_first = True
            node = ConcatConstantNode(
                node_id=self.node_id,
                node_name=self.node.name,
                predecessors=predecessors,
                axis=axis,
                constant_shape=constant_shape,
                variable_input_first=variable_input_first,
                input_names=input_names,
            )
            node.loop_dim_size = loop_dim_size
            dim_list = [LayerDim(d) for d in loop_dims]
            node.operand_dimensionality_order = {}
            for op in node.input_operand_source.keys():
                node.operand_dimensionality_order[op] = dim_list
            node.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP] = dim_list
            return node
        except (ValueError, IndexError):
            pass

        try:  # Try second one as constant input
            input_1, input_2 = self.node.input[0], self.node.input[1]
            constant_tensor = get_onnx_tensor_type(input_2, self.onnx_model)
            if constant_tensor.category != OnnxTensorCategory.HIDDEN or "constant" not in input_2.lower():
                raise ValueError("Second input is not a constant tensor")

            constant_shape = tuple(constant_tensor.shape)
            variable_input_first = True
            node = ConcatConstantNode(
                node_id=self.node_id,
                node_name=self.node.name,
                predecessors=predecessors,
                axis=axis,
                constant_shape=constant_shape,
                variable_input_first=variable_input_first,
                input_names=input_names,
            )
            node.loop_dim_size = loop_dim_size
            dim_list = [LayerDim(d) for d in loop_dims]
            node.operand_dimensionality_order = {}
            for op in node.input_operand_source.keys():
                node.operand_dimensionality_order[op] = dim_list
            node.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP] = dim_list
            return node
        except (ValueError, IndexError):
            pass

        # Fallback to ConcatNode (variable inputs)
        input_shapes = get_onnx_input_shapes(self.node, self.onnx_model)
        node = ConcatNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            axis=axis,
            output_shape=output_shape,
            input_names=input_names,
            axis_exists_in_input=True,
            input_shapes=input_shapes,
        )
        node.loop_dim_size = loop_dim_size
        dim_list = [LayerDim(d) for d in loop_dims]
        node.operand_dimensionality_order = {}
        for op in node.input_operand_source.keys():
            node.operand_dimensionality_order[op] = dim_list
        node.operand_dimensionality_order[Constants.OUTPUT_LAYER_OP] = dim_list
        return node
