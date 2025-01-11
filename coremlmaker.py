#!/usr/bin/env python3

import argparse
import logging
import sys
import onnx
import numpy as np
from onnx import numpy_helper, shape_inference
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder


def load_onnx_model(onnx_path: str):
    """
    Loads an ONNX model from the specified file.
    Returns the model protobuf object.
    """
    logging.info(f"Loading ONNX model from: {onnx_path}")
    try:
        model = onnx.load(onnx_path)
        logging.info("ONNX model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load ONNX model: {e}")
        sys.exit(1)


def infer_shapes(onnx_model):
    """
    Runs ONNX shape inference on the model.
    If shape inference fails or shapes remain undefined,
    the user will be prompted later in the pipeline.
    """
    logging.info("Attempting to run shape inference on the ONNX model.")
    try:
        inferred_model = shape_inference.infer_shapes(onnx_model)
        logging.info("Shape inference completed.")
        return inferred_model
    except Exception as e:
        logging.warning(f"Shape inference failed: {e}")
        return onnx_model  # Return the model unchanged if inference fails


def extract_tensor_data(initializer):
    """
    Convert an ONNX initializer (weight/bias) to a NumPy array.
    """
    return numpy_helper.to_array(initializer)


def parse_onnx_model(onnx_model):
    """
    Parse the ONNX model graph to extract:
      - Node information (op_type, name, inputs, outputs).
      - Initializer data for weights/biases.
      - Input/Output tensor shapes if available.

    Returns:
      nodes_info: list of dict describing each ONNX node
      initializers: dict of {tensor_name: np.array} for weight/bias
      input_shapes: dict of {input_name: list/tuple} for model inputs
      output_shapes: dict of {output_name: list/tuple} for model outputs
      constants_map = {}  # <---  dictionary

    """
    logging.info("Parsing ONNX model graph.")
    graph = onnx_model.graph

    initializers = {}
    constants_map = {}  # <---
    nodes_info = []

    # 1. Extract initializers (weights, biases, etc.)
    for init in graph.initializer:
        data_array = numpy_helper.to_array(init)
        initializers[init.name] = data_array

    # 2. Gather input and output shapes
    input_shapes = {}
    for inp in graph.input:
        shape = []
        if inp.type.tensor_type.shape.dim:
            for d in inp.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value > 0 else None)
        input_shapes[inp.name] = shape

    output_shapes = {}
    for out in graph.output:
        shape = []
        if out.type.tensor_type.shape.dim:
            for d in out.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value > 0 else None)
        output_shapes[out.name] = shape

    # 3. Iterate over ONNX nodes
    for node in graph.node:
        if node.op_type == "Constant":
            # This node stores a constant. Let's read its output name
            # and extract the constant value from its attributes.
            if len(node.output) > 0:
                const_out_name = node.output[0]
                for attr in node.attribute:
                    if attr.name == "value":
                        arr = numpy_helper.to_array(attr.t)
                        constants_map[const_out_name] = arr
            # We do NOT add this node to nodes_info, because
            # we treat it purely as data rather than a compute layer.
            continue

        # Otherwise, normal node
        node_dict = {
            'name': node.name,
            'op_type': node.op_type,
            'inputs': list(node.input),
            'outputs': list(node.output),
            'attributes': {}
        }
        for attr in node.attribute:
            # parse attributes
            if attr.type == onnx.AttributeProto.FLOAT:
                node_dict['attributes'][attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                node_dict['attributes'][attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.STRING:
                node_dict['attributes'][attr.name] = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.TENSOR:
                node_dict['attributes'][attr.name] = numpy_helper.to_array(attr.t)
            elif attr.type == onnx.AttributeProto.FLOATS:
                node_dict['attributes'][attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.INTS:
                node_dict['attributes'][attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.STRINGS:
                node_dict['attributes'][attr.name] = [s.decode('utf-8') for s in attr.strings]
            else:
                node_dict['attributes'][attr.name] = None

        nodes_info.append(node_dict)

    logging.info("Graph parsing complete.")
    return nodes_info, initializers, constants_map, input_shapes, output_shapes


def prompt_for_missing_dimensions(shape, tensor_name):
    """
    Prompts the user to fill in missing dimensions (None) for the given tensor.
    Returns a list of dimensions with no None values.
    """
    new_shape = []
    for i, dim in enumerate(shape):
        if dim is None:
            user_dim = input(
                f"Dimension #{i} for tensor '{tensor_name}' is undefined. "
                "Please enter the dimension size (int): "
            )
            try:
                new_shape.append(int(user_dim))
            except ValueError:
                logging.error("Invalid dimension, defaulting to 1.")
                new_shape.append(1)
        else:
            new_shape.append(dim)
    return new_shape


def build_coreml_model(nodes_info, initializers, input_shapes, output_shapes):
    """
    Builds a Core ML NeuralNetworkBuilder model by iterating over the ONNX nodes
    and adding equivalent layers in Core ML.

    This function returns the Core ML spec.
    """
    logging.info("Building Core ML model using NeuralNetworkBuilder.")

    # First, check for any missing shapes in input_shapes.
    for inp_name, shape in input_shapes.items():
        ##if any(d is None for d in shape):
        if len(shape) != int:

            logging.info(f"Input '{inp_name}' has incomplete shape {shape}.")
            shape = prompt_for_missing_dimensions(shape, inp_name)
            input_shapes[inp_name] = shape

    # Similarly for outputs (though this is less common)
    for out_name, shape in output_shapes.items():
        if any(d is None for d in shape):
            logging.info(f"Output '{out_name}' has incomplete shape {shape}.")
            shape = prompt_for_missing_dimensions(shape, out_name)
            output_shapes[out_name] = shape

    # Construct input_features and output_features for the builder
    # For simplicity, assume we treat all inputs/outputs as multi-array of float
    # shape in format: [batch, C, H, W] or [C, H, W], etc.
    input_features = []
    for inp_name, shape in input_shapes.items():
        if len(shape) == 0:
            # If truly unknown, prompt user for dimension count
            logging.warning(
                f"Input '{inp_name}' has no shape info. Defaulting to (1,)."
            )
            shape = [1]
            input_shapes[inp_name] = shape
        input_features.append(
            (inp_name, ct.models.datatypes.Array(*shape))
        )

    output_features = []
    for out_name, shape in output_shapes.items():
        if len(shape) == 0:
            logging.warning(
                f"Output '{out_name}' has no shape info. Defaulting to (1,)."
            )
            shape = [1]
            output_shapes[out_name] = shape
        output_features.append(
            (out_name, ct.models.datatypes.Array(*shape))
        )

    # Initialize the NeuralNetworkBuilder
    builder = NeuralNetworkBuilder(
        input_features=input_features,
        output_features=output_features,
        disable_rank5_shape_mapping=True
    )

    # Maintain a mapping of ONNX tensor names to Core ML layer outputs
    # so we know how to chain layers properly.
    tensor_coreml_map = {}
    constants_loaded = set()  # track constants we've explicitly loaded

    # The inputs map directly to the model inputs.
    for inp_name, _ in input_features:
        tensor_coreml_map[inp_name] = inp_name

    def load_constant_if_needed(const_name):
        """
        If const_name is in constants_map and not already loaded,
        create a load_constant_nd layer so it's available as a tensor
        in the NN spec. Then mark it as loaded.
        Returns the name we should use in the builder as input_name.
        """
        if const_name in constants_map and const_name not in constants_loaded:
            value = constants_map[const_name]
            load_layer_name = f"load_constant_{const_name}"
            # If shape is scalar, use [1], otherwise use value.shape
            shape_list = [1] if value.shape == () else list(value.shape)

            builder.add_load_constant_nd(
                name=load_layer_name,
                output_name=const_name,
                constant_value=value,
                shape=shape_list
            )
            constants_loaded.add(const_name)
        return const_name
    # Helper function to get coreml input list from ONNX node input names
    def get_coreml_input_names(onnx_input_names):
        """
        Return a list of the mapped Core ML tensor names (the outputs of previous layers
        or model inputs) that correspond to these ONNX input names.
        """

        def get_coreml_input_names(onnx_input_names):
            ml_inputs = []
            for name in onnx_input_names:
                if name in tensor_coreml_map:
                    ml_inputs.append(tensor_coreml_map[name])
                elif name in initializers:
                    # weight/bias
                    ml_inputs.append(name)
                elif name in constants_map:
                    # This is a constant node output => load if needed
                    loaded_name = load_constant_if_needed(name)
                    ml_inputs.append(loaded_name)
                else:
                    logging.warning(f"Unknown input '{name}'. Passing through.")
                    ml_inputs.append(name)
            return ml_inputs

    # (Optional) Debug: Print mapping before we begin
    logging.info("Initial Tensor->CoreML Map: {}".format(tensor_coreml_map))

    # Iterate over nodes in topological order
    for node_idx, node in enumerate(nodes_info):
        op_type = node['op_type']
        # If no node name, fall back to op+index
        node_name = node['name'] if node['name'] else f"{op_type}_{node_idx}"
        node_inputs = node['inputs']
        node_outputs = node['outputs']
        attrs = node['attributes']

        logging.info(f"Processing node #{node_idx}: {op_type} ({node_name})")

        # Prepare Core ML input for the builder
        coreml_inputs = get_coreml_input_names(node_inputs)

        # Create the Core ML output names for each ONNX output
        # and store them in our tensor->Core ML map
        coreml_output_names = []
        for out_name in node_outputs:
            mapped_name = f"{node_name}_out_{out_name}"
            coreml_output_names.append(mapped_name)
            tensor_coreml_map[out_name] = mapped_name

        if op_type == 'Conv':
            # (Same as in your code)
            strides = attrs.get('strides', [1, 1])
            pads = attrs.get('pads', [0, 0, 0, 0])
            kernel_shape = attrs.get('kernel_shape', [1, 1])
            group = attrs.get('group', 1)

            weight_name = node_inputs[1] if len(node_inputs) > 1 else None
            bias_name = node_inputs[2] if len(node_inputs) > 2 else None

            if weight_name not in initializers:
                logging.error(f"Missing convolution weights: {weight_name}")
                continue
            W = initializers[weight_name]
            b = initializers[bias_name] if bias_name and bias_name in initializers else None

            output_channels = W.shape[0]
            kernel_channels = W.shape[1]
            k_height, k_width = kernel_shape

            builder.add_convolution(
                name=node_name,
                kernel_channels=kernel_channels,
                output_channels=output_channels,
                height=k_height,
                width=k_width,
                stride_height=strides[0],
                stride_width=strides[1],
                border_mode='valid',
                groups=group,
                W=W,
                b=b,
                has_bias=(b is not None),
                is_deconv=False,
                output_shape=None,
                input_names=[coreml_inputs[0]],
                output_names=coreml_output_names
            )
            # optional: handle pads if not zero
            # ...

        elif op_type in ['Relu', 'Sigmoid', 'Tanh']:
            # (Same as in your code)
            if op_type == 'Relu':
                builder.add_activation(
                    name=node_name,
                    non_linearity='RELU',
                    input_name=coreml_inputs[0],
                    output_name=coreml_output_names[0]
                )
            elif op_type == 'Sigmoid':
                builder.add_activation(
                    name=node_name,
                    non_linearity='SIGMOID',
                    input_name=coreml_inputs[0],
                    output_name=coreml_output_names[0]
                )
            elif op_type == 'Tanh':
                builder.add_activation(
                    name=node_name,
                    non_linearity='TANH',
                    input_name=coreml_inputs[0],
                    output_name=coreml_output_names[0]
                )

        elif op_type in ['Add', 'Sub', 'Mul', 'Div']:
            if op_type == 'Add':
                mode = 'ADD'
            elif op_type == 'Sub':
                mode = 'SUB'
            elif op_type == 'Mul':
                mode = 'MULTIPLY'
            else:  # 'Div'
                mode = 'DIV'

            # For these ops, we typically need at least 2 inputs
            if len(coreml_inputs) < 2:
                logging.warning(f"{op_type} node with fewer than 2 inputs is unexpected.")
                # pass-through or skip
                if len(coreml_inputs) >= 1:
                    for idx, out_name in enumerate(coreml_output_names):
                        tensor_coreml_map[node_outputs[idx]] = coreml_inputs[0]
                continue

            if mode == 'DIV':
                # Use the new broadcastable divide API
                builder.add_divide_broadcastable(
                    name=node_name,
                    input_names=[coreml_inputs[0], coreml_inputs[1]],
                    output_name=coreml_output_names[0]
                )

            elif mode == 'SUB':

                builder.add_subtract_broadcastable(
                    name=node_name,
                    input_names=[coreml_inputs[0], coreml_inputs[1]],
                    output_name=coreml_output_names[0]
                )
            else:
                # For Add, Sub, Mul, we can do it directly with add_elementwise
                builder.add_elementwise(
                    name=node_name,
                    input_names=[coreml_inputs[0], coreml_inputs[1]],
                    output_name=coreml_output_names[0],
                    mode=mode
                )
        elif op_type == 'Gather':  # 'GatherAlongAxis':

            axis = attrs.get('axis', 0)
            if len(coreml_inputs) < 2:
                logging.warning("GatherAlongAxis expects data + indices as two inputs.")
                # skip or pass-through
                continue

            data_input = coreml_inputs[0]
            indices_input = coreml_inputs[1]

            builder.add_gather_along_axis(
                name=node_name,
                input_names=[data_input, indices_input],
                output_name=coreml_output_names[0],
                axis=axis
            )
        elif op_type == 'Shape':

            if len(coreml_inputs) < 1:
                logging.warning("Shape node with no input?")
                continue

            builder.add_get_shape(
                name=node_name,
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0]
            )

        elif op_type == 'Concat':
            """
            ONNX Concat => Core ML add_concat_nd.
            We'll parse the axis from the attributes or default to 1 (channels).
            """
            axis = attrs.get('axis', 1)
            # Core ML's `add_concat_nd` merges inputs along given axis
            # For multi-input Concat, we just pass all inputs
            builder.add_concat_nd(
                name=node_name,
                input_names=coreml_inputs,
                output_name=coreml_output_names[0],
                axis=axis
            )

        elif op_type == 'Split':
            """
            ONNX Split => In Core ML, `add_split_nd`.
            We'll parse 'split' attribute for the chunk sizes, plus 'axis'.
            Each output is part of the same operation. We'll ensure each 
            ONNX output maps to the correct index in `output_names`.
            """
            axis = attrs.get('axis', 0)
            split_sizes = attrs.get('split', None)
            # If split_sizes is None, default to equal splits or prompt user
            if split_sizes is None:
                # If we have multiple outputs, we can guess equal splits, or just prompt
                if len(node_outputs) > 1:
                    logging.info("No 'split' sizes found. Attempting equal splits.")
                    # Just as an example, if the dimension is known
                    # you'd do the math. Let's default to 1 for each output:
                    split_sizes = [1] * len(node_outputs)
                else:
                    # Single output => pass-through
                    split_sizes = [1]

            # add_split_nd expects output_names = a list of separate outputs
            # we already have `coreml_output_names` for each ONNX output
            builder.add_split_nd(
                name=node_name,
                input_name=coreml_inputs[0],
                output_names=coreml_output_names,  # Must match the count
                axis=axis,
                split_sizes=split_sizes
            )

        elif op_type == 'MaxPool':
            """
            ONNX MaxPool => Core ML add_pooling(..., pooling_type='MAX').
            We'll parse kernel_shape, strides, and pads. 
            Note: ONNX pads for pooling is `[pad_top, pad_left, pad_bottom, pad_right]`.
            """
            kernel_shape = attrs.get('kernel_shape', [1, 1])
            strides = attrs.get('strides', [1, 1])
            pads = attrs.get('pads', [0, 0, 0, 0])
            builder.add_pooling(
                name=node_name,
                height=kernel_shape[0],
                width=kernel_shape[1],
                stride_height=strides[0],
                stride_width=strides[1],
                layer_type='MAX',
                padding_type='VALID',
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0]
            )
            # If pads != [0,0,0,0], you can set custom padding or 'SAME'
            # but it might require more advanced logic.

        elif op_type == 'Resize':
            """
            ONNX Resize => Typically an upsampling in Core ML. 
            Common attributes: 'mode' (nearest/linear), 'scales' or 'sizes'.
            We'll do a simple nearest/linear scaling with add_upsample or 
            add_resize_bilinear if available. 
            """
            # Let’s parse 'mode' => 'nearest' or 'linear'
            mode = attrs.get('mode', 'nearest')
            # Either 'scale' or 'scales' can appear.
            scales = None
            if 'scales' in attrs:
                # Usually a 4-element array: [1.0, 1.0, scale_h, scale_w]
                scales = attrs['scales']
            else:
                # Some models might pass it as input #1
                if len(coreml_inputs) > 1 and coreml_inputs[1] in initializers:
                    scales_arr = initializers[coreml_inputs[1]]
                    if isinstance(scales_arr, np.ndarray):
                        scales = scales_arr.tolist()

            if scales is None:
                # Prompt or fallback
                logging.warning("No scales found for Resize; defaulting to x2 for H,W.")
                scales = [1.0, 1.0, 2.0, 2.0]

            # For a 4D NCHW, typically scales = [1,1,scale_h, scale_w]
            scale_h = scales[2] if len(scales) >= 3 else 1.0
            scale_w = scales[3] if len(scales) >= 4 else 1.0

            # If the 'mode' is 'nearest', we can do builder.add_upsample(..., mode='NN')
            # If 'linear' or 'bilinear', we can do builder.add_upsample(..., mode='BILINEAR')
            # Some older versions of coremltools had separate add_resize_bilinear
            # We'll choose add_upsample for demonstration:

            coreml_mode = 'NN' if mode.lower() == 'nearest' else 'BILINEAR'
            builder.add_upsample(
                name=node_name,
                scaling_factor_h=scale_h,
                scaling_factor_w=scale_w,
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0],
                mode=coreml_mode
            )

        elif op_type == 'Reshape':
            # If second input is a shape, that might be in constants_map
            if len(coreml_inputs) >= 2:
                # The second input is presumably the shape
                shape_input = coreml_inputs[1]
                # If that shape was loaded as a constant, you can read from constants_map
                if shape_input in constants_map:
                    target_shape = constants_map[shape_input].tolist()
                    builder.add_reshape(
                        name=node_name,
                        input_name=coreml_inputs[0],
                        output_name=coreml_output_names[0],
                        target_shape=target_shape,
                        mode=0
                    )
                    continue
                    #shape_input_name = node_inputs[1] if len(node_inputs) > 1 else None
            #if shape_input_name in initializers:
            #    target_shape = initializers[shape_input_name].tolist()
            else:
                target_shape = attrs.get('shape', None)
                if target_shape is None:
                    shape_str = input(
                        "Reshape layer missing shape. Enter comma-separated dims (e.g. 1,3,224,224): "
                    )
                    target_shape = [int(x) for x in shape_str.split(',')]
            target_shape = [1 if d == -1 else d for d in target_shape]

            builder.add_reshape(
                name=node_name,
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0],
                target_shape=target_shape,
                mode=0
            )

        elif op_type == 'Transpose':
            perm = attrs.get('perm', None)
            if perm is None:
                perm_str = input(
                    f"Transpose node '{node_name}' missing 'perm'. Enter comma-separated dims (e.g. 0,2,3,1): "
                )
                perm = [int(x) for x in perm_str.split(',')]
            builder.add_permute(
                name=node_name,
                dim=perm,
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0]
            )

        elif op_type == 'LeakyRelu':
            alpha = attrs.get('alpha', 0.1)
            builder.add_activation(
                name=node_name,
                non_linearity='LEAKYRELU',
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0],
                params=[alpha]
            )

        elif op_type == 'Softmax':

            builder.add_softmax(
                name=node_name,
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0],
            )

        elif op_type == 'Slice':
            """
            also have 'add_slice_dynamic' and 'add_slice_static'

            Handle a single-axis slice using the basic add_slice(...) method.

            Suppose the ONNX node provides the following attributes:
              - 'starts': list of start indices
              - 'ends':   list of end indices
              - 'axes':   list of axes
              - 'steps':  list of strides

            Because we only have one axis to slice on, we'll assume 
            each attribute list has exactly 1 element.
            """
            starts = attrs.get('starts', [0])  # e.g. [2]
            ends = attrs.get('ends', [-1])  # e.g. [5]
            axes = attrs.get('axes', [0])  # e.g. [1]
            steps = attrs.get('steps', [1])  # e.g. [1]

            # Single-axis slice:
            axis = "channel"  # axes[0]    if len(axes)   > 0 else 0
            start_index = starts[0] if len(starts) > 0 else 0
            end_index = ends[0] if len(ends) > 0 else -1
            stride = steps[0] if len(steps) > 0 else 1

            # The end_index is exclusive. Using -1 slices until the end (like Python slicing).
            # If your model uses 0-based indexing, it's consistent with Python slicing semantics.

            builder.add_slice(
                name=node_name,
                input_name=coreml_inputs[0],
                output_name=coreml_output_names[0],
                axis=axis,
                start_index=start_index,
                end_index=end_index,
                stride=stride
            )
        else:
            # Fallback: unknown operation
            logging.warning(f"Unsupported layer type: {op_type}")
            print(f"[Interactive] For node '{node_name}' ({op_type}), choose an option:")
            print("  (s) skip layer")
            print("  (r) replace with another layer type")
            print("  (c) create a custom layer")
            choice = input("Enter choice [s/r/c]: ").strip().lower()
            if choice == 's':
                logging.info(f"Skipping layer {node_name}. Outputs will be unmapped.")
                for idx, out_name in enumerate(node_outputs):
                    if len(coreml_inputs) > 0:
                        tensor_coreml_map[out_name] = coreml_inputs[0]
                    else:
                        tensor_coreml_map[out_name] = out_name
            elif choice == 'r':
                new_op = input("Enter the new operation type (e.g., 'relu'): ").strip().lower()
                if new_op == 'relu':
                    builder.add_activation(
                        name=node_name,
                        non_linearity='RELU',
                        input_name=coreml_inputs[0],
                        output_name=coreml_output_names[0]
                    )
                else:
                    logging.warning(f"Unknown replacement op '{new_op}', skipping.")
                    for idx, out_name in enumerate(node_outputs):
                        if len(coreml_inputs) > 0:
                            tensor_coreml_map[out_name] = coreml_inputs[0]
                        else:
                            tensor_coreml_map[out_name] = out_name

            elif choice == 'c':
                # Create a custom layer
                custom_name = f"Custom_{node_name}"
                logging.info(f"Creating custom layer placeholder '{custom_name}'.")
                builder.add_activation(
                    name=custom_name,
                    non_linearity='LINEAR',
                    input_name=coreml_inputs[0],
                    output_name=coreml_output_names[0]
                )
            else:
                logging.warning("Invalid choice, skipping layer with pass-through.")
                for idx, out_name in enumerate(node_outputs):
                    if len(coreml_inputs) > 0:
                        tensor_coreml_map[out_name] = coreml_inputs[0]
                    else:
                        tensor_coreml_map[out_name] = out_name

        # (Optional) Debug: Print out updated mapping after each node
        logging.info(f"Updated mapping after node {node_name}:")
        for k, v in tensor_coreml_map.items():
            logging.info(f"  {k} => {v}")

    # Return the builder.spec so we can save it as .mlmodel
    return builder.spec


def main(input_onnx: str = None, output_mlmodel: str = None):
    """
    Main function to perform ONNX -> Core ML conversion.

    :param input_onnx: Path to the ONNX model file.
    :param output_mlmodel: Path to the output .mlmodel file.
    """
    if input_onnx is None or output_mlmodel is None:
        print("Error: Please provide both input ONNX path and output .mlmodel path.")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # 1. Load ONNX model
    onnx_model = load_onnx_model(input_onnx)

    # 2. Infer shapes
    onnx_model_inferred = infer_shapes(onnx_model)

    # 3. Parse ONNX
    nodes_info, initializers, input_shapes, output_shapes, constants_map = parse_onnx_model(onnx_model_inferred)

    # 4. Build Core ML Model
    spec = build_coreml_model(nodes_info, initializers, input_shapes, output_shapes)

    # 5. Save the final model
    logging.info(f"Saving converted model to '{output_mlmodel}'")
    ct.models.utils.save_spec(spec, output_mlmodel)
    logging.info("Model conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to Core ML model layer by layer.")
    parser.add_argument("input_onnx", type=str, help="Path to the ONNX model file.")
    parser.add_argument("output_mlmodel", type=str, help="Path to the output .mlmodel file.")
    args = parser.parse_args()

    main(args.input_onnx, args.output_mlmodel)

# CoreML Maker . Convert an ONNX model by translating operations to build a CoreML model, layer per layer.
# coremlmaker.py
# 2025 - Quet Almahdi Morris
# github.com/oil3