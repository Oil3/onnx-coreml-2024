import onnx
from onnx import numpy_helper
import json
import numpy as np
from pathlib import Path


def get_tensor_shape(graph, tensor_name):
    """
    Retrieve the shape and data type of a tensor by its name from the graph.
    Args:
        graph: ONNX GraphProto object.
        tensor_name: Name of the tensor.
    Returns:
        A dictionary containing 'shape' and 'dtype' of the tensor, or None if not found.
    """
    for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if value_info.name == tensor_name:
            tensor_type = value_info.type.tensor_type
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in tensor_type.shape.dim]
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_type.elem_type].name
            return {"shape": shape, "dtype": dtype}

    for initializer in graph.initializer:  # Look for initializers (e.g., weights, biases)
        if initializer.name == tensor_name:
            shape = list(initializer.dims)
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type].name
            return {"shape": shape, "dtype": dtype}

    return None
def save_weights_and_get_filenames(graph, output_dir):
    """
    Saves weights, biases, and constants from the ONNX model graph as .npy files
    and returns a mapping of tensor names to saved file paths.

    Args:
        graph: ONNX GraphProto object.
        output_dir: Directory where .npy files will be saved.

    Returns:
        A dictionary mapping tensor names to their respective .npy file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_to_file = {}

    for initializer in graph.initializer:
        tensor_name = initializer.name
        tensor_data = numpy_helper.to_array(initializer)

        # Create a unique filename
        sanitized_name = f"{tensor_name.replace('/', '_').replace('\\', '_')}"
        file_path = output_dir / f"{sanitized_name}.npy"
        np.save(file_path, tensor_data)

        tensor_to_file[tensor_name] = str(file_path)

    # Handle constants stored in node attributes
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.name == "value":
                tensor_data = numpy_helper.to_array(attr.t)

                # Ensure unique filename for constants
                sanitized_name = f"{node.name}_{attr.name}".replace("/", "_").replace("\\", "_")
                file_path = output_dir / f"{sanitized_name}.npy"
                np.save(file_path, tensor_data)

                tensor_to_file[f"{node.name}_{attr.name}"] = str(file_path)

    return tensor_to_file


def parse_onnx_model(onnx_model_path, output_file_path, save_weights=False):
    """
    Parses an ONNX model and writes detailed layer information to a file.
    Optionally saves weights and biases as .npy files.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        output_file_path (str): Path to the output file for storing layer information.
        save_weights (bool): Whether to save weights and biases as .npy files.
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph

    metadata = extract_metadata(model)

    initializer_names = {initializer.name for initializer in graph.initializer}
    model_inputs = [
        {
            "name": input_tensor.name,
            "shape": get_tensor_shape(graph, input_tensor.name)["shape"],
            "dtype": get_tensor_shape(graph, input_tensor.name)["dtype"],
        }
        for input_tensor in graph.input if input_tensor.name not in initializer_names
    ]

    model_outputs = [
        {
            "name": output_tensor.name,
            "shape": get_tensor_shape(graph, output_tensor.name)["shape"],
            "dtype": get_tensor_shape(graph, output_tensor.name)["dtype"],
        }
        for output_tensor in graph.output
    ]

    tensor_to_file = {}
    if save_weights:
        weights_dir = Path(output_file_path).parent / "weights"
        tensor_to_file = save_weights_and_get_filenames(graph, weights_dir)

    layers_info = []
    for i, node in enumerate(graph.node):
        layer_info = {
            "layer_number": i,
            "layer_name": node.output[0] if node.output else f"Unnamed_Layer_{i}",
            "layer_type": node.op_type,
            "attributes": [],
            "inputs": [],
            "outputs": [],
        }

        for attr in node.attribute:
            value = None
            file_reference = None
            if attr.type == onnx.AttributeProto.TENSOR:
                value = numpy_helper.to_array(attr.t).tolist()
                file_reference = tensor_to_file.get(f"{node.name}_{attr.name}", None)

            attribute_entry = {
                "name": attr.name,
                "type": onnx.AttributeProto.AttributeType.Name(attr.type),
                "value": value,
            }
            if file_reference:
                attribute_entry["file"] = file_reference
            layer_info["attributes"].append(attribute_entry)

        # Special handling for Reshape layers
        if node.op_type == "Reshape":
            reshape_shape_name = node.input[1] if len(node.input) > 1 else None
            if reshape_shape_name:
                reshape_tensor = tensor_to_file.get(reshape_shape_name)
                if reshape_tensor:
                    reshape_data = load_weights(reshape_tensor)
                    layer_info["attributes"].append({
                        "name": "target_shape",
                        "type": "TENSOR",
                        "value": reshape_data.tolist()
                    })

        for input_name in node.input:
            input_details = get_tensor_shape(graph, input_name)
            layer_info["inputs"].append({
                "name": input_name,
                "shape": input_details["shape"] if input_details else None,
                "dtype": input_details["dtype"] if input_details else None,
                "file": tensor_to_file.get(input_name, None)
            })

        for output_name in node.output:
            output_details = get_tensor_shape(graph, output_name)
            layer_info["outputs"].append({
                "name": output_name,
                "shape": output_details["shape"] if output_details else None,
                "dtype": output_details["dtype"] if output_details else None,
            })

        layers_info.append(layer_info)

    connections = generate_connections(graph, layers_info, model_inputs)

    output_data = {
        "metadata": metadata,
        "model_inputs": model_inputs,
        "model_outputs": model_outputs,
        "layers": layers_info,
        "connections": connections
    }

    with open(output_file_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Model information saved to {output_file_path}")

def extract_metadata(model):
    """
    Extracts all metadata properties from the ONNX model as a block.
    Args:
        model: ONNX ModelProto object.
    Returns:
        A dictionary containing all metadata properties.
    """
    metadata = {
        "ir_version": model.ir_version,
        "opset_import": [{"domain": imp.domain, "version": imp.version} for imp in model.opset_import],
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "model_version": model.model_version,
        "doc_string": model.doc_string,
        "metadata_props": {prop.key: prop.value for prop in model.metadata_props}
    }
    return metadata

def generate_connections(graph, layers_info, model_inputs):
    """
    Generate connections between layers based on inputs and outputs.

    Args:
        graph: ONNX GraphProto object.
        layers_info: List of layer dictionaries.

    Returns:
        A list of connections between layers.
    """
    # Map tensor names to their source layers
    tensor_to_layer = {output["name"]: layer["layer_name"] for layer in layers_info for output in layer["outputs"]}
    connections = []

    # Handle connections for each layer
    for layer in layers_info:
        for input_tensor in layer["inputs"]:
            source_layer = tensor_to_layer.get(input_tensor["name"])
            if source_layer:
                # Standard connection from another layer
                connections.append({
                    "source_layer": source_layer,
                    "source_output": input_tensor["name"],
                    "target_layer": layer["layer_name"],
                    "target_input": input_tensor["name"]
                })
            else:
                # Entry point connection (model input)
                for model_input in model_inputs:
                    if model_input["name"] == input_tensor["name"]:
                        connections.append({
                            "source_layer": "model_input",
                            "source_output": model_input["name"],
                            "target_layer": layer["layer_name"],
                            "target_input": input_tensor["name"]
                        })

    return connections

if __name__ == "__main__":
    onnx_model_path = input("Enter the path to the ONNX model file: ").strip()
    output_file_path = input("Enter the path to save the layer information file (JSON) [optional]: ").strip()
    save_weights_flag = "yes" #input("Do you want to save weights and biases as .npy files? (yes/no): ").strip().lower() == "yes"

    # Set default output file path if not provided
    if not output_file_path:
        output_file_path = str(Path(onnx_model_path).with_suffix(".json"))

    # Validate paths
    if not Path(onnx_model_path).is_file():
        print("The provided ONNX model path is invalid.")
    else:
        parse_onnx_model(onnx_model_path, output_file_path, save_weights=save_weights_flag)

# CoreML Maker . Convert an ONNX model by translating operations to build a CoreML model, layer per layer.
# protoparser2.py
# 2025 - Quet Almahdi Morris
# github.com/oil3