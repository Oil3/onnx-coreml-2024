#!/usr/bin/env python3
# preprocess_onnx.py - Duplicates initializers across Identity nodes

import onnx
from onnx import numpy_helper
import numpy as np
import argparse
import os


def preprocess_onnx_model(input_path, output_path, verbose=True):
    """
    Preprocess an ONNX model to duplicate initializers across Identity nodes.
    
    This solves the CoreML conversion issue where CoreML's internal layers
    (like Conv_w_transpose) try to access weights by their original name,
    even when they've been remapped through Identity nodes.
    
    Args:
        input_path (str): Path to the input ONNX model
        output_path (str): Path to save the preprocessed ONNX model
        verbose (bool): Whether to print verbose information
    """
    if verbose:
        print(f"Loading ONNX model from {input_path}")
    
    # Load the model
    model = onnx.load(input_path)
    
    # Get all initializers and their names
    initializers = {init.name: init for init in model.graph.initializer}
    
    # Track Identity nodes that connect to initializers
    identity_map = {}  # Maps: output_name -> input_name
    initializer_identity_map = {}  # Maps: output_name -> input_name for initializers
    
    # Find all Identity nodes connecting initializers
    for node in model.graph.node:
        if node.op_type == 'Identity' and len(node.input) > 0 and len(node.output) > 0:
            input_name = node.input[0]
            output_name = node.output[0]
            identity_map[output_name] = input_name
            
            # Check if this Identity connects to an initializer
            if input_name in initializers:
                initializer_identity_map[output_name] = input_name
                if verbose:
                    print(f"Identity node connects initializer: {input_name} -> {output_name}")
    
    # Check if we found any Identity nodes connecting to initializers
    if len(initializer_identity_map) == 0:
        if verbose:
            print("No Identity nodes connecting to initializers found. Model remains unchanged.")
        onnx.save(model, output_path)
        return False
    
    # Create new initializers with the output names
    new_initializers = []
    for output_name, input_name in initializer_identity_map.items():
        # Get the original initializer
        init = initializers[input_name]
        
        # Instead of creating a tensor from scratch, duplicate the initializer
        # and change only its name
        new_init = onnx.TensorProto()
        new_init.CopyFrom(init)
        new_init.name = output_name
        
        new_initializers.append(new_init)
        if verbose:
            print(f"Created new initializer: {output_name} (copied from {input_name})")
    
    # Add the new initializers to the model
    for init in new_initializers:
        model.graph.initializer.append(init)
    
    # Save the modified model
    onnx.save(model, output_path)
    if verbose:
        print(f"Saved preprocessed model to {output_path}")
        print(f"Added {len(new_initializers)} duplicate initializers to address Identity node remapping.")
    
    return True


def main():
    """Command line interface for preprocessing ONNX models"""
    parser = argparse.ArgumentParser(description='Preprocess ONNX model for CoreML conversion')
    parser.add_argument('input', help='Path to the input ONNX model')
    parser.add_argument('-o', '--output', help='Path to save the preprocessed ONNX model (default: input_preprocessed.onnx)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        basename, ext = os.path.splitext(args.input)
        args.output = f"{basename}_preprocessed{ext}"
    
    # Preprocess the model
    preprocess_onnx_model(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main() 