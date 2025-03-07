from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pdb
import os
from onnxx import onnx_pb
from _converter import convert
from typing import Text, IO
#from nnBuilderMethods import method_registry
# coremltoolsx and onnxx are modified versions renamed to prevent conflicts with or from original versions
# '/usr/bin/python3 convert.py' with macOS' stock Python from /usr/bin/python3 (or a version 3.9)
#
#
def onnx_to_coreml(onnx_model_path: str, output_path: str = None) -> None:
    """
    Convert an ONNX model to CoreML model.

    Args:
        onnx_model_path (str): Input path to the ONNX model file.
        output_path (str): Output path for the CoreML .mlmodel file. Defaults to None for same as inputPath. Or.mlpackage.
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.splitext(onnx_model_path)[0] + ".mlmodel"

    print(f"Loading ONNX model from: {onnx_model_path}")
    with open(onnx_model_path, 'rb') as onnx_model_file:
        onnx_model_proto = onnx_pb.ModelProto()
        onnx_model_proto.ParseFromString(onnx_model_file.read())

    print(f"Converting ONNX model to CoreML...")
    coreml_model = convert(onnx_model_proto)

    print(f"Saving CoreML model to: {output_path}")
    coreml_model.save(output_path)
    print("Conversion complete.")


if __name__ == '__main__':
    import sys
   # pdb.set_trace()

    if len(sys.argv) > 1:
        onnx_model_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        onnx_model_path = input("Input path of ONNX model ? ").strip()
        output_path = input(
            "Output path for the CoreML file? (default: same as input): ").strip() or None

    # Run the conversion
    if not os.path.isfile(onnx_model_path):
        print(f"File not found: {onnx_model_path}")
        sys.exit(1)

    onnx_to_coreml(onnx_model_path, output_path)


