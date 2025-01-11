# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import onnxx
import onnxx.onnx_cpp2py_export.printer as C  # noqa: N812


def to_text(proto: onnxx.ModelProto | onnxx.FunctionProto | onnxx.GraphProto) -> str:
    if isinstance(proto, onnxx.ModelProto):
        return C.model_to_text(proto.SerializeToString())
    if isinstance(proto, onnxx.FunctionProto):
        return C.function_to_text(proto.SerializeToString())
    if isinstance(proto, onnxx.GraphProto):
        return C.graph_to_text(proto.SerializeToString())
    raise TypeError("Unsupported argument type.")
