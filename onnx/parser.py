# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import onnxx
import onnxx.onnx_cpp2py_export.parser as C  # noqa: N812


class ParseError(Exception):
    pass


def parse_model(model_text: str) -> onnxx.ModelProto:
    """Parse a string to build a ModelProto.

    Arguments:
        model_text (string): formatted string
    Returns:
        ModelProto
    """
    (success, msg, model_proto_str) = C.parse_model(model_text)
    if success:
        return onnxx.load_from_string(model_proto_str)
    raise ParseError(msg)


def parse_graph(graph_text: str) -> onnxx.GraphProto:
    """Parse a string to build a GraphProto.

    Arguments:
        graph_text (string): formatted string
    Returns:
        GraphProto
    """
    (success, msg, graph_proto_str) = C.parse_graph(graph_text)
    if success:
        graph_proto = onnxx.GraphProto()
        graph_proto.ParseFromString(graph_proto_str)
        return graph_proto
    raise ParseError(msg)


def parse_function(function_text: str) -> onnxx.FunctionProto:
    """Parse a string to build a FunctionProto.

    Arguments:
        function_text (string): formatted string
    Returns:
        FunctionProto
    """
    (success, msg, function_proto_str) = C.parse_function(function_text)
    if success:
        function_proto = onnxx.FunctionProto()
        function_proto.ParseFromString(function_proto_str)
        return function_proto
    raise ParseError(msg)


def parse_node(node_text: str) -> onnxx.NodeProto:
    """Parse a string to build a NodeProto.

    Arguments:
        node_text: formatted string
    Returns:
        NodeProto
    """
    (success, msg, node_proto_str) = C.parse_node(node_text)
    if success:
        node_proto = onnxx.NodeProto()
        node_proto.ParseFromString(node_proto_str)
        return node_proto
    raise ParseError(msg)
