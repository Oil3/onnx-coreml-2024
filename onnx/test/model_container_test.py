# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import onnxx
import onnxx.external_data_helper as ext_data
import onnxx.helper
import onnxx.model_container
import onnxx.numpy_helper


def _linear_regression():
    X = onnxx.helper.make_tensor_value_info("X", onnxx.TensorProto.FLOAT, [None, None])
    Y = onnxx.helper.make_tensor_value_info("Y", onnxx.TensorProto.FLOAT, [None])
    graph = onnxx.helper.make_graph(
        [
            onnxx.helper.make_node("MatMul", ["X", "A"], ["XA"]),
            onnxx.helper.make_node("MatMul", ["XA", "B"], ["XB"]),
            onnxx.helper.make_node("MatMul", ["XB", "C"], ["Y"]),
        ],
        "mm",
        [X],
        [Y],
        [
            onnxx.numpy_helper.from_array(
                np.arange(9).astype(np.float32).reshape((-1, 3)), name="A"
            ),
            onnxx.numpy_helper.from_array(
                (np.arange(9) * 10).astype(np.float32).reshape((-1, 3)),
                name="B",
            ),
            onnxx.numpy_helper.from_array(
                (np.arange(9) * 10).astype(np.float32).reshape((-1, 3)),
                name="C",
            ),
        ],
    )
    onnx_model = onnxx.helper.make_model(graph)
    onnxx.checker.check_model(onnx_model)
    return onnx_model


def _large_linear_regression():
    X = onnxx.helper.make_tensor_value_info("X", onnxx.TensorProto.FLOAT, [None, None])
    Y = onnxx.helper.make_tensor_value_info("Y", onnxx.TensorProto.FLOAT, [None])
    graph = onnxx.helper.make_graph(
        [
            onnxx.helper.make_node("MatMul", ["X", "A"], ["XA"]),
            onnxx.helper.make_node("MatMul", ["XA", "B"], ["XB"]),
            onnxx.helper.make_node("MatMul", ["XB", "C"], ["Y"]),
        ],
        "mm",
        [X],
        [Y],
        [
            onnxx.model_container.make_large_tensor_proto(
                "#loc0", "A", onnxx.TensorProto.FLOAT, (3, 3)
            ),
            onnxx.numpy_helper.from_array(
                np.arange(9).astype(np.float32).reshape((-1, 3)), name="B"
            ),
            onnxx.model_container.make_large_tensor_proto(
                "#loc1", "C", onnxx.TensorProto.FLOAT, (3, 3)
            ),
        ],
    )
    onnx_model = onnxx.helper.make_model(graph)
    large_model = onnxx.model_container.make_large_model(
        onnx_model.graph,
        {
            "#loc0": (np.arange(9) * 100).astype(np.float32).reshape((-1, 3)),
            "#loc1": (np.arange(9) + 10).astype(np.float32).reshape((-1, 3)),
        },
    )
    large_model.check_model()
    return large_model


class TestLargeOnnx(unittest.TestCase):
    def test_large_onnx_no_large_initializer(self):
        model_proto = _linear_regression()
        assert isinstance(model_proto, onnxx.ModelProto)
        large_model = onnxx.model_container.make_large_model(model_proto.graph)
        assert isinstance(large_model, onnxx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnxx")
            large_model.save(filename)
            copy = onnxx.model_container.ModelContainer()
            with self.assertRaises(RuntimeError):
                assert copy.model_proto
            copy.load(filename)
            assert copy.model_proto is not None
            onnxx.checker.check_model(copy.model_proto)

    def test_large_one_weight_file(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnxx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnxx")
            saved_proto = large_model.save(filename, True)
            assert isinstance(saved_proto, onnxx.ModelProto)
            copy = onnxx.model_container.ModelContainer()
            copy.load(filename)
            copy.check_model()
            loaded_model = onnxx.load_model(filename, load_external_data=True)
            onnxx.checker.check_model(loaded_model)

    def test_large_multi_files(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnxx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnxx")
            saved_proto = large_model.save(filename, False)
            assert isinstance(saved_proto, onnxx.ModelProto)
            copy = onnxx.load_model(filename)
            onnxx.checker.check_model(copy)
            for tensor in ext_data._get_all_tensors(copy):
                if ext_data.uses_external_data(tensor):
                    tested = 0
                    for ext in tensor.external_data:
                        if ext.key == "location":  # type: ignore[attr-defined]
                            assert os.path.exists(ext.value)
                            tested += 1
                    self.assertEqual(tested, 1)
            loaded_model = onnxx.load_model(filename, load_external_data=True)
            onnxx.checker.check_model(loaded_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
