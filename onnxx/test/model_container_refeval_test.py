# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import numpy.testing as npt

import onnxx
import onnxx.helper
import onnxx.model_container
import onnxx.numpy_helper
import onnxx.reference


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
                (np.arange(9) * 100).astype(np.float32).reshape((-1, 3)),
                name="B",
            ),
            onnxx.numpy_helper.from_array(
                (np.arange(9) + 10).astype(np.float32).reshape((-1, 3)),
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


class TestLargeOnnxReferenceEvaluator(unittest.TestCase):
    def common_check_reference_evaluator(self, container):
        X = np.arange(9).astype(np.float32).reshape((-1, 3))
        ref = onnxx.reference.ReferenceEvaluator(container)
        got = ref.run(None, {"X": X})
        expected = np.array(
            [
                [945000, 1015200, 1085400],
                [2905200, 3121200, 3337200],
                [4865400, 5227200, 5589000],
            ],
            dtype=np.float32,
        )
        npt.assert_allclose(expected, got[0])  # type: ignore[index]

    def test_large_onnx_no_large_initializer(self):
        model_proto = _linear_regression()
        large_model = onnxx.model_container.make_large_model(model_proto.graph)
        self.common_check_reference_evaluator(large_model)
        with self.assertRaises(ValueError):
            large_model["#anymissingkey"]

        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnxx")
            large_model.save(filename)
            copy = onnxx.model_container.ModelContainer()
            copy.load(filename)
            self.common_check_reference_evaluator(copy)

    def test_large_one_weight_file(self):
        large_model = _large_linear_regression()
        self.common_check_reference_evaluator(large_model)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnxx")
            large_model.save(filename, True)
            copy = onnxx.model_container.ModelContainer()
            copy.load(filename)
            loaded_model = onnxx.load_model(filename, load_external_data=True)
            self.common_check_reference_evaluator(loaded_model)

    def test_large_multi_files(self):
        large_model = _large_linear_regression()
        self.common_check_reference_evaluator(large_model)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnxx")
            large_model.save(filename, False)
            copy = onnxx.load_model(filename)
            self.common_check_reference_evaluator(copy)
            loaded_model = onnxx.load_model(filename, load_external_data=True)
            self.common_check_reference_evaluator(loaded_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
