# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnxx
from onnxx.backend.test.case.base import Base
from onnxx.backend.test.case.model import expect


class SingleRelu(Base):
    @staticmethod
    def export() -> None:
        node = onnxx.helper.make_node("Relu", ["x"], ["y"], name="test")
        graph = onnxx.helper.make_graph(
            nodes=[node],
            name="SingleRelu",
            inputs=[
                onnxx.helper.make_tensor_value_info("x", onnxx.TensorProto.FLOAT, [1, 2])
            ],
            outputs=[
                onnxx.helper.make_tensor_value_info("y", onnxx.TensorProto.FLOAT, [1, 2])
            ],
        )
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 9)],
        )

        x = np.random.randn(1, 2).astype(np.float32)
        y = np.maximum(x, 0)

        expect(model, inputs=[x], outputs=[y], name="test_single_relu_model")
