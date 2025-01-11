# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnxx
from onnxx.backend.test.case.base import Base
from onnxx.backend.test.case.model import expect


class SingleSign(Base):
    @staticmethod
    def export() -> None:
        node = onnxx.helper.make_node("Sign", ["x"], ["y"], name="test")

        x = np.array([-1.0, 4.5, -4.5, 3.1, 0.0, 2.4, -5.5]).astype(np.float32)
        y = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, -1.0]).astype(np.float32)

        graph = onnxx.helper.make_graph(
            nodes=[node],
            name="SingleSign",
            inputs=[
                onnxx.helper.make_tensor_value_info("x", onnxx.TensorProto.FLOAT, [7])
            ],
            outputs=[
                onnxx.helper.make_tensor_value_info("y", onnxx.TensorProto.FLOAT, [7])
            ],
        )
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 9)],
        )
        expect(model, inputs=[x], outputs=[y], name="test_sign_model")
