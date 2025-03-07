# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnxx
from onnxx.backend.test.case.base import Base
from onnxx.backend.test.case.model import expect


class ShrinkTest(Base):
    @staticmethod
    def export() -> None:
        node = onnxx.helper.make_node(
            "Shrink",
            ["x"],
            ["y"],
            lambd=1.5,
            bias=1.5,
        )
        graph = onnxx.helper.make_graph(
            nodes=[node],
            name="Shrink",
            inputs=[
                onnxx.helper.make_tensor_value_info("x", onnxx.TensorProto.FLOAT, [5])
            ],
            outputs=[
                onnxx.helper.make_tensor_value_info("y", onnxx.TensorProto.FLOAT, [5])
            ],
        )
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 10)],
        )

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y = np.array([-0.5, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

        expect(model, inputs=[x], outputs=[y], name="test_shrink")
