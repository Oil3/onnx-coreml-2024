# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnxx
from onnxx.backend.test.case.base import Base
from onnxx.backend.test.case.node import expect


class Constant(Base):
    @staticmethod
    def export() -> None:
        values = np.random.randn(5, 5).astype(np.float32)
        node = onnxx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["values"],
            value=onnxx.helper.make_tensor(
                name="const_tensor",
                data_type=onnxx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

        expect(node, inputs=[], outputs=[values], name="test_constant")
