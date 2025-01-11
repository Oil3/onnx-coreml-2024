# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnxx
from onnxx.backend.test.case.base import Base
from onnxx.backend.test.case.node import expect


class EyeLike(Base):
    @staticmethod
    def export_without_dtype() -> None:
        shape = (4, 4)
        node = onnxx.helper.make_node(
            "EyeLike",
            inputs=["x"],
            outputs=["y"],
        )

        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.eye(shape[0], shape[1], dtype=np.int32)
        expect(node, inputs=[x], outputs=[y], name="test_eyelike_without_dtype")

    @staticmethod
    def export_with_dtype() -> None:
        shape = (3, 4)
        node = onnxx.helper.make_node(
            "EyeLike",
            inputs=["x"],
            outputs=["y"],
            dtype=onnxx.TensorProto.DOUBLE,
        )

        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.eye(shape[0], shape[1], dtype=np.float64)
        expect(node, inputs=[x], outputs=[y], name="test_eyelike_with_dtype")

    @staticmethod
    def export_populate_off_main_diagonal() -> None:
        shape = (4, 5)
        off_diagonal_offset = 1
        node = onnxx.helper.make_node(
            "EyeLike",
            inputs=["x"],
            outputs=["y"],
            k=off_diagonal_offset,
            dtype=onnxx.TensorProto.FLOAT,
        )

        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_eyelike_populate_off_main_diagonal",
        )
