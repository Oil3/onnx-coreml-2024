# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnxx
from onnxx.backend.test.case.base import Base
from onnxx.backend.test.case.node import expect
from onnxx.reference.ops.aionnxml.op_binarizer import compute_binarizer


class Binarizer(Base):
    @staticmethod
    def export() -> None:
        threshold = 1.0
        node = onnxx.helper.make_node(
            "Binarizer",
            inputs=["X"],
            outputs=["Y"],
            threshold=threshold,
            domain="ai.onnxx.ml",
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = compute_binarizer(x, threshold)[0]

        expect(node, inputs=[x], outputs=[y], name="test_ai_onnx_ml_binarizer")
