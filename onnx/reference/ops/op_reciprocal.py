# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnxx.reference.ops._op import OpRunUnaryNum


class Reciprocal(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        with np.errstate(divide="ignore"):
            return (np.reciprocal(x).astype(x.dtype),)
