# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnxx.reference.ops._op import OpRunUnary


class IsNaN(OpRunUnary):
    def _run(self, data):  # type: ignore
        return (np.isnan(data),)
