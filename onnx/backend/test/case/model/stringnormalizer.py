# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Sequence

import numpy as np

import onnxx
from onnxx.backend.test.case.base import Base
from onnxx.backend.test.case.model import expect


class NormalizeStrings(Base):
    @staticmethod
    def export() -> None:
        def make_graph(
            node: onnxx.helper.NodeProto,
            input_shape: Sequence[int],
            output_shape: Sequence[int],
        ) -> onnxx.helper.GraphProto:
            graph = onnxx.helper.make_graph(
                nodes=[node],
                name="StringNormalizer",
                inputs=[
                    onnxx.helper.make_tensor_value_info(
                        "x", onnxx.TensorProto.STRING, input_shape
                    )
                ],
                outputs=[
                    onnxx.helper.make_tensor_value_info(
                        "y", onnxx.TensorProto.STRING, output_shape
                    )
                ],
            )
            return graph

        # 1st model_monday_casesensintive_nochangecase
        stopwords = ["monday"]
        node = onnxx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            is_case_sensitive=1,
            stopwords=stopwords,
        )

        x = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        y = np.array(["tuesday", "wednesday", "thursday"]).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_casesensintive_nochangecase",
        )

        # 2nd model_nostopwords_nochangecase
        node = onnxx.helper.make_node(
            "StringNormalizer", inputs=["x"], outputs=["y"], is_case_sensitive=1
        )

        x = np.array(["monday", "tuesday"]).astype(object)
        y = x

        graph = make_graph(node, [2], [2])
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_nostopwords_nochangecase",
        )

        # 3rd model_monday_casesensintive_lower
        stopwords = ["monday"]
        node = onnxx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="LOWER",
            is_case_sensitive=1,
            stopwords=stopwords,
        )

        x = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        y = np.array(["tuesday", "wednesday", "thursday"]).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_casesensintive_lower",
        )

        # 4 model_monday_casesensintive_upper
        stopwords = ["monday"]
        node = onnxx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            is_case_sensitive=1,
            stopwords=stopwords,
        )

        x = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        y = np.array(["TUESDAY", "WEDNESDAY", "THURSDAY"]).astype(object)

        graph = make_graph(node, [4], [3])
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_casesensintive_upper",
        )

        # 5 monday_insensintive_upper_twodim
        stopwords = ["monday"]
        node = onnxx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            stopwords=stopwords,
        )

        input_shape = [1, 6]
        output_shape = [1, 4]
        x = (
            np.array(
                ["Monday", "tuesday", "wednesday", "Monday", "tuesday", "wednesday"]
            )
            .astype(object)
            .reshape(input_shape)
        )
        y = (
            np.array(["TUESDAY", "WEDNESDAY", "TUESDAY", "WEDNESDAY"])
            .astype(object)
            .reshape(output_shape)
        )

        graph = make_graph(node, input_shape, output_shape)
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_insensintive_upper_twodim",
        )

        # 6 monday_empty_output
        stopwords = ["monday"]
        node = onnxx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            is_case_sensitive=0,
            stopwords=stopwords,
        )

        x = np.array(["monday", "monday"]).astype(object)
        y = np.array([""]).astype(object)

        graph = make_graph(node, [2], [1])
        model = onnxx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
            opset_imports=[onnxx.helper.make_opsetid("", 10)],
        )
        expect(
            model,
            inputs=[x],
            outputs=[y],
            name="test_strnorm_model_monday_empty_output",
        )
