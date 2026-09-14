# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

from ml_dtypes import bfloat16

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils


@dataclass
class StridedCopy(MLIROperator):
    """AIE-accelerated strided copy operator"""

    input_sizes: list
    input_strides: list
    input_offset: int
    output_sizes: list
    output_strides: list
    output_offset: int
    input_buffer_size: int = field(repr=False)
    output_buffer_size: int = field(repr=False)
    dtype: object = field(default=bfloat16, repr=False)
    transfer_size: int | None = None
    num_aie_channels: int = 1
    input_offset_parameter: str | None = None
    output_offset_parameter: str | None = None
    kwargs: dict = field(default_factory=dict, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "input_sizes": "isz",
        "input_strides": "ist",
        "input_offset": "ioff",
        "output_sizes": "osz",
        "output_strides": "ost",
        "output_offset": "ooff",
        "transfer_size": "tr",
        "num_aie_channels": "ch",
        "input_offset_parameter": "ipar",
        "output_offset_parameter": "opar",
    }

    def __post_init__(self):
        if len(self.input_sizes) != len(self.input_strides):
            raise ValueError(
                f"input_sizes and input_strides must have the same length "
                f"({len(self.input_sizes)} vs {len(self.input_strides)})"
            )
        if len(self.output_sizes) != len(self.output_strides):
            raise ValueError(
                f"output_sizes and output_strides must have the same length "
                f"({len(self.output_sizes)} vs {len(self.output_strides)})"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "strided_copy",
                (
                    aie_utils.get_current_device(),
                    self.dtype,
                    self.input_buffer_size,
                    self.input_sizes,
                    self.input_strides,
                    self.input_offset,
                    self.output_buffer_size,
                    self.output_sizes,
                    self.output_strides,
                    self.output_offset,
                    self.transfer_size,
                    self.num_aie_channels,
                ),
                {
                    **self.kwargs,
                    "input_offset_parameter": self.input_offset_parameter,
                    "output_offset_parameter": self.output_offset_parameter,
                },
            ),
        )

    def get_kernel_artifacts(self):
        return []

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (int(self.input_buffer_size),), dtype=self.dtype),
            AIERuntimeArgSpec("out", (int(self.output_buffer_size),), dtype=self.dtype),
        ]

    def reference(self, input_flat):
        """CPU reference: the same gather/scatter data movement the op performs.

        StridedCopy is pure data movement, so the reference is the module-level
        gather-by-input-tap / scatter-by-output-tap over the access patterns. The
        runtime offset parameters (``input_offset_parameter`` /
        ``output_offset_parameter``, e.g. ``cache_offset``) default to their reset
        value 0 here, which is correct whenever the parameter is 0 (e.g. the
        ``num_preceding_tokens == 0`` decode step). For a non-zero runtime offset,
        call ``strided_copy.reference.reference`` with the matching
        ``*_offset_addend`` directly.
        """
        import torch

        from .reference import reference as _strided_copy_reference

        return _strided_copy_reference(
            torch.as_tensor(input_flat).reshape(-1),
            self.input_sizes,
            self.input_strides,
            self.input_offset,
            self.output_buffer_size,
            self.output_sizes,
            self.output_strides,
            self.output_offset,
            num_aie_channels=self.num_aie_channels,
        )
