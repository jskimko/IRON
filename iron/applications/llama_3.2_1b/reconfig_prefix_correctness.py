#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental-prefix correctness for the real llama reconfiguration chain.

Runs llama decode's first N ops on the NPU (the reconfiguration chain, under a
chosen reconfig_method) and checks the prefix output against an INDEPENDENT CPU
reference -- each op's ``.reference()`` chained via the OperatorSequence
``reference`` dispatch. Cross-method bit-identity only proves the delivery
methods agree; this proves the chain is actually correct against ground truth.

The reference is seeded identically to the NPU run by copying every buffer's host
content (weights + inputs) from the NPU callable BEFORE the NPU runs, so the CPU
chain computes its own intermediates rather than reusing the NPU's.

N=1 (RMSNorm) is the first step: static methods only (a single configure block is
odd, which the union dispatcher rejects); reconfiguration begins at N=2.

Usage:
    python reconfig_prefix_correctness.py W.safetensors tok.model -N 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import llama_inference_harness as harness  # noqa: E402
from llama_npu import AIELlamaOperators, max_seq_len  # noqa: E402
from iron.common.context import AIEContext  # noqa: E402
from iron.common.sequence import OperatorSequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights_path")
    ap.add_argument("tokenizer_path")
    ap.add_argument(
        "--reconfig-method",
        default=None,
        choices=[None, "loadpdi", "loadpdi_union", "write32", "ctrlpkt"],
    )
    ap.add_argument("-N", "--prefix-n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rel-tol", type=float, default=0.04)
    ap.add_argument("--abs-tol", type=float, default=1e-2)
    args = ap.parse_args()

    config = harness.LlamaConfig(args.weights_path, args.tokenizer_path)
    ops = AIELlamaOperators(
        config,
        max_seq_len,
        reconfig_method=args.reconfig_method,
        decode_prefix_n=args.prefix_n,
    )
    fo = ops.decode.fused_op
    cb = ops.decode.fused
    out_name = fo.output_args[0]
    last_op = fo.runlist[-1][0].__class__.__name__

    # Seed the changing inputs deterministically (weights are already loaded).
    torch.manual_seed(args.seed)
    for name in fo.input_args:
        view = cb.get_buffer(name).torch_view()
        view[:] = torch.randn(view.numel(), dtype=torch.float32).to(torch.bfloat16)

    # Build the independent CPU reference (same prefix, reference dispatch) and seed
    # it from the NPU callable's host buffers (weights + inputs) BEFORE the NPU runs.
    ref_seq = OperatorSequence(
        "fused_op_ref",
        fo.runlist,
        input_args=fo.input_args,
        output_args=fo.output_args,
        buffer_sizes=fo.explicit_buffer_sizes,
        dispatch="reference",
        context=AIEContext(),
    ).compile()
    ref_cb = ref_seq.get_callable()
    for name in fo.subbuffer_layout:
        ref_cb.get_buffer(name).torch_view()[:] = (
            cb.get_buffer(name).to_torch().flatten()
        )

    # NPU reconfiguration-dispatched prefix.
    cb.input_buffer.to("npu")
    if cb.params is not None:  # only when the prefix consumes runtime params
        cb.params.write("cache_offset", np.int32(0))
        cb.params.write("softmax_vector_size", np.int32(1))
        cb.params.sync()
    cb()
    npu_out = cb.get_buffer(out_name).to_torch().flatten().to(torch.float32)

    # CPU reference chain.
    ref_cb()
    ref_out = ref_cb.get_buffer(out_name).to_torch().flatten().to(torch.float32)

    n = min(npu_out.numel(), ref_out.numel())
    diff = (npu_out[:n] - ref_out[:n]).abs()
    max_abs = float(diff.max())
    max_rel = float((diff / (ref_out[:n].abs() + 1e-6)).max())
    passed = (diff <= (args.abs_tol + args.rel_tol * ref_out[:n].abs())).all().item()
    print(
        f"[N={args.prefix_n} method={args.reconfig_method} out={out_name} "
        f"last_op={last_op}] max_abs={max_abs:.4g} max_rel={max_rel:.4g} "
        f"ref_peak={float(ref_out[:n].abs().max()):.4g} "
        f"PASS={passed}"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
