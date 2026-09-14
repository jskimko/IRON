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
    # rope_angles must be real interleaved cos/sin (not random) or RoPE is not a
    # meaningful test; other inputs are seeded random.
    torch.manual_seed(args.seed)
    for name in fo.input_args:
        view = cb.get_buffer(name).torch_view()
        n = view.numel()
        if name == "rope_angles":
            angles = config.angles.flatten().to(torch.float32)
            reps = (n + angles.numel() - 1) // angles.numel()
            view[:] = angles.repeat(reps)[:n].to(torch.bfloat16)
        else:
            view[:] = torch.randn(n, dtype=torch.float32).to(torch.bfloat16)

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
        # A prefix declares only the params its ops consume, so write each one
        # only if present (fixed decode step: cache_offset=0, vector_size=1).
        for pname, pval in (("cache_offset", 0), ("softmax_vector_size", 1)):
            try:
                cb.params.write(pname, np.int32(pval))
            except RuntimeError:
                pass  # not declared by this prefix
        cb.params.sync()
    cb()
    npu_out = cb.get_buffer(out_name).to_torch().flatten().to(torch.float32)

    # CPU reference chain.
    ref_cb()
    ref_out = ref_cb.get_buffer(out_name).to_torch().flatten().to(torch.float32)

    n = min(npu_out.numel(), ref_out.numel())
    diff = (npu_out[:n] - ref_out[:n]).abs()
    ref_abs = ref_out[:n].abs()
    ref_peak = float(ref_abs.max())
    # bf16 has a 7-bit mantissa; one ulp at the peak magnitude is the granularity a
    # correct-but-bf16 result can differ by. A pass allows a per-element tolerance
    # (abs + rel*|ref|) OR a small fixed number of ulp-at-peak, so a few elements
    # rounding by 1-2 ulp is not flagged as a compute error.
    import math

    ulp_at_peak = 2.0 ** (math.floor(math.log2(ref_peak)) - 7) if ref_peak > 0 else 0.0
    per_elem = args.abs_tol + args.rel_tol * ref_abs
    fail_mask = diff > per_elem
    n_fail = int(fail_mask.sum())
    # A structural error touches many elements or exceeds a few ulp; rounding does not.
    passed = (n_fail == 0) or (
        float(diff.max()) <= 3 * ulp_at_peak and n_fail <= max(4, int(0.001 * n))
    )
    print(
        f"[N={args.prefix_n} method={args.reconfig_method} out={out_name} "
        f"last_op={last_op}] max_abs={float(diff.max()):.4g} "
        f"mean_abs={float(diff.mean()):.4g} "
        f"max_rel={float((diff / (ref_abs + 1e-6)).max()):.4g} "
        f"ref_peak={ref_peak:.4g} ulp@peak={ulp_at_peak:.4g} "
        f"n_fail={n_fail}/{n} PASS={passed}"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
