#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic raw-logit capture for the reconfig-method A/B/C correctness bar.

Builds the fused decode ELF under a chosen ``reconfig_method`` and runs ONE
fixed decode step (fixed token id, ``num_preceding_tokens=0`` so the runtime
params equal the baked defaults ``cache_offset=0`` / ``softmax_vector_size=1``),
then saves the raw logits vector. Delivery/packaging must not change the compute,
so the logits from every method must agree within bf16 tolerance.

Usage (one method per process, then diff the .npy files):

    python reconfig_correctness.py W.safetensors tok.model --reconfig-method write32 \
        --out /tmp/logits_write32.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import llama_inference_harness as harness  # noqa: E402
import llama_npu  # noqa: E402
from llama_npu import AIELlamaOperators, max_seq_len  # noqa: E402


def capture_logits(config, reconfig_method, token_id):
    """One fixed decode step's raw logits (float32 numpy), for a given method."""
    ops = AIELlamaOperators(config, max_seq_len, reconfig_method=reconfig_method)
    # llama_forward_pass_decode reads the module-global aie_ops.
    llama_npu.aie_ops = ops

    state = harness.LlamaModelState(config)
    state.token_ids = torch.tensor([[token_id]], dtype=torch.long)
    state.num_preceding_tokens = 0  # cache_offset=0, softmax_vector_size=1 (defaults)

    logits, _ = llama_npu.llama_forward_pass_decode(config, state)
    return logits.to(torch.float32).view(-1).contiguous().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_path")
    parser.add_argument("tokenizer_path")
    parser.add_argument(
        "--reconfig-method",
        default=None,
        choices=[None, "loadpdi", "loadpdi_union", "write32", "ctrlpkt"],
    )
    parser.add_argument("--out", required=True, help="output .npy for the logits")
    parser.add_argument("--token-id", type=int, default=42)
    args = parser.parse_args()

    config = harness.LlamaConfig(args.weights_path, args.tokenizer_path)
    logits = capture_logits(config, args.reconfig_method, args.token_id)

    np.save(args.out, logits)
    finite = np.isfinite(logits).all()
    print(
        f"[{args.reconfig_method}] saved {args.out} shape={logits.shape} "
        f"finite={finite} "
        f"min={np.nanmin(logits):.4g} max={np.nanmax(logits):.4g} "
        f"argmax={int(np.nanargmax(logits))}"
    )


if __name__ == "__main__":
    main()
