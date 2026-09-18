#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a dummy Llama-3.2-1B model + tokenizer for exercising the NPU
reconfiguration examples without the gated meta-llama download.

Weight VALUES are random-but-small (finite, non-degenerate) -- generated text is
garbage, but that is irrelevant to compiling and dispatching the reconfiguration
ELFs and to timing them (tokens/second). Shapes and names mirror the harness
LlamaConfig exactly, and the tokenizer is a self-contained byte-level tiktoken
BPE so any prompt encodes. Only torch + safetensors are needed (see
requirements_examples.txt). Deterministic (seed 0).

    python make_dummy_model.py [out_dir]        # default: ./llama-dummy
    # -> <out_dir>/model.safetensors, <out_dir>/tokenizer.model

Then run against it, e.g.:
    python reconfig_correctness.py <out_dir>/model.safetensors <out_dir>/tokenizer.model \\
        --reconfig-method ctrlpkt_parallel --out logits.npy
"""

import argparse
import base64
import os

import torch
from safetensors.torch import save_file

# LlamaConfig (hardcoded in llama_inference_harness.py).
VOCAB = 128256
EMB = 2048
N_LAYERS = 16
N_HEADS = 32
N_KV = 8
HEAD_DIM = EMB // N_HEADS  # 64
HIDDEN = 8192


def _w(*shape):
    # Small normal weights in bf16; std 0.02 keeps activations finite through 16 layers.
    return (torch.randn(*shape) * 0.02).to(torch.bfloat16)


def make_weights(out_dir):
    tensors = {}
    tensors["model.embed_tokens.weight"] = _w(VOCAB, EMB)  # tied to the output head
    tensors["model.norm.weight"] = torch.ones(EMB, dtype=torch.bfloat16)
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = torch.ones(EMB, dtype=torch.bfloat16)
        tensors[p + "post_attention_layernorm.weight"] = torch.ones(
            EMB, dtype=torch.bfloat16
        )
        tensors[p + "self_attn.q_proj.weight"] = _w(N_HEADS * HEAD_DIM, EMB)
        tensors[p + "self_attn.k_proj.weight"] = _w(N_KV * HEAD_DIM, EMB)
        tensors[p + "self_attn.v_proj.weight"] = _w(N_KV * HEAD_DIM, EMB)
        tensors[p + "self_attn.o_proj.weight"] = _w(EMB, N_HEADS * HEAD_DIM)
        tensors[p + "mlp.gate_proj.weight"] = _w(HIDDEN, EMB)
        tensors[p + "mlp.up_proj.weight"] = _w(HIDDEN, EMB)
        tensors[p + "mlp.down_proj.weight"] = _w(EMB, HIDDEN)

    path = os.path.join(out_dir, "model.safetensors")
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"{len(tensors)} tensors, {total / 1e9:.2f} GB -> {path}")
    save_file(tensors, path)


def make_tokenizer(out_dir):
    # Byte-level tiktoken BPE ranks: 256 single bytes + unique fillers to 128000
    # (special tokens 128000..128255 are added by the harness Encoding, not here).
    # Every byte is present, so any prompt encodes. Format: "<base64> <rank>".
    n_ranks = 128000
    lines = []
    for r in range(n_ranks):
        if r < 256:
            tok = bytes([r])
        else:
            # 0xFF marker (distinct from any single byte) + 24-bit rank => unique.
            tok = bytes([0xFF, (r >> 16) & 0xFF, (r >> 8) & 0xFF, r & 0xFF])
        lines.append(f"{base64.b64encode(tok).decode()} {r}")
    path = os.path.join(out_dir, "tokenizer.model")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"tokenizer.model: {n_ranks} ranks -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", default="llama-dummy")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(0)
    st = os.path.join(args.out_dir, "model.safetensors")
    if os.path.exists(st) and os.path.getsize(st) > 2_000_000_000:
        print(
            f"model.safetensors already present ({os.path.getsize(st) / 1e9:.2f} GB), skipping"
        )
    else:
        make_weights(args.out_dir)
    make_tokenizer(args.out_dir)


if __name__ == "__main__":
    main()
