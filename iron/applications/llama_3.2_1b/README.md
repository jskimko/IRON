<!--
SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Golden Model Inference

## Model and Tokenizer Download Instructions

To download the necessary files for the model, please follow the links below:

- **Model File**: [model.safetensors](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main)
- **Tokenizer File**: [tokenizer.model](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main/original)

The CI tests expect these files to be placed in the directory `/srv/llama3.2-1b`;
when calling `inference.py`, you will supply the paths to those two files on the command line, so you can place them anywhere.

## Installation Instructions

Before running `inference.py`, ensure you have the proper environment. To build the environment from scratch, follow the instructions below:

1. Follow the IRON installation instructions in the repository root fist.
   After this, you should have an `ironenv` environment set up and activated.

2. Install the following additional requirements:
   ```
   python3 -m pip install -r requirements_examples.txt
   ```

## Running Inference

Inference with Llama-3.2-1B can be run by specifying a number of tokens to generate based on a prompt. This is done with `inference.py`:  
```bash  
cd golden_model
python inference.py /path/to/model.safetensors /path/to/tokenizer.model --num_tokens <NUM_TOKENS> --prompt <PROMPT>
```

`inference.py` has the following command format:  
```bash
python inference.py <weights_file_path> <tokenizer_file_path> [--num_tokens NUM_TOKENS] [--prompt PROMPT] [--use_prompt_template] [--save_outputs]
```

### Arguments:
- `weights_file_path`: Path to the weights file (e.g., `model.safetensors`).
- `tokenizer_file_path`: Path to the tokenizer file (e.g., `tokenizer.model`).
- `--num_tokens`: (Optional) Number of tokens to predict. Default is `1`.
- `--prompt`: (Optional) Prompt for the model to generate text from. Default is the text in `prompts.txt`.
- `--use_prompt_template`: (Optional) Use a prompt template for the model. Should be passed in when using Instruct weights.
- `--save_outputs`: (Optional) Enable hooks to save outputs of at each layer of the model.

## Running the reconfiguration examples (NPU)

`llama_npu.py` and `reconfig_correctness.py` run the fused decode on the NPU and select a runtime reconfiguration-delivery method with `--reconfig-method` (`loadpdi`, `loadpdi_union`, `write32`, `ctrlpkt`, `ctrlpkt_parallel`; omit for the expand baseline).

These require a **mlir-aie built from the control-packet reconfiguration branch** -- the mechanism (aiecc `--reconfig-method` / `--ctrlpkt-*`, `iron.Reconfiguration`) is not in the pinned public `mlir_aie` wheel. Given that toolchain built:

1. Install the example Python deps (torch CPU, numpy, tiktoken, safetensors). Do **not** also install the `mlir_aie` / `llvm-aie` wheels -- the built toolchain provides `aie`/`aiecc`/peano.
   ```bash
   python3 -m pip install -r requirements_examples.txt
   ```
2. Point at the built toolchain and source the reconfiguration env (layers its `aie` python + `aiecc` + peano, plus XRT's `pyxrt`, onto the active environment; also puts this repo on `PYTHONPATH`):
   ```bash
   export MLIR_AIE=/path/to/built/mlir-aie          # repo root: has install/ and ironenv/
   source iron/applications/llama_3.2_1b/reconfig_env.sh
   ```
3. Get a model. Exercising the mechanism or measuring tokens/second does **not** need the gated meta-llama weights -- generate a dummy model (random weights + a self-contained tokenizer). Only real *text quality* needs the real download from the section above.
   ```bash
   python iron/applications/llama_3.2_1b/make_dummy_model.py llama-dummy
   # -> llama-dummy/model.safetensors, llama-dummy/tokenizer.model
   ```
4. Run one decode step, capturing the logits (swap in the real weights/tokenizer for meaningful text):
   ```bash
   python iron/applications/llama_3.2_1b/reconfig_correctness.py \
       llama-dummy/model.safetensors llama-dummy/tokenizer.model \
       --reconfig-method ctrlpkt_parallel --out logits.npy
   ```
   Or a timed multi-token decode (prints tokens/second):
   ```bash
   python iron/applications/llama_3.2_1b/llama_npu.py \
       llama-dummy/model.safetensors llama-dummy/tokenizer.model \
       --reconfig-method ctrlpkt_parallel --num-tokens 32 --prompt-len 8
   ```

`reconfig_env.sh` honors `MLIR_AIE_INSTALL_DIR`, `PEANO_INSTALL_DIR`, and `XILINX_XRT` overrides if your layout differs.