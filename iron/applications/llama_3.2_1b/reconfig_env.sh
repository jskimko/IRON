#!/usr/bin/env bash
# Source this to run the llama_3.2_1b reconfiguration examples against a
# LOCALLY-BUILT mlir-aie control-packet toolchain.
#
# The reconfiguration mechanism (aiecc --reconfig-method / --ctrlpkt-* flags,
# iron.Reconfiguration) is NOT in the pinned public mlir_aie wheel, so these
# examples need a mlir-aie built from the matching control-packet branch. This
# script layers that built toolchain's aie python + aiecc + peano (and XRT's
# pyxrt) onto your active Python environment; it does NOT install anything.
#
# Usage:
#   export MLIR_AIE=/path/to/built/mlir-aie          # repo root: has install/ + ironenv/
#   source iron/applications/llama_3.2_1b/reconfig_env.sh
#   python iron/applications/llama_3.2_1b/reconfig_correctness.py \
#       model.safetensors tokenizer.model --reconfig-method ctrlpkt_parallel --out logits.npy
#
# Prereqs in your active venv (the toolchain provides aie/aiecc/peano/pyxrt, not these):
#   pip install -r requirements_examples.txt        # torch (CPU), numpy, tiktoken, safetensors
# Do NOT also pip-install the mlir_aie / llvm-aie wheels -- the built toolchain on
# PYTHONPATH/PATH is what carries the reconfiguration mechanism.
#
# Optional overrides: MLIR_AIE_INSTALL_DIR, PEANO_INSTALL_DIR, XILINX_XRT.

: "${MLIR_AIE:?export MLIR_AIE=/path/to/built/mlir-aie (repo root with install/ + ironenv/) before sourcing}"

# Built mlir-aie toolchain (aie python + aiecc).
export MLIR_AIE_INSTALL_DIR="${MLIR_AIE_INSTALL_DIR:-$MLIR_AIE/install}"
if [ ! -x "$MLIR_AIE_INSTALL_DIR/bin/aiecc" ]; then
  echo "reconfig_env: WARNING: aiecc not found under $MLIR_AIE_INSTALL_DIR/bin --" \
       "build the toolchain first (e.g. 'ninja -C build install' in mlir-aie)." >&2
fi

# Peano (llvm-aie), as installed into the mlir-aie build's ironenv.
if [ -z "${PEANO_INSTALL_DIR:-}" ]; then
  for d in "$MLIR_AIE"/ironenv/lib/python*/site-packages/llvm-aie; do
    [ -d "$d" ] && PEANO_INSTALL_DIR="$d" && break
  done
fi
export PEANO_INSTALL_DIR

# XRT (device runtime + pyxrt).
export XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"
[ -f "$XILINX_XRT/setup.sh" ] && source "$XILINX_XRT/setup.sh" >/dev/null 2>&1

# Layer the toolchain onto the active env (built aie python first, so it shadows
# any pinned mlir_aie wheel that may be installed).
export PATH="$MLIR_AIE_INSTALL_DIR/bin${PEANO_INSTALL_DIR:+:$PEANO_INSTALL_DIR/bin}:$PATH"
export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python:$XILINX_XRT/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:$XILINX_XRT/lib:${LD_LIBRARY_PATH:-}"

# Make the IRON repo importable without an editable install (resolve repo root
# from this script's location: iron/applications/llama_3.2_1b/ -> repo root).
IRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"
export PYTHONPATH="$IRON_ROOT:$PYTHONPATH"

echo "reconfig_env: mlir-aie=$MLIR_AIE_INSTALL_DIR peano=${PEANO_INSTALL_DIR:-<unset>}" \
     "xrt=$XILINX_XRT iron=$IRON_ROOT" >&2
