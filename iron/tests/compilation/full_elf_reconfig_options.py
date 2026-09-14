#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FullElf build's reconfiguration-flag selection must stay pinned per mode.

``_full_elf_reconfig_options`` is the pure option-selector behind
``OperatorSequence(reconfig_method=...)`` / ``FullElfArtifact.reconfig_method``
(see AieccFullElfCompilationRule). It maps a reconfig method to the aiecc flags
that pick a reconfiguration-delivery variant of the fused full-ELF build.

The ``None`` case is the highest-risk assertion: it is the A-expand baseline,
byte-identical to the flags the rule hardcoded before ``reconfig_method``
existed, so the whole shipped decode build depends on it not drifting. The other
modes drop ``--expand-load-pdis`` and either add nothing (native ``load_pdi``)
or ``--reconfig-method=X`` (write32/ctrlpkt). Task 2.2 reuses this same selector
for the union/runlist dispatch, so these assertions are its regression guard too.
"""

import pytest

from iron.common.compilation.base import _full_elf_reconfig_options


def test_none_is_the_a_expand_baseline():
    # A-expand: byte-identical to the pre-reconfig_method hardcoded flags.
    assert _full_elf_reconfig_options(None) == [
        "--expand-load-pdis",
        "--get-scratchpad-parameters",
    ]


def test_loadpdi_drops_expand_and_adds_no_reconfig_method():
    # Native load_pdi: drop --expand-load-pdis, no --reconfig-method.
    assert _full_elf_reconfig_options("loadpdi") == ["--get-scratchpad-parameters"]


def test_loadpdi_union_maps_to_reconfig_method_loadpdi():
    # Union fold of native load_pdi: --reconfig-method=loadpdi (NOT =loadpdi_union),
    # so the loadpdi delivery gets the union/runlist packaging.
    assert _full_elf_reconfig_options("loadpdi_union") == [
        "--reconfig-method=loadpdi",
        "--get-scratchpad-parameters",
    ]


def test_write32_uses_reconfig_method():
    assert _full_elf_reconfig_options("write32") == [
        "--reconfig-method=write32",
        "--get-scratchpad-parameters",
    ]


def test_ctrlpkt_uses_reconfig_method():
    assert _full_elf_reconfig_options("ctrlpkt") == [
        "--reconfig-method=ctrlpkt",
        "--get-scratchpad-parameters",
    ]


def test_ctrlpkt_parallel_adds_parallel_columns():
    # The parallel ctrlpkt delivery: same overlay method, plus the (default-off)
    # per-column shim-DMA collapse that runs columns in parallel.
    assert _full_elf_reconfig_options("ctrlpkt_parallel") == [
        "--reconfig-method=ctrlpkt",
        "--ctrlpkt-parallel-columns",
        "--get-scratchpad-parameters",
    ]


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        _full_elf_reconfig_options("bogus")
