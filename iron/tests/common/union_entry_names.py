#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``_union_entry_names`` must reproduce the union ELF's entrypoints exactly.

``SequenceUnionRunlistCallable`` opens one ``pyxrt.ext.kernel(ctx, name)`` per
entry name this function returns; a name that does not match the ELF fails the
dispatch outright, so the reconstruction has to mirror ``fuse_mlir``'s
configure-block coalescing (consecutive same-design steps share one entry) and
the split's ``op<design_idx>_<Class>_<block_idx>`` naming. These are device-free
unit tests of that pure function -- no ``pyxrt``/device construction below.
"""

import pytest

from iron.common.sequence import _union_entry_names


def _op(class_name):
    """An object whose ``__class__.__name__`` is ``class_name`` (all the
    function reads besides identity)."""
    return type(class_name, (), {})()


def _designs_from(ops):
    """Build the (designs, design_of) pair ``unique_designs`` would return for a
    list of distinct design-representative ops (in first-appearance order)."""
    design_of = {id(op): i for i, op in enumerate(ops)}
    return list(ops), design_of


def test_coalesces_consecutive_same_design():
    rms, gemv, rope, sc = _op("RMSNorm"), _op("GEMV"), _op("RoPE"), _op("StridedCopy")
    designs, design_of = _designs_from([rms, gemv, rope, sc])
    # gemv runs twice back-to-back -> ONE configure block; 4 blocks total (even).
    runlist = [(rms, "x"), (gemv, "a"), (gemv, "b"), (rope, "c"), (sc, "d")]

    names = _union_entry_names(runlist, designs, design_of, False, "main")

    assert names == [
        "main:op0_RMSNorm_0",
        "main:op1_GEMV_1",
        "main:op2_RoPE_2",
        "main:op3_StridedCopy_3",
    ]


def test_ctrlpkt_prepends_init():
    a, b = _op("RMSNorm"), _op("GEMV")
    designs, design_of = _designs_from([a, b])
    runlist = [(a, "x"), (b, "y")]

    names = _union_entry_names(runlist, designs, design_of, True, "main")

    assert names[0] == "main:init"
    assert names[1:] == ["main:op0_RMSNorm_0", "main:op1_GEMV_1"]


def test_name_uses_design_class_not_step_class():
    # A step op whose own class differs from its design's representative class
    # (what share_designs can produce): the entry name must use the design's
    # class, matching how build_fused_mlir names the configure -- else the name
    # diverges from the ELF and dispatch breaks.
    rms, gemv = _op("RMSNorm"), _op("GEMV")
    alias = _op("GemvAlias")  # a different class collapsed onto the GEMV design
    designs, design_of = _designs_from([rms, gemv])
    design_of[id(alias)] = 1  # alias maps to design 1 (GEMV)
    runlist = [(rms, "x"), (alias, "y")]  # 2 configure blocks (even)

    names = _union_entry_names(runlist, designs, design_of, False, "main")

    # The alias step carries the DESIGN's class (GEMV), never its own class.
    assert names == ["main:op0_RMSNorm_0", "main:op1_GEMV_1"]
    assert not any("GemvAlias" in n for n in names)


def test_odd_configure_count_raises():
    a, b, c = _op("RMSNorm"), _op("GEMV"), _op("RoPE")
    designs, design_of = _designs_from([a, b, c])
    runlist = [(a, "x"), (b, "y"), (c, "z")]  # 3 configure blocks (odd)

    with pytest.raises(NotImplementedError):
        _union_entry_names(runlist, designs, design_of, False, "main")
