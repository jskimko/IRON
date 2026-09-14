# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import time
from pathlib import Path
import numpy as np
import ml_dtypes
from . import compilation as comp
from .base import AIEOperatorBase, MLIROperator
import aie.utils as aie_utils
from aie.iron.device import NPU2
from aie.utils.hostruntime.tensor_class import CPUOnlyTensor
from aie.utils.npukernel import NPUKernel

try:
    import pyxrt
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
except ImportError:
    # Host stacks without XRT (e.g. the HRX/amdxdna runtime) have no pyxrt. The two
    # on-device dispatch policies below are XRT-native (pyxrt.elf / hw_context / run,
    # plus XRTTensor views), so they cannot run there; _require_xrt() makes that
    # explicit at construction. The CPU policy and the whole compile path do not care,
    # and must keep importing.
    pyxrt = None
    XRTTensor = None

logger = logging.getLogger(__name__)


def _torch():
    """Import torch for CPU reference/compare paths. Compile and NPU dispatch do not."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "OperatorSequence CPU reference/compare modes need torch. "
            "Compile and NPU dispatch do not."
        ) from exc
    return torch


def _require_xrt() -> None:
    """Fail with the reason, rather than an AttributeError on ``None.elf``."""
    if pyxrt is None:
        raise RuntimeError(
            "this OperatorSequence dispatch policy needs the XRT host runtime (pyxrt), "
            "which is not installed. Use SequenceCPUCallable, or run a single operator "
            "(AIEOperatorBase), which dispatches through aie.utils.DefaultNPURuntime and "
            "works on any backend."
        )


# ##########################################################################
# Dispatch policies
# ##########################################################################


class SequenceDispatch:
    """Policy object that decides how an :class:`OperatorSequence` is compiled
    and how its runtime callable is built.

    One concrete policy corresponds to one dispatch mode. The three hooks are:

    * ``resolve(device)`` -- the only device-aware step; expands ``"auto"`` to
      a concrete policy and validates device requirements. Called once, at
      ``set_up_artifacts()`` time (the device is not known at construction).
    * ``set_up_artifacts(seq)`` -- registers the compile artifacts for this
      mode on the owning sequence.
    * ``make_callable(seq)`` -- returns the runtime callable for this mode.
    """

    name = None

    def resolve(self, device):
        """Return the concrete policy for ``device`` (default: unchanged)."""
        return self

    def set_up_artifacts(self, seq):
        """Register the compile artifacts needed by this mode on ``seq``."""
        raise NotImplementedError

    def make_callable(self, seq):
        """Return the runtime callable for this mode."""
        raise NotImplementedError


class AutoDispatch(SequenceDispatch):
    """Selects the platform default: full-ELF on NPU2, chained-xclbin elsewhere."""

    name = "auto"

    def resolve(self, device):
        if isinstance(device, NPU2):
            return FusedDispatch()
        return SeparateDispatch()


def _trace_tag(seq):
    """Tracing adds a runtime-sequence argument, so a traced build cannot reuse an
    untraced one's ELF. Empty when untraced."""
    return f"_traced{seq.trace_size}" if seq.trace_size else ""


class FusedDispatch(SequenceDispatch):
    """Single-ELF dispatch (NPU2 only): all operators fused into one ELF."""

    name = "fused"

    def resolve(self, device):
        if not isinstance(device, NPU2):
            raise RuntimeError(
                "dispatch='fused' requires NPU2; NPU1 has no full-ELF dispatch"
            )
        return self

    def set_up_artifacts(self, seq):
        mlir_artifact = self.build_fused_mlir(seq)
        kernel_objects = self._collect_kernel_artifacts(seq)
        full_elf_artifact = comp.FullElfArtifact(
            f"{seq.name}{_trace_tag(seq)}.elf",
            mlir_input=mlir_artifact,
            dependencies=[mlir_artifact] + kernel_objects,
            extra_flags=seq.extra_flags,
            trace_size=seq.trace_size,
            reconfig_method=seq.reconfig_method,
        )
        seq.add_artifacts([full_elf_artifact])

    def build_fused_mlir(self, seq):
        """Build the fused MLIR source that inlines every operator into a single
        module.

        ``seq``'s buffer-layout attributes (``subbuffer_layout``,
        ``buffer_sizes``, ``slice_info``) must already be set.
        """
        operator_mlir_map = {}
        comp_runlist = []
        designs, design_of = seq.unique_designs()
        design_names = []

        for idx, op in enumerate(designs):
            mlir_artifact = op.get_mlir_artifact()
            if len(op.get_kernel_artifacts()) > 0:
                mlir_artifact.generator.kwargs["func_prefix"] = f"op{idx}_"
            op_name = f"op{idx}_{op.__class__.__name__}"
            design_names.append(op_name)
            operator_mlir_map[op_name] = mlir_artifact

        for op, *bufs in seq.runlist:
            comp_runlist.append((design_names[design_of[id(op)]], *bufs))

        return comp.SequenceMLIRArtifact(
            f"{seq.name}{_trace_tag(seq)}_fused.mlir",
            operator_mlir_map=operator_mlir_map,
            runlist=comp_runlist,
            subbuffer_layout=seq.subbuffer_layout,
            buffer_sizes=seq.buffer_sizes,
            slice_info=seq.slice_info,
            trace_size=seq.trace_size,
        )

    def _collect_kernel_artifacts(self, seq):
        """Kernel artifacts from all child operators, prefixed per operator index."""
        kernel_artifacts = []
        for idx, op in enumerate(seq.unique_designs()[0]):
            objs = op.get_kernel_artifacts()
            for obj in objs:
                obj.filename = f"op{idx}_{obj.filename}"
                obj.prefix_symbols = f"op{idx}_"
            kernel_artifacts.extend(objs)
        return kernel_artifacts

    def make_callable(self, seq):
        # loadpdi_union/write32/ctrlpkt build a union ELF (--reconfig-method) whose
        # schedule is per-config entrypoints dispatched via pyxrt.runlist. None
        # (expand) and "loadpdi" keep the single baked main:sequence entry.
        if seq.reconfig_method in ("loadpdi_union", "write32", "ctrlpkt"):
            return SequenceUnionRunlistCallable(seq)
        return SequenceFullELFCallable(seq)


class SeparateDispatch(SequenceDispatch):
    """Chained-xclbin dispatch: one xclbin+insts per unique operator, linked
    via ``--xclbin-input`` and invoked sequentially. Owns the compiled
    per-operator xclbin/insts maps consumed by the runtime callable.
    """

    name = "separate"

    def __init__(self):
        self.combined_xclbin = None
        self.op_xclbin_map = {}  # id(op) -> xclbin artifact
        self.op_insts_map = {}  # id(op) -> insts artifact
        self.op_kernel_name_map = {}  # id(op) -> kernel_name

    def set_up_artifacts(self, seq):
        # Short hash keeps kernel names under xclbinutil's 64-char "name:name" limit.
        name_hash = hashlib.sha1(seq.name.encode()).hexdigest()[:6]

        artifacts = []
        prev_xclbin = None
        for idx, op in enumerate(seq.unique_operators()):
            op_label = f"f{name_hash}_op{idx}"
            kernel_id = f"0x{0x901 + idx:x}"

            xclbin, insts = op.get_artifacts(prefix=f"{op_label}_")
            # Copy so we don't mutate the (possibly aliased) shared flags list.
            xclbin.extra_flags = list(xclbin.extra_flags) + [
                f"--xclbin-instance-name={op_label}",
                f"--xclbin-kernel-id={kernel_id}",
            ]
            xclbin.kernel_name = op_label

            if prev_xclbin is not None:
                xclbin.xclbin_input = prev_xclbin
                xclbin.dependencies.add(prev_xclbin)

            artifacts.append(insts)
            self.op_xclbin_map[id(op)] = xclbin
            self.op_insts_map[id(op)] = insts
            self.op_kernel_name_map[id(op)] = op_label
            prev_xclbin = xclbin

        # The last xclbin in the chain carries all the linked instances.
        artifacts.append(prev_xclbin)
        self.combined_xclbin = prev_xclbin
        seq.add_artifacts(artifacts)

    def make_callable(self, seq):
        return SequenceXclbinCallable(seq, self)


class CompareDispatch(SeparateDispatch):
    """Same compile path as ``separate``, but the callable additionally re-runs
    each operator's CPU ``reference()`` on the NPU-produced inputs and flags
    per-step deviation.

    Args:
        rel_tol / abs_tol: Per-step tolerances; a step counts as a mismatch
            only when it exceeds both.
        raise_on_mismatch: When True (default), raise ``RuntimeError`` on the
            first mismatching step instead of only logging it.
    """

    name = "compare"

    def __init__(self, rel_tol=0.05, abs_tol=1e-2, raise_on_mismatch=True):
        super().__init__()
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.raise_on_mismatch = raise_on_mismatch

    def make_callable(self, seq):
        return SequenceCompareCallable(seq, self)


class ReferenceDispatch(SequenceDispatch):
    """Pure-CPU evaluation via each operator's ``reference()``; compiles nothing."""

    name = "reference"

    def set_up_artifacts(self, seq):
        pass

    def make_callable(self, seq):
        return SequenceReferenceCallable(seq)


_DISPATCH_ALIASES = {
    "auto": AutoDispatch,
    "fused": FusedDispatch,
    "separate": SeparateDispatch,
    "compare": CompareDispatch,
    "reference": ReferenceDispatch,
}


# ##########################################################################
# Compileable: operator sequence
# ##########################################################################


class OperatorSequence(AIEOperatorBase):
    """Operator that concatenates a runlist of operators into a
    single dispatch.

    Args:
        dispatch: Dispatch strategy, given either as a mode name or as a
            :class:`SequenceDispatch` instance. Recognised names:
            ``"auto"`` (default) selects ``"fused"`` on NPU2 and
            ``"separate"`` on NPU1.  ``"fused"`` uses a single-ELF
            dispatch (requires NPU2).  ``"separate"`` compiles each
            sub-operator to its own xclbin and invokes them sequentially.
            ``"reference"`` runs only the per-operator CPU reference
            implementations (no NPU compilation/dispatch).  ``"compare"``
            runs the ``"separate"`` xclbin path and, after each NPU step,
            also runs the operator's CPU reference on the NPU-produced
            inputs and logs the deviation for testing/debugging.  Pass a
            :class:`CompareDispatch` instance to tune the compare tolerances.
        reconfig_method: Selects the aiecc reconfiguration flags used for the
            full-ELF build (``dispatch="fused"``/``"auto"`` on NPU2 only; ignored
            by other dispatch modes). ``None`` (default) keeps today's
            ``--expand-load-pdis`` behaviour (static single dispatch). ``"loadpdi"``
            uses native load_pdi (drops ``--expand-load-pdis``, no
            ``--reconfig-method``; static single dispatch). ``"loadpdi_union"``,
            ``"write32"`` and ``"ctrlpkt"`` build the ``--reconfig-method`` union
            ELF, dispatched entry-by-entry via ``SequenceUnionRunlistCallable``
            (``"loadpdi_union"`` folds native load_pdi into the union to isolate
            runlist overhead against the static ``"loadpdi"``). See
            ``iron/common/compilation/base.py``'s
            ``_full_elf_reconfig_options`` for the exact option lists.
    """

    def __init__(
        self,
        name,
        runlist,
        input_args,
        output_args,
        buffer_sizes=None,
        dispatch="auto",
        extra_flags=None,
        trace_size=0,
        share_designs=False,
        reconfig_method=None,
        *args,
        **kwargs,
    ):
        dispatch = self._coerce_dispatch(dispatch)
        if reconfig_method not in comp._RECONFIG_METHODS:
            raise ValueError(
                f"reconfig_method={reconfig_method!r} is not supported; expected "
                f"one of {comp._RECONFIG_METHODS}"
            )
        if not all(
            isinstance(op, MLIROperator) and all(isinstance(buf, str) for buf in bufs)
            for op, *bufs in runlist
        ):
            raise TypeError(
                "runlist entries must be (MLIROperator, *str) tuples; "
                "each operator must be an MLIROperator and each buffer name must be a str"
            )
        super().__init__(*args, **kwargs)
        self.runlist = runlist
        # Sharing changes which designs are built, so it belongs in the name that
        # keys the build artifacts.
        self.name = name + "_shared" if share_designs else name
        self.input_args = input_args
        self.output_args = output_args
        self.explicit_buffer_sizes = (
            buffer_sizes or {}
        )  # Optional dict: buffer_name -> size_in_bytes
        # Extra aiecc flags forwarded to the full-ELF build.
        self.extra_flags = extra_flags or []
        # Bytes of hardware trace buffer per runlist step; 0 leaves the design untraced.
        self.trace_size = trace_size
        self.share_designs = share_designs
        # Selects the full-ELF build's aiecc reconfiguration flags; see
        # FusedDispatch.set_up_artifacts / AieccFullElfCompilationRule.
        self.reconfig_method = reconfig_method
        self._dispatch = dispatch

    @staticmethod
    def _coerce_dispatch(dispatch):
        """Normalise the ``dispatch`` argument to a :class:`SequenceDispatch`."""
        if isinstance(dispatch, SequenceDispatch):
            return dispatch
        elif isinstance(dispatch, str) and dispatch in _DISPATCH_ALIASES:
            return _DISPATCH_ALIASES[dispatch]()
        raise TypeError("selected dispatch mode not supported")

    def unique_operators(self):
        """Operators in runlist order, de-duplicated by identity."""
        seen = {}
        for op, *_ in self.runlist:
            seen.setdefault(id(op), op)
        return list(seen.values())

    def unique_designs(self):
        """The designs to build, and which design each operator uses.

        With ``share_designs`` set, operators reporting the same ``design_key``
        collapse onto one design, so it is built, prefixed and configured once.
        """
        designs = []
        design_of = {}
        first_with_key = {}
        for op in self.unique_operators():
            key = op.design_key() if self.share_designs else None
            if key is not None and key in first_with_key:
                shared = designs[first_with_key[key]]
                if op.get_arg_spec() != shared.get_arg_spec():
                    raise ValueError(
                        f"{op.name} and {shared.name} report the same design_key but "
                        "different runtime arguments, so the design cannot be shared"
                    )
                design_of[id(op)] = first_with_key[key]
                continue
            if key is not None:
                first_with_key[key] = len(designs)
            design_of[id(op)] = len(designs)
            designs.append(op)
        return designs, design_of

    def calculate_buffer_layout(self):
        args = {}  # base_buffer_name -> args_spec
        sliced_buffers = (
            {}
        )  # full_buffer_name (with slice) -> (base_name, start, end, args_spec)

        for op, *bufs in self.runlist:
            args_specs = op.get_arg_spec()
            if len(args_specs) != len(bufs):
                raise ValueError(
                    f"Number of buffers ({len(bufs)}) must match operator argument "
                    f"specification ({len(args_specs)}) for operator {op!r}"
                )
            for i, buf_name in enumerate(bufs):
                args_spec = args_specs[i]

                # Parse slice notation: "buffer_name[start:end]"
                if "[" in buf_name and buf_name.endswith("]"):
                    base_name = buf_name[: buf_name.index("[")]
                    slice_part = buf_name[buf_name.index("[") + 1 : -1]
                    start, end = map(int, slice_part.split(":"))
                    sliced_buffers[buf_name] = (base_name, start, end, args_spec)
                    # Track that base buffer exists (size will be set later)
                    if (
                        base_name not in args
                        and base_name not in self.explicit_buffer_sizes
                    ):
                        raise ValueError(
                            f"Sliced buffer '{buf_name}' requires explicit size for base buffer '{base_name}' in buffer_sizes parameter"
                        )
                else:
                    if buf_name not in args:
                        args[buf_name] = args_spec
                    else:
                        if np.prod(args[buf_name].shape) != np.prod(args_spec.shape):
                            raise ValueError(
                                f"Buffer '{buf_name}' has conflicting sizes between operators: "
                                f"{args[buf_name].shape} vs {args_spec.shape}"
                            )

        # Verify all input/output args are present (either as regular or sliced buffers)
        all_buffer_names = set(args.keys()) | set(sliced_buffers.keys())
        for arg in self.input_args:
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Input argument {arg} not found in runlist buffers")
        for arg in self.output_args:
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Output argument {arg} not found in runlist buffers")

        subbuffer_layout = {}
        slice_info = {}  # full_buffer_name -> (base_name, start, end)

        def add_buffers(buffer_type, args_list):
            offset = 0
            for arg in args_list:
                if arg in self.explicit_buffer_sizes:
                    # Explicit size specified - this is a parent buffer for slices
                    length = self.explicit_buffer_sizes[arg]
                    subbuffer_layout[arg] = (buffer_type, offset, length)
                    offset += length
                elif arg in args:
                    arg_spec = args[arg]
                    length = int(
                        np.prod(arg_spec.shape) * np.dtype(arg_spec.dtype).itemsize
                    )
                    subbuffer_layout[arg] = (buffer_type, offset, length)
                    offset += length
                # Note: sliced buffers are handled separately, not in args_list
            return offset  # == total length

        # Add sliced buffer entries to layout (they reference parent buffers)
        for buf_name, (base_name, start, end, args_spec) in sliced_buffers.items():
            slice_info[buf_name] = (base_name, start, end)

        input_buffer_size = add_buffers("input", self.input_args)
        output_buffer_size = add_buffers("output", self.output_args)
        scratch_args = [
            arg
            for arg in args
            if arg not in self.input_args and arg not in self.output_args
        ]
        # Also include explicit buffers that are only used for slicing
        for explicit_buf in self.explicit_buffer_sizes:
            if (
                explicit_buf not in self.input_args
                and explicit_buf not in self.output_args
                and explicit_buf not in scratch_args
            ):
                scratch_args.append(explicit_buf)
        scratch_buffer_size = add_buffers("scratch", scratch_args)

        buffer_sizes = (input_buffer_size, output_buffer_size, scratch_buffer_size)
        return subbuffer_layout, buffer_sizes, slice_info

    def set_up_artifacts(self):
        """Resolve the dispatch policy and build its compile artifacts."""
        self.subbuffer_layout, self.buffer_sizes, self.slice_info = (
            self.calculate_buffer_layout()
        )
        self._dispatch = self._dispatch.resolve(aie_utils.get_current_device())
        self._dispatch.set_up_artifacts(self)

    def get_arg_spec(self):
        raise NotImplementedError(
            "OperatorSequence does not expose a unified arg spec; "
            "use get_layout_for_buffer() to inspect individual buffer layouts"
        )

    def get_callable(self):
        """Return the runtime callable for the resolved dispatch policy."""
        return self._dispatch.make_callable(self)

    def get_layout_for_buffer(self, buffer_name):
        """Return the (buffer_type, offset, length) layout for a named buffer.

        Sliced buffers are resolved recursively to their parent's absolute
        offset.

        Args:
            buffer_name: Name of the buffer, optionally with slice notation.

        Returns:
            Tuple of (buf_type, offset_bytes, length_bytes).
        """
        if buffer_name in self.slice_info:
            buf_name, start, end = self.slice_info[buffer_name]
            buf_type, parent_start, parent_end = self.get_layout_for_buffer(buf_name)
            return buf_type, parent_start + start, parent_start + end

        buf_type, offset, length = self.subbuffer_layout[buffer_name]
        return buf_type, offset, length


# ##########################################################################
# Module helpers
# ##########################################################################


BF16 = np.dtype(ml_dtypes.bfloat16)


def _n_elements(nbytes):
    return max(nbytes, BF16.itemsize) // BF16.itemsize


# ##########################################################################
# Runtime callables
# ##########################################################################


class SequenceCallable:
    """Base for the runtime callables of an ``OperatorSequence``.

    Subclasses provide a buffer model (``_allocate_buffers`` / ``get_buffer``)
    and a step-execution primitive (``_run``). Shared here: step/arg zipping,
    input and output syncing, and timing. Calling the object runs the whole
    sequence once.
    """

    def __init__(self, op):
        self.op = op
        self.last_elapsed = 0.0
        self._buffer_cache = {}
        self._allocate_buffers()

    def _allocate_buffers(self):
        raise NotImplementedError

    def get_buffer(self, buffer_name):
        raise NotImplementedError

    def _iter_steps(self):
        """Yield ``(op, in_names, in_specs, out_name, out_spec)`` per runlist step."""
        for step_op, *buf_names in self.op.runlist:
            specs = step_op.get_arg_spec()
            if len(specs) != len(buf_names):
                raise ValueError(
                    f"Operator {step_op!r} arg-spec count {len(specs)} does not "
                    f"match runlist buffer count {len(buf_names)}"
                )
            *in_names, out_name = buf_names
            *in_specs, out_spec = specs
            yield step_op, in_names, in_specs, out_name, out_spec

    def _sync_inputs(self):
        pass

    def _sync_outputs(self):
        pass

    def _run(self):
        raise NotImplementedError

    def __call__(self):
        self._sync_inputs()
        t0 = time.perf_counter()
        self._run()
        self.last_elapsed = time.perf_counter() - t0
        self._sync_outputs()


class _ConsolidatedBufferCallable(SequenceCallable):
    """Shared buffer model for the NPU dispatchers that consume
    ``OperatorSequence``'s three consolidated input/output/scratch arenas
    addressed by baked offset.

    Both the single-ELF (``SequenceFullELFCallable``) and union-runlist
    (``SequenceUnionRunlistCallable``) dispatchers pack every operator argument
    into the same three ``XRTTensor`` arenas, so buffer allocation,
    ``get_buffer`` sub-viewing, host<->device syncing and the parameter
    scratchpad are identical. Subclasses supply only the kernel/dispatch setup
    (``__init__``) and ``_run``; they must set ``self._params = None`` and
    implement ``_make_scratchpad`` (the writer for this dispatcher's runtime
    parameter scratchpad -- one run for the full-ELF, all entry runs for the
    union).
    """

    def _make_scratchpad(self, params_path):
        """Build the runtime-parameter writer bound to this dispatcher's run(s)."""
        raise NotImplementedError

    @property
    def params(self):
        """Lazy runtime-parameter writer bound to this ELF's ctrl scratchpad.

        The ``params.txt`` describing the runtime parameters is requested from
        aiecc via ``--get-scratchpad-parameters``; it is a graph output, so it
        lands in aiecc's ``--output-dir``, which compile_mlir_module() points at
        the work dir (see ``_aiecc_work_dir``) for the fused MLIR source.
        Returns ``None`` if the sequence declared no runtime parameters: the
        file is still written, but holds a count of zero and there is no ctrl
        scratchpad buffer object to bind to.
        """
        if self._params is not None:
            return self._params
        mlir_filename = self.op.artifacts[0].mlir_input.filename
        params_path = comp._aiecc_work_dir(mlir_filename) / "params.txt"
        if not params_path.exists():
            return None
        if params_path.read_text().split("\n", 1)[0].strip() == "0":
            return None
        self._params = self._make_scratchpad(str(params_path))
        return self._params

    def _allocate_buffers(self):
        in_sz, out_sz, scratch_sz = self.op.buffer_sizes
        self.input_buffer = XRTTensor((_n_elements(in_sz),), dtype=ml_dtypes.bfloat16)
        self.output_buffer = XRTTensor((_n_elements(out_sz),), dtype=ml_dtypes.bfloat16)
        self.scratch_buffer = XRTTensor(
            (_n_elements(scratch_sz),), dtype=ml_dtypes.bfloat16
        )
        # Trace lowering appends one buffer covering every configured design, after
        # the consolidated three. Its size depends on how many channels and
        # sub-designs claim a share, so read it from the lowered module.
        self.trace_buffer = None
        if self.op.trace_size:
            total = comp.trace_buffer_size(self.lowered_mlir_text())
            if total:
                self.trace_buffer = XRTTensor((total,), dtype=np.int8)

    def lowered_mlir_text(self) -> str:
        """aiecc's post-lowering module, which carries the trace buffer layout."""
        mlir_filename = self.op.artifacts[0].mlir_input.filename
        path = comp._aiecc_work_dir(mlir_filename) / "input_with_addresses.mlir"
        return path.read_text()

    def get_buffer(self, buffer_name):
        if buffer_name in self._buffer_cache:
            return self._buffer_cache[buffer_name]
        buf_type, offset, length = self.op.get_layout_for_buffer(buffer_name)
        parent = {
            "input": self.input_buffer,
            "output": self.output_buffer,
            "scratch": self.scratch_buffer,
        }[buf_type]
        sub = parent.subview(offset, (length // BF16.itemsize,), ml_dtypes.bfloat16)
        self._buffer_cache[buffer_name] = sub
        return sub

    def _sync_inputs(self):
        # Sub-views handed out by get_buffer() share the parent's coherence map, so
        # a write through one (e.g. torch_view()) marks its byte range host-dirty
        # there too, and `to("npu")` here syncs every dirty range in one pass.
        self.input_buffer.to("npu")

    def _sync_outputs(self):
        # _run just rewrote the output arena on the device, so the device holds the
        # authoritative copy. Force the device->host sync: assert device residency first
        # so `to("cpu")` fires even if a prior read of get_buffer(...) marked some
        # range "cpu" (otherwise a looped dispatch would read stale output).
        self.output_buffer.device = "npu"
        self.output_buffer.to("cpu")
        if self.trace_buffer is not None:
            self.trace_buffer.device = "npu"
            self.trace_buffer.to("cpu")


class SequenceFullELFCallable(_ConsolidatedBufferCallable):
    """Single-ELF dispatch (NPU2): every operator shares three consolidated
    input/output/scratch buffers addressed by offset. ``get_buffer`` returns a
    sub-view into whichever consolidated buffer holds the named argument.

    The whole schedule runs as ONE ``main:sequence`` dispatch (the 322 configs
    are baked inline for ``--expand-load-pdis`` / native ``load_pdi``); the
    union-runlist form is ``SequenceUnionRunlistCallable``.
    """

    def __init__(self, op, device_name="main", sequence_name="sequence"):
        _require_xrt()
        self.device_name = device_name
        self.sequence_name = sequence_name

        assert isinstance(op.artifacts[0], comp.FullElfArtifact)
        xrt_elf = pyxrt.elf(str(op.artifacts[0].filename))
        xrt_context = pyxrt.hw_context(aie_utils.DefaultNPURuntime._device, xrt_elf)
        self.xrt_kernel = pyxrt.ext.kernel(
            xrt_context, f"{self.device_name}:{self.sequence_name}"
        )

        super().__init__(op)

        # Persistent run handle: reused across dispatches so that the
        # ctrl-scratchpad backing buffer (and any ParameterScratchpad state
        # built on top of it) stays valid across calls.
        self.run_handle = pyxrt.run(self.xrt_kernel)
        self.run_handle.set_arg(0, self.input_buffer.buffer_object())
        self.run_handle.set_arg(1, self.output_buffer.buffer_object())
        self.run_handle.set_arg(2, self.scratch_buffer.buffer_object())
        if self.trace_buffer is not None:
            self.run_handle.set_arg(3, self.trace_buffer.buffer_object())

        self._params = None

    def _make_scratchpad(self, params_path):
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        return ParameterScratchpad(self.run_handle, params_path)

    def _run(self):
        self.run_handle.start()
        ret_code = self.run_handle.wait()
        if ret_code != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"Kernel execution failed with return code {ret_code}")


# Bytes of the inert control-packet BO handed to every ctrlpkt entry (and to
# main:init). The overlay bakes the reconfiguration, so the slot is never read;
# it only has to exist and be large enough. Mirrors the reference harness's
# dummy ctrl BO (test/python/npu-xrt/test_reconfig_runlist.py).
_CTRL_BO_BYTES = 4096


def _union_entry_names(runlist, designs, design_of, needs_ctrl_bo, device_name):
    """Union ELF entrypoint names in schedule (configure) order.

    Pure function (no device state) so it can be unit-tested off-device.

    The fused-MLIR build (``fuse_mlir`` in ``compilation/sequence.py``) coalesces
    *consecutive* runlist steps that use the same design into ONE
    ``aiex.configure`` block -- it reconfigures only when the design changes. The
    union split then makes one ELF entrypoint per configure block, named
    ``<device>:op<design_idx>_<Class>_<block_idx>`` (design name exactly as in
    ``FusedDispatch.build_fused_mlir`` -- keyed on the *design's* representative
    class, not the step op's, so it stays correct if ``share_designs`` ever
    collapses two classes onto one design; block index counts configure blocks
    from 0). This mirrors the coalescing exactly, so the names match the ELF's
    entrypoints one-for-one. ``needs_ctrl_bo`` (ctrlpkt) prepends ``<device>:init``.

    Raises ``NotImplementedError`` for an odd configure-block count, which
    ``fuse_mlir`` pads with an extra reset configure
    (``needs_additional_reset``) that this dispatcher does not wire (llama's
    schedule is even).
    """
    names = []
    if needs_ctrl_bo:
        names.append(f"{device_name}:init")
    last_design_name = None
    block_idx = 0
    for step_op, *_bufs in runlist:
        d = design_of[id(step_op)]
        design_name = f"op{d}_{designs[d].__class__.__name__}"
        if design_name == last_design_name:
            # Consecutive same-design steps share one configure entry.
            continue
        last_design_name = design_name
        names.append(f"{device_name}:{design_name}_{block_idx}")
        block_idx += 1
    if block_idx % 2 == 1:
        raise NotImplementedError(
            "union runlist dispatch: workload has an odd configure-point count "
            "and needs an extra reset entry, which is not implemented"
        )
    return names


class _BroadcastParameterScratchpad:
    """Fan a runtime-parameter write/sync out to every union entry's ctrl
    scratchpad.

    The single-ELF dispatch has one ``pyxrt.run`` whose scratchpad BO every
    config reads, so one ``ParameterScratchpad`` suffices. The union dispatches
    each config as its OWN ``pyxrt.run``; a runtime parameter must land in the
    scratchpad of whichever entry actually reads it (e.g. ``softmax_vector_size``
    on the Softmax config), so we build one ``ParameterScratchpad`` per entry run
    and broadcast writes/syncs to all. Writing a scratchpad an entry never reads
    is harmless.

    Cost/measurement caveat: this fans ``write``/``sync`` out to every entry per
    token (a device sync each). It is intrinsic to the *harness*, not the
    dispatch method being measured, so it can add a small per-token overhead to
    the union's reported timings that a consumer-targeted implementation would
    not pay. Every union entry was found to expose a ctrl scratchpad BO
    (322/322 for the llama write32 build), so a per-run construction failure is a
    genuine error and propagates rather than being silently skipped.
    """

    def __init__(self, runs, params_path):
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad,
        )

        self._pads = [ParameterScratchpad(run, params_path) for run in runs]
        if not self._pads:
            raise RuntimeError("union has no entry runs to bind parameters to")

    def write(self, name, value):
        for pad in self._pads:
            pad.write(name, value)

    def sync(self):
        for pad in self._pads:
            pad.sync()

    def read(self, name):
        return self._pads[0].read(name)


class SequenceUnionRunlistCallable(_ConsolidatedBufferCallable):
    """Union-ELF dispatch (NPU2): the ``--reconfig-method`` fold packages the
    schedule as one ELF with one entrypoint per configure block
    (``main:op<design_idx>_<Class>_<block_idx>``); each entry reconfigures the
    array and runs the config's step(s) -- consecutive same-design runlist steps
    coalesce into one block (llama: 882 steps -> 322 entries). A single
    ``pyxrt.runlist`` submits every entry in schedule order, atomically and in
    place; each binds the SAME three consolidated base BOs (each entry bakes its
    own sub-view offsets into those arenas, so there is no per-step BO math).

    ``ctrlpkt`` additionally stands up a resident overlay via a prepended
    ``main:init`` and gives every entry an extra inert control-packet BO; the
    ``write32``/``loadpdi_union`` forms need neither. Correctness across the
    separate per-entry dispatches relies on (a) the shared scratch arena (op K's
    writes persist for op K+1) and (b) the runlist executing in order. Runtime
    parameters are broadcast to every entry's scratchpad (see
    ``_BroadcastParameterScratchpad``), since the scratchpad is per-entry-run.
    """

    def __init__(self, op, device_name="main"):
        _require_xrt()
        self.device_name = device_name

        if op.trace_size:
            # Trace lowering appends a 4th buffer, but the union binds only the 3
            # base BOs per entry (arg slot 3 is the ctrl BO for ctrlpkt), so the
            # trace buffer would never be wired. Fail loud rather than silently.
            raise NotImplementedError(
                "hardware tracing is not supported for union runlist dispatch"
            )

        assert isinstance(op.artifacts[0], comp.FullElfArtifact)
        self._xrt_elf = pyxrt.elf(str(op.artifacts[0].filename))
        self._xrt_context = pyxrt.hw_context(
            aie_utils.DefaultNPURuntime._device, self._xrt_elf
        )
        # ctrlpkt is the only method whose entries carry a control-packet slot
        # and that needs the main:init overlay standup prepended.
        self._needs_ctrl_bo = op.reconfig_method == "ctrlpkt"

        super().__init__(op)

        # One inert control-packet BO, shared by main:init and every entry.
        self._ctrl_buffer = None
        if self._needs_ctrl_bo:
            self._ctrl_buffer = XRTTensor(
                (_n_elements(_CTRL_BO_BYTES),), dtype=ml_dtypes.bfloat16
            )
            self._ctrl_buffer.to("npu")

        base_bos = (
            self.input_buffer.buffer_object(),
            self.output_buffer.buffer_object(),
            self.scratch_buffer.buffer_object(),
        )

        # Build the runlist ONCE and reuse it across decode tokens: rebuilding
        # 322 pyxrt.run + set_arg per token would reintroduce exactly the
        # per-token host overhead this dispatch mode is meant to measure. The
        # kernels and runs are retained (self._keep) for the callable's life so
        # the runlist's references stay valid.
        self._runlist = pyxrt.runlist(self._xrt_context)
        self._keep = []  # kernels + runs kept alive for the runlist's life
        self._entry_runs = []  # per-config entry runs (excludes main:init)
        for name in self._entry_names():
            kernel = pyxrt.ext.kernel(self._xrt_context, name)
            run = pyxrt.run(kernel)
            if name.endswith(":init"):
                # main:init has a fixed overlay-defined signature: only the
                # (inert) control-packet BO, no data arenas.
                run.set_arg(0, self._ctrl_buffer.buffer_object())
            else:
                for i, bo in enumerate(base_bos):
                    run.set_arg(i, bo)
                if self._needs_ctrl_bo:
                    run.set_arg(len(base_bos), self._ctrl_buffer.buffer_object())
                self._entry_runs.append(run)
            self._runlist.add(run)
            self._keep.append((kernel, run))

        self._params = None

    def _entry_names(self):
        """This ELF's entrypoint names in schedule (configure) order.

        Delegates to :func:`_union_entry_names`; ``ctrlpkt`` prepends
        ``main:init``.
        """
        designs, design_of = self.op.unique_designs()
        return _union_entry_names(
            self.op.runlist,
            designs,
            design_of,
            self._needs_ctrl_bo,
            self.device_name,
        )

    def _make_scratchpad(self, params_path):
        # The union has one ctrl scratchpad per entry run, so a parameter must
        # reach whichever entry reads it; broadcast to all entry runs.
        return _BroadcastParameterScratchpad(self._entry_runs, params_path)

    def _run(self):
        self._runlist.execute()
        self._runlist.wait()


class _PerBufferCallable(SequenceCallable):
    """Callable whose buffers are allocated one per name, with slice views into
    their parent. Inputs sync to the device before the run, all non-input
    buffers back to the host afterwards.
    """

    def _make_buffer(self, n_elements):
        raise NotImplementedError

    def _make_subbuffer(self, parent, offset_bytes, size_bytes):
        raise NotImplementedError

    def _allocate_buffers(self):
        self._buffers = {}
        for name, (_, _, length) in self.op.subbuffer_layout.items():
            self._buffers[name] = self._make_buffer(_n_elements(length))

    def _resolve_buffer(self, buf_name):
        if buf_name in self._buffers:
            return self._buffers[buf_name]
        if buf_name in self.op.slice_info:
            base_name, start_bytes, end_bytes = self.op.slice_info[buf_name]
            sub = self._make_subbuffer(
                self._buffers[base_name], start_bytes, end_bytes - start_bytes
            )
            self._buffers[buf_name] = sub
            return sub
        raise ValueError(f"Unknown buffer '{buf_name}' in fused runlist")

    def get_buffer(self, buffer_name):
        if buffer_name not in self._buffer_cache:
            self._buffer_cache[buffer_name] = self._resolve_buffer(buffer_name)
        return self._buffer_cache[buffer_name]

    def _sync_inputs(self):
        for name in self.op.input_args:
            self._buffers[name].to("npu")

    def _sync_outputs(self):
        for name in self.op.subbuffer_layout:
            if name not in self.op.input_args:
                self._buffers[name].to("cpu")


class SequenceXclbinCallable(_PerBufferCallable):
    """Executes each runlist step as its own xclbin dispatch. Buffers shared by
    name give zero-copy handoff between consecutive operators.

    The compiled per-operator xclbin/insts maps live on the ``SeparateDispatch``
    policy passed in as ``dispatch``.
    """

    def __init__(self, op, dispatch):
        _require_xrt()
        self._dispatch = dispatch
        super().__init__(op)

    def _make_buffer(self, n_elements):
        return XRTTensor((n_elements,), dtype=ml_dtypes.bfloat16)

    def _make_subbuffer(self, parent, offset_bytes, size_bytes):
        return parent.subview(
            offset_bytes, (size_bytes // BF16.itemsize,), ml_dtypes.bfloat16
        )

    def _allocate_buffers(self):
        super()._allocate_buffers()
        dispatch = self._dispatch
        combined_xclbin_path = dispatch.combined_xclbin.filename
        self._op_callable_map = {}  # id(op) -> NPUKernel
        for op_id, xclbin in dispatch.op_xclbin_map.items():
            self._op_callable_map[op_id] = NPUKernel(
                xclbin_path=combined_xclbin_path,
                kernel_name=dispatch.op_kernel_name_map[op_id],
                insts_path=dispatch.op_insts_map[op_id].filename,
            )
        self._execution_plan = [
            (
                self._op_callable_map[id(step_op)],
                [self._resolve_buffer(name) for name in buf_names],
            )
            for step_op, *buf_names in self.op.runlist
        ]

    def _run(self):
        # Walk the execution plan alongside the resolved runlist steps; the
        # per-step behaviour is delegated to _run_step so that compare mode can
        # reuse this loop verbatim.
        for step_idx, ((kernel, args), step) in enumerate(
            zip(self._execution_plan, self._iter_steps())
        ):
            self._run_step(step_idx, kernel, args, step)

    def _run_step(self, step_idx, kernel, args, step):
        kernel(*args)


def _reshape_for_spec(flat_tensor, spec):
    """Slice a flat host buffer to ``spec``'s element count and reshape (a view)."""
    n = int(np.prod(spec.shape)) if spec.shape else 1
    return flat_tensor[:n].reshape(spec.shape)


class SequenceReferenceCallable(_PerBufferCallable):
    """Pure-CPU evaluation via each operator's ``reference()``; no NPU dispatch.
    Device syncs are no-ops on the CPU buffers.
    """

    def _make_buffer(self, n_elements):
        return CPUOnlyTensor((n_elements,), dtype=BF16)

    def _make_subbuffer(self, parent, offset_bytes, size_bytes):
        start = offset_bytes // BF16.itemsize
        end = (offset_bytes + size_bytes) // BF16.itemsize
        # Alias the parent's memory (numpy slice is zero-copy) so a write to
        # this slice is visible when a later step reads the parent by name.
        view = CPUOnlyTensor((end - start,), dtype=BF16)
        view._data = parent.data[start:end]
        view._shape = view._data.shape
        return view

    def _run(self):
        torch = _torch()
        for step_op, in_names, in_specs, out_name, out_spec in self._iter_steps():
            inputs = [
                _reshape_for_spec(self._resolve_buffer(n).torch_view(), s).clone()
                for n, s in zip(in_names, in_specs)
            ]
            out = step_op.reference(*inputs)
            out_flat = self._resolve_buffer(out_name).torch_view()
            n_out = int(np.prod(out_spec.shape)) if out_spec.shape else 1
            out_flat[:n_out].copy_(out.reshape(-1).to(torch.bfloat16))


class SequenceCompareCallable(SequenceXclbinCallable):
    """Runs the xclbin pipeline and, after each step, re-runs the operator's
    reference on the same NPU-produced inputs, logging per-step deviation. The
    NPU output propagates on both sides, so each comparison isolates a single
    operator (no error accumulation).
    """

    def __init__(self, op, dispatch):
        super().__init__(op, dispatch)
        self.rel_tol = dispatch.rel_tol
        self.abs_tol = dispatch.abs_tol
        self.raise_on_mismatch = dispatch.raise_on_mismatch
        self.last_step_stats = []

    def _read_to_cpu(self, name, spec):
        buf = self._resolve_buffer(name)
        buf.to("cpu")
        n = int(np.prod(spec.shape)) if spec.shape else 1
        return buf.torch_view()[:n].clone().reshape(spec.shape)

    def _run(self):
        # Reset per-invocation stats, then reuse SequenceXclbinCallable._run's
        # execution-plan loop; only the per-step behaviour (_run_step) differs.
        self.last_step_stats = []
        super()._run()

    def _run_step(self, step_idx, kernel, args, step):
        step_op, in_names, in_specs, out_name, out_spec = step

        cpu_inputs = [
            self._read_to_cpu(name, spec) for name, spec in zip(in_names, in_specs)
        ]

        kernel(*args)

        torch = _torch()
        npu_out = self._read_to_cpu(out_name, out_spec).to(torch.float32)
        ref_out = step_op.reference(*cpu_inputs)

        stats = {
            "step": step_idx,
            "op": type(step_op).__name__,
            "op_name": getattr(step_op, "name", type(step_op).__name__),
            "inputs": list(in_names),
            "output": out_name,
        }

        ref_flat = ref_out.reshape(out_spec.shape).to(torch.float32)
        diff = (npu_out - ref_flat).abs()
        ref_mag = ref_flat.abs()
        max_abs = float(diff.max())
        ref_max = float(ref_mag.max())
        rel = float((diff / (ref_mag + 1e-6)).max())
        mean_abs = float(diff.mean())
        stats.update(
            skipped=False,
            max_abs=max_abs,
            mean_abs=mean_abs,
            max_rel=rel,
            ref_max=ref_max,
        )
        fail = (max_abs > self.abs_tol) and (rel > self.rel_tol)
        stats["mismatch"] = fail
        level = logging.ERROR if fail else logging.INFO
        logger.log(
            level,
            "[compare step %d] %s -> %s: max_abs=%.4g mean_abs=%.4g max_rel=%.4g ref_max=%.4g%s",
            step_idx,
            stats["op"],
            out_name,
            max_abs,
            mean_abs,
            rel,
            ref_max,
            "  MISMATCH" if fail else "",
        )
        if fail and self.raise_on_mismatch:
            raise RuntimeError(
                f"[compare step {step_idx}] {stats['op']} (name={stats['op_name']}) "
                f"-> {out_name}: NPU output deviates from reference "
                f"(max_abs={max_abs:.4g}, max_rel={rel:.4g}, "
                f"ref_max={ref_max:.4g}; inputs={list(in_names)}; "
                f"tolerances abs_tol={self.abs_tol}, rel_tol={self.rel_tol})"
            )
        self.last_step_stats.append(stats)
