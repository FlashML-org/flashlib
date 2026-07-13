from __future__ import annotations
_KERNEL_ALIAS_BY_IR_NAME = {'flash_kmeans_assign_lowdim_pack_e50c_v1': 'dispatch_kernel_0000', 'flash_kmeans_assign_lowdim_e50c_v1': 'dispatch_kernel_0001', 'flash_kmeans_assign_cleanroom_tcgen05_v10': 'dispatch_kernel_0002', 'flash_kmeans_assign_d160_pad192_pack_f9b2_v1': 'dispatch_kernel_0003', 'flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1': 'dispatch_kernel_0004', 'flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1': 'dispatch_kernel_0005', 'flash_kmeans_assign_highd_splitd_6fcf_v1': 'dispatch_kernel_0006', 'flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1': 'dispatch_kernel_0007', 'flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1': 'dispatch_kernel_0008', 'flash_kmeans_assign_microdim_pack_6cd2_v1': 'dispatch_kernel_0009', 'flash_kmeans_assign_microdim_6cd2_v1': 'dispatch_kernel_0010', 'flash_kmeans_assign_gap_pad_pack_v1': 'dispatch_kernel_0011', 'flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1': 'dispatch_kernel_0012', 'flash_kmeans_assign_cleanroom_tcgen05_v15': 'dispatch_kernel_0013', 'flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1': 'dispatch_kernel_0014', 'flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1': 'dispatch_kernel_0015', 'flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1': 'dispatch_kernel_0016', 'flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1': 'dispatch_kernel_0017', 'flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1': 'dispatch_kernel_0018', 'flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1': 'dispatch_kernel_0019', 'flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1': 'dispatch_kernel_0020', 'flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1': 'dispatch_kernel_0021', 'flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4': 'dispatch_kernel_0022', 'flash_kmeans_assign_microdim_raw_tma_08f9_v1': 'dispatch_kernel_0023', 'flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1': 'dispatch_kernel_0024', 'flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4': 'dispatch_kernel_0025', 'flash_kmeans_assign_d288_exactd_a532_v1': 'dispatch_kernel_0026', 'flash_kmeans_assign_d288_splitk_cta_0438_v1_partial': 'dispatch_kernel_0027', 'flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce': 'dispatch_kernel_0028', 'flash_kmeans_assign_d480_splitk_partial_d32k256_v1': 'dispatch_kernel_0029', 'flash_kmeans_assign_d480_splitk_reduce_d32k256_v1': 'dispatch_kernel_0030', 'flash_kmeans_assign_highd_splitk_partial_8de8_v1': 'dispatch_kernel_0031', 'flash_kmeans_assign_highd_splitk_reduce_8de8_v1': 'dispatch_kernel_0032', 'flash_kmeans_assign_microdim_direct_9c0d_v1': 'dispatch_kernel_0033', 'flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2': 'dispatch_kernel_0034', 'flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1': 'dispatch_kernel_0035', 'flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1': 'dispatch_kernel_0036', 'flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1': 'dispatch_kernel_0037'}
_KERNEL_ALIAS_BY_REQUEST = {'{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_lowdim_pack_e50c_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0000', '{"computed_smem_bytes":100352,"constants":[],"ir_name":"flash_kmeans_assign_lowdim_e50c_v1","kwargs":{"smem_bytes":100352,"validate":false},"threads":192}': 'dispatch_kernel_0001', '{"computed_smem_bytes":100352,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_v10","kwargs":{"smem_bytes":100352,"validate":false},"threads":192}': 'dispatch_kernel_0002', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_d160_pad192_pack_f9b2_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0003', '{"computed_smem_bytes":149504,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1","kwargs":{"smem_bytes":149504,"validate":false},"threads":192}': 'dispatch_kernel_0004', '{"computed_smem_bytes":198656,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1","kwargs":{"smem_bytes":198656,"validate":false},"threads":192}': 'dispatch_kernel_0005', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitd_6fcf_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0006', '{"computed_smem_bytes":43008,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1","kwargs":{"smem_bytes":43008,"validate":false},"threads":192}': 'dispatch_kernel_0007', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0008', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_microdim_pack_6cd2_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0009', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_microdim_6cd2_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0010', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_gap_pad_pack_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0011', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0012', '{"computed_smem_bytes":133120,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_v15","kwargs":{"smem_bytes":133120,"validate":false},"threads":192}': 'dispatch_kernel_0013', '{"computed_smem_bytes":43008,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1","kwargs":{"smem_bytes":43008,"validate":false},"threads":192}': 'dispatch_kernel_0014', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0015', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0016', '{"computed_smem_bytes":198656,"constants":[],"ir_name":"flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1","kwargs":{"smem_bytes":198656,"validate":false},"threads":192}': 'dispatch_kernel_0017', '{"computed_smem_bytes":75776,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1","kwargs":{"smem_bytes":75776,"validate":false},"threads":192}': 'dispatch_kernel_0018', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":64}': 'dispatch_kernel_0019', '{"computed_smem_bytes":43008,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1","kwargs":{"smem_bytes":43008,"validate":false},"threads":192}': 'dispatch_kernel_0020', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0021', '{"computed_smem_bytes":23552,"constants":[],"ir_name":"flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4","kwargs":{"smem_bytes":23552,"validate":false},"threads":192}': 'dispatch_kernel_0022', '{"computed_smem_bytes":52224,"constants":[],"ir_name":"flash_kmeans_assign_microdim_raw_tma_08f9_v1","kwargs":{"smem_bytes":52224,"validate":false},"threads":192}': 'dispatch_kernel_0023', '{"computed_smem_bytes":46592,"constants":[],"ir_name":"flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1","kwargs":{"smem_bytes":46592,"validate":false},"threads":256}': 'dispatch_kernel_0024', '{"computed_smem_bytes":22528,"constants":[],"ir_name":"flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4","kwargs":{"smem_bytes":22528,"validate":false},"threads":192}': 'dispatch_kernel_0025', '{"computed_smem_bytes":26624,"constants":[],"ir_name":"flash_kmeans_assign_d288_exactd_a532_v1","kwargs":{"smem_bytes":26624,"validate":false},"threads":192}': 'dispatch_kernel_0026', '{"computed_smem_bytes":26624,"constants":[],"ir_name":"flash_kmeans_assign_d288_splitk_cta_0438_v1_partial","kwargs":{"smem_bytes":26624,"validate":false},"threads":192}': 'dispatch_kernel_0027', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0028', '{"computed_smem_bytes":22528,"constants":[],"ir_name":"flash_kmeans_assign_d480_splitk_partial_d32k256_v1","kwargs":{"smem_bytes":22528,"validate":false},"threads":192}': 'dispatch_kernel_0029', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_d480_splitk_reduce_d32k256_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":256}': 'dispatch_kernel_0030', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_partial_8de8_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0031', '{"computed_smem_bytes":0,"constants":[],"ir_name":"flash_kmeans_assign_highd_splitk_reduce_8de8_v1","kwargs":{"smem_bytes":0,"validate":false},"threads":128}': 'dispatch_kernel_0032', '{"computed_smem_bytes":51200,"constants":[],"ir_name":"flash_kmeans_assign_microdim_direct_9c0d_v1","kwargs":{"smem_bytes":51200,"validate":false},"threads":192}': 'dispatch_kernel_0033', '{"computed_smem_bytes":26624,"constants":[],"ir_name":"flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2","kwargs":{"smem_bytes":26624,"validate":false},"threads":192}': 'dispatch_kernel_0034', '{"computed_smem_bytes":29696,"constants":[],"ir_name":"flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1","kwargs":{"smem_bytes":29696,"validate":false},"threads":160}': 'dispatch_kernel_0035', '{"computed_smem_bytes":43008,"constants":[],"ir_name":"flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1","kwargs":{"smem_bytes":43008,"validate":false},"threads":256}': 'dispatch_kernel_0036', '{"computed_smem_bytes":35840,"constants":[],"ir_name":"flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1","kwargs":{"smem_bytes":35840,"validate":false},"threads":416}': 'dispatch_kernel_0037'}

import ctypes
import importlib
import json
import threading
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace as _dataclass_replace
from importlib import resources
from types import SimpleNamespace

from .kernels import get_kernel
from ._runtime import launch_stream_context, resolve_launch_defaults


_DISPATCH_LAUNCH_OPTIONS = ContextVar("dispatch_launch_options", default=(None, None))


@contextmanager
def dispatch_launch_options(*, stream=None, timeout_ms=None):
    token = _DISPATCH_LAUNCH_OPTIONS.set((stream, timeout_ms))
    try:
        yield
    finally:
        _DISPATCH_LAUNCH_OPTIONS.reset(token)


def _resolved_launch_options(stream, timeout_ms):
    default_stream, default_timeout_ms = _DISPATCH_LAUNCH_OPTIONS.get()
    return (
        default_stream if stream is None else stream,
        default_timeout_ms if timeout_ms is None else timeout_ms,
    )


_active_launch_capture = ContextVar("flashlib_active_launch_capture", default=None)
_pending_tensor_map_recipe = ContextVar("flashlib_pending_tensor_map_recipe", default=None)
_launch_capture_prepare_lock = threading.RLock()


def _replace(value, /, **changes):
    replacer = getattr(value, "__replace__", None)
    if callable(replacer):
        return replacer(**changes)
    return _dataclass_replace(value, **changes)


dc = SimpleNamespace(replace=_replace)


def _import_dispatch_module(short_name):
    return importlib.import_module(f"{__package__}._dispatch.{short_name}")


_DISPATCH_OWNED_DICT_SUFFIXES = ("CACHE", "SCRATCH", "INPUTS", "OUTPUTS", "FLAGS")


def _cache_value_references_owned_object(value, owned_ids, seen):
    identity = id(value)
    if identity in owned_ids:
        return True
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, dict):
        return any(
            _cache_value_references_owned_object(item, owned_ids, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(
            _cache_value_references_owned_object(item, owned_ids, seen)
            for item in value
        )
    return False


def release_dispatch_caches(owned_objects):
    '''Clear route-owned tensor dictionaries after a prepared sequence binds.

    Generated dispatch modules may temporarily cache tensor-map descriptors and
    workspaces while a route is prepared.  A bound ``PreparedKernelSequence``
    retains every CUDA argument, so those module globals are no longer owners.
    To avoid clearing dispatch registries or scalar statistics, this contract
    is limited to dict-valued, private, uppercase names with an explicit
    workspace/cache suffix below this generated package's ``_dispatch``
    namespace.
    '''

    prefix = f"{__package__}._dispatch."
    owned_ids = {
        id(value)
        for value in owned_objects
        if callable(getattr(value, "data_ptr", None))
    }
    if not owned_ids:
        return 0
    cleared = 0
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(prefix):
            continue
        for name, value in tuple(vars(module).items()):
            if (
                name.startswith("_")
                and name.endswith(_DISPATCH_OWNED_DICT_SUFFIXES)
                and name.isupper()
                and isinstance(value, dict)
            ):
                removed = False
                for key, item in tuple(value.items()):
                    if _cache_value_references_owned_object(item, owned_ids, set()):
                        value.pop(key, None)
                        removed = True
                cleared += int(removed)
    return cleared


def _decode_capture(value):
    if isinstance(value, dict) and "__ir__" in value:
        return _ir_proxy(
            value["__ir__"],
            value.get("threads", 256),
            value.get("computed_smem_bytes", 0),
            value.get("cluster_dims", (1, 1, 1)),
            value.get("cta_group", 1),
            value.get("constants", ()),
            value.get("arg_keys", ()),
        )
    if isinstance(value, dict) and set(value) == {"__kernel__"}:
        return DispatchKernel(value["__kernel__"])
    if isinstance(value, dict) and set(value) == {"__kernel_source__"}:
        return value["__kernel_source__"]
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_decode_capture(item) for item in value["__tuple__"])
    if isinstance(value, dict) and set(value) == {"__dict_items__"}:
        return {
            _decode_capture(key): _decode_capture(item)
            for key, item in value["__dict_items__"]
        }
    if isinstance(value, dict):
        return {key: _decode_capture(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_capture(item) for item in value]
    return value


@dataclass(frozen=True)
class _IRProxy:
    symbol: str
    threads: int = 256
    computed_smem_bytes: int = 0
    constants: tuple = ()
    grid: object = None
    arg_keys: tuple = ()

    def __replace__(self, /, **changes):
        values = {
            "symbol": self.symbol,
            "threads": self.threads,
            "computed_smem_bytes": self.computed_smem_bytes,
            "constants": self.constants,
            "grid": self.grid,
            "arg_keys": self.arg_keys,
        }
        unknown = sorted(set(changes) - set(values))
        if unknown:
            raise TypeError(f"unknown frozen WeaveIR field(s): {unknown}")
        values.update(changes)
        return _IRProxy(**values)


def _ir_proxy(
    name, threads=256, computed_smem_bytes=0, cluster_dims=(1, 1, 1),
    cta_group=1, constants=(), arg_keys=(),
):
    return _IRProxy(
        name.rpartition(":")[2], int(threads), int(computed_smem_bytes),
        tuple(tuple(item) for item in constants),
        SimpleNamespace(cluster_dims=tuple(cluster_dims), cta_group=int(cta_group)),
        tuple(arg_keys),
    )


def pack_kernel_args(schedule, /, **bindings):
    expected = tuple(schedule.arg_keys)
    missing = sorted(set(expected) - set(bindings))
    unexpected = sorted(set(bindings) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"kernel argument bindings do not match frozen WeaveIR.args: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return [bindings[key] for key in expected]


class PreparedKernelSequence:
    def __init__(
        self,
        launches,
        result,
        input_bindings=(),
        result_template=None,
        tensor_map_bindings=(),
        input_alias_topology=(),
        stream=None,
    ):
        if not launches:
            raise RuntimeError("prepared semantic route did not capture a CUDA launch")
        self._launches = tuple(launches)
        self._result = result
        self._input_bindings = tuple(tuple(bindings) for bindings in input_bindings)
        if self._input_bindings and len(self._input_bindings) != len(self._launches):
            raise RuntimeError("prepared semantic route has corrupt input bindings")
        self._result_template = result_template
        self._input_alias_topology = tuple(
            tuple(group) for group in input_alias_topology
        )
        direct_input_keys = {key for bindings in self._input_bindings for _, key in bindings}
        self._direct_input_keys = tuple(sorted(direct_input_keys))
        self._tensor_map_bindings = _own_tensor_map_bindings(
            self._launches,
            tuple(tensor_map_bindings),
            stream=stream,
        )
        self._bound_input_keys = tuple(
            sorted(direct_input_keys | {binding.input_key for binding in self._tensor_map_bindings})
        )
        self._input_references_retained = True

    @property
    def launch_count(self):
        return len(self._launches)

    @property
    def bound_input_keys(self):
        return self._bound_input_keys

    def rebind_inputs(
        self,
        inputs,
        *,
        stream=None,
        materialize_result=True,
        preserve_prepared_stream=False,
        retain_input_references=True,
    ):
        if not isinstance(materialize_result, bool):
            raise TypeError("materialize_result must be a bool")
        if not isinstance(preserve_prepared_stream, bool):
            raise TypeError("preserve_prepared_stream must be a bool")
        if not isinstance(retain_input_references, bool):
            raise TypeError("retain_input_references must be a bool")
        if preserve_prepared_stream and stream is not None:
            raise ValueError("preserve_prepared_stream requires stream=None")
        if materialize_result and not retain_input_references:
            raise ValueError(
                "materialize_result requires retain_input_references so the result remains valid"
            )
        if not any(self._input_bindings) and not self._tensor_map_bindings:
            raise RuntimeError(
                "prepared semantic route has no input bindings; "
                "capture it with capture_kernel_launches(inputs=...)"
            )
        missing = sorted(set(self.bound_input_keys) - set(inputs))
        if missing:
            raise KeyError(f"missing prepared semantic input binding(s): {missing!r}")
        _validate_public_tensor_alias_topology(inputs, self._input_alias_topology)
        pointer_values = None
        inputs_already_scrubbed = False
        if not retain_input_references:
            pointer_values = {}
            for key in self._direct_input_keys:
                value = inputs[key]
                data_ptr = getattr(value, "data_ptr", None)
                if not callable(data_ptr):
                    raise TypeError(f"prepared CUDA tensor binding {key!r} is not tensor-like")
                pointer_values[key] = int(data_ptr())
            inputs_already_scrubbed = not self._input_references_retained
        with launch_stream_context(stream):
            for binding in self._tensor_map_bindings:
                binding.refresh(inputs[binding.input_key])
            for launch, bindings in zip(self._launches, self._input_bindings, strict=True):
                launch.rebind_tensor_arguments(
                    bindings,
                    inputs,
                    stream=stream,
                    preserve_stream=preserve_prepared_stream,
                    retain_inputs=retain_input_references,
                    pointer_values=pointer_values,
                    inputs_already_scrubbed=inputs_already_scrubbed,
                )
        self._input_references_retained = retain_input_references
        # Stateful public runtimes may own the output independently of the
        # semantic return tree.  Let those callers skip recursively rebuilding
        # a result they will not observe while retaining the default behavior
        # for normal prepared-dispatch callers.
        if materialize_result and self._result_template is not None:
            self._result = _materialize_result_template(self._result_template, inputs)
        return self

    def _rebind_stream_bound_scrubbed_inputs(self, inputs, *, stream):
        '''Rebind one validated fixed-stream runtime slot without generic checks.

        This private path is valid only after a stateful wrapper selected the
        sequence through a cache key containing the complete public pointer
        alias topology, recorded every caller-owned tensor, and scrubbed the
        sequence's caller references. Public prepared callers continue to use
        :meth:`rebind_inputs` and its full validation.
        '''
        if self._input_references_retained:
            raise RuntimeError(
                "fixed-stream semantic rebind requires scrubbed input references"
            )
        if stream is None:
            raise ValueError("fixed-stream semantic rebind requires an explicit stream")
        pointer_values = {}
        for key in self._direct_input_keys:
            value = inputs[key]
            data_ptr = getattr(value, "data_ptr", None)
            if not callable(data_ptr):
                raise TypeError(f"prepared CUDA tensor binding {key!r} is not tensor-like")
            pointer_values[key] = int(data_ptr())
        for binding in self._tensor_map_bindings:
            binding.rebind_stream_bound(inputs[binding.input_key], stream=stream)
        for launch, bindings in zip(self._launches, self._input_bindings, strict=True):
            launch.rebind_tensor_arguments(
                bindings,
                inputs,
                preserve_stream=True,
                retain_inputs=False,
                pointer_values=pointer_values,
                inputs_already_scrubbed=True,
            )
        return self

    def release_bound_inputs(self):
        '''Drop caller tensor references after their launch stream was recorded.'''
        if self._input_references_retained:
            for launch, bindings in zip(self._launches, self._input_bindings, strict=True):
                keepalive = list(launch._keepalive)
                for index, _key in bindings:
                    value = keepalive[index]
                    data_ptr = getattr(value, "data_ptr", None)
                    if callable(data_ptr):
                        keepalive[index] = int(data_ptr())
                launch._keepalive = tuple(keepalive)
            self._input_references_retained = False
        self._result = None

    def record_stream(self, stream):
        '''Tie every tensor launch argument, including private scratch, to a stream.'''
        if stream is None:
            raise ValueError("prepared semantic record_stream requires an explicit stream")
        seen = set()
        for launch in self._launches:
            for value in launch._keepalive:
                identity = id(value)
                record_stream = getattr(value, "record_stream", None)
                if identity not in seen and callable(record_stream):
                    seen.add(identity)
                    record_stream(stream)
        # Variant-bank descriptor tensors are slot-owned but only the active
        # variant appears in a launch keepalive; record every variant so a
        # non-synchronizing release stays allocator-safe.
        for binding in self._tensor_map_bindings:
            for value in binding.variants.values():
                identity = id(value)
                record_stream = getattr(value, "record_stream", None)
                if identity not in seen and callable(record_stream):
                    seen.add(identity)
                    record_stream(stream)

    def _finish_rebind(self, result):
        self._result = result
        return self

    def __call__(self, _inputs=None, *, stream=None, timeout_ms=None):
        last = len(self._launches) - 1
        for index, launch in enumerate(self._launches):
            launch.launch(stream=stream, timeout_ms=timeout_ms if index == last else None)
        return self._result


class KernelLaunchCapture:
    def __init__(self, *, stream=None, arch=None, inputs=None, rebind=None):
        if rebind is not None and not isinstance(rebind, PreparedKernelSequence):
            raise TypeError("rebind must be a PreparedKernelSequence")
        if inputs is not None and rebind is not None:
            raise ValueError("inputs and rebind are mutually exclusive capture modes")
        if rebind is not None:
            raise RuntimeError(
                "capture(rebind=...) is unsupported because an in-place topology "
                "update cannot be transactional; capture a new sequence instead"
            )
        self.stream = stream
        self.arch = arch
        self._launches = []
        self._input_bindings = []
        self._input_key_by_identity = _public_tensor_input_identities(inputs)
        self._input_key_by_pointer = _public_tensor_input_pointers(inputs)
        self._input_alias_topology = _public_tensor_alias_topology(inputs)
        self._tensor_map_bindings = {}
        self._route_caches_released = False
        self._rebind = rebind
        self._rebind_index = 0
        self.host_data_reads = 0

    @property
    def host_data_dependent(self):
        '''True when the route read device memory while its kernels were only
        being recorded — its host branch decisions cannot be frozen.'''
        return self.host_data_reads > 0

    @property
    def rebinding(self):
        return self._rebind is not None

    def add(self, launch):
        if self.rebinding:
            raise RuntimeError("rebind capture requires launch topology, not a newly prepared launch")
        self._launches.append(launch)
        self._input_bindings.append(
            ()
            if not self._input_key_by_identity
            else tuple(
                (index, self._input_key_by_identity[id(arg)])
                for index, arg in enumerate(launch._keepalive)
                if id(arg) in self._input_key_by_identity
            )
        )
        for arg in launch._keepalive:
            recipe = getattr(arg, "_loom_tensor_map_recipe", None)
            if recipe is None:
                continue
            source_pointer = int(recipe[2])
            input_key = self._input_key_by_pointer.get(source_pointer)
            if input_key is None:
                continue
            self._tensor_map_bindings.setdefault(
                id(arg),
                _TensorMapBinding(
                    input_key=input_key,
                    tensor=arg,
                    recipe=tuple(recipe),
                    pointer=source_pointer,
                ),
            )

    def add_kernel_launch(
        self,
        exported,
        *,
        mode,
        grid,
        block,
        args,
        arg_types,
        shared_mem,
        stream,
        cluster_dims=None,
    ):
        resolved_arch, resolved_stream, _ = resolve_launch_defaults(
            arch=self.arch,
            stream=self.stream if self.stream is not None else stream,
            timeout_ms=None,
        )
        with launch_stream_context(resolved_stream):
            kernel = exported.compile(arch=resolved_arch, options=["--use_fast_math"])
            kwargs = {
                "grid": grid,
                "block": block,
                "args": tuple(args),
                "arg_types": arg_types,
                "shared_mem": shared_mem,
                "stream": resolved_stream,
            }
            if self.rebinding:
                if self._rebind_index >= self._rebind.launch_count:
                    raise RuntimeError(
                        "prepared semantic route launch-count mismatch: "
                        f"expected {self._rebind.launch_count}, captured more launches"
                    )
                prepared = self._rebind._launches[self._rebind_index]
                if mode == "cluster":
                    kernel.rebind_launch_cluster(
                        prepared, cluster_dims=cluster_dims, **kwargs
                    )
                elif mode == "cooperative":
                    kernel.rebind_launch_cooperative(prepared, **kwargs)
                elif mode == "regular":
                    kernel.rebind_launch(prepared, **kwargs)
                else:
                    raise RuntimeError(f"unsupported captured launch mode: {mode!r}")
                self._rebind_index += 1
                return
            if mode == "cluster":
                prepared = kernel.prepare_launch_cluster(
                    cluster_dims=cluster_dims, **kwargs
                )
            elif mode == "cooperative":
                prepared = kernel.prepare_launch_cooperative(**kwargs)
            elif mode == "regular":
                prepared = kernel.prepare_launch(**kwargs)
            else:
                raise RuntimeError(f"unsupported captured launch mode: {mode!r}")
            self.add(prepared)

    def bind(self, result):
        if self.rebinding:
            if self._rebind_index != self._rebind.launch_count:
                raise RuntimeError(
                    "prepared semantic route launch-count mismatch: "
                    f"expected {self._rebind.launch_count}, captured {self._rebind_index}"
                )
            return self._rebind._finish_rebind(result)
        result_template = _capture_result_template(result, self._input_key_by_identity)
        route_cache_owned_objects = self._route_cache_owned_objects()
        sequence = PreparedKernelSequence(
            self._launches,
            result,
            self._input_bindings,
            result_template,
            tuple(self._tensor_map_bindings.values()),
            self._input_alias_topology,
            self.stream,
        )
        self.release_route_caches(route_cache_owned_objects)
        return sequence

    def _route_cache_owned_objects(self):
        return tuple(arg for launch in self._launches for arg in launch._keepalive)

    def release_route_caches(self, owned_objects=None):
        if self._route_caches_released:
            return 0
        if owned_objects is None:
            owned_objects = self._route_cache_owned_objects()
        self._route_caches_released = True
        return release_dispatch_caches(tuple(owned_objects))


@dataclass(frozen=True)
class _BoundInputResult:
    key: str


@dataclass
class _TensorMapBinding:
    input_key: str
    tensor: object
    recipe: tuple
    pointer: int
    pointer_carriers: tuple = ()
    staging_slots: list = field(default_factory=list)
    variants: dict = field(default_factory=dict)
    variant_capacity: int = 4

    def __post_init__(self):
        if not self.variants:
            self.variants[self.pointer] = self.tensor

    def _acquire_staging_slot(self, torch):
        for slot in self.staging_slots:
            if slot.event.query():
                return slot
        slot = _TensorMapStagingSlot(
            host_buffer=torch.empty(128, dtype=torch.uint8, pin_memory=True),
            event=torch.cuda.Event(blocking=False, interprocess=False),
        )
        self.staging_slots.append(slot)
        return slot

    def _encode_into(self, pointer, tensor, *, stream=None):
        from cuda.bindings import driver
        import torch

        arguments = list(self.recipe)
        arguments[2] = pointer
        err, tmap = driver.cuTensorMapEncodeTiled(*arguments)
        if err != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled rebind failed: CUresult={err}")
        slot = self._acquire_staging_slot(torch)
        ctypes.memmove(
            int(slot.host_buffer.data_ptr()),
            int(tmap.getPtr()),
            128,
        )
        tensor.copy_(slot.host_buffer, non_blocking=True)
        slot.event.record(torch.cuda.current_stream() if stream is None else stream)
        self.tensor = tensor
        self.pointer = pointer
        self.recipe = tuple(arguments)
        tensor._loom_tensor_map_recipe = self.recipe

    def _activate(self, pointer, tensor):
        descriptor_pointer = int(tensor.data_ptr())
        for carrier in self.pointer_carriers:
            carrier.value = descriptor_pointer
        arguments = list(self.recipe)
        arguments[2] = pointer
        self.tensor = tensor
        self.pointer = pointer
        self.recipe = tuple(arguments)
        tensor._loom_tensor_map_recipe = self.recipe

    def refresh(self, source, *, stream=None):
        pointer = int(source.data_ptr())
        if pointer == self.pointer:
            return
        self._encode_into(pointer, self.tensor, stream=stream)
        # Generic public rebinding mutates the active descriptor in place.
        # Reset the private variant bank so no stale pointer key can name it.
        self.variants.clear()
        self.variants[pointer] = self.tensor

    def rebind_stream_bound(self, source, *, stream):
        if stream is None:
            raise ValueError("stream-bound tensor-map rebind requires an explicit stream")
        pointer = int(source.data_ptr())
        if pointer == self.pointer:
            return
        cached = self.variants.pop(pointer, None)
        if cached is not None:
            # LRU recency: re-insert the hit so eviction removes the
            # least-recently-activated variant, not the newest one.
            self.variants[pointer] = cached
            self._activate(pointer, cached)
            return

        import torch

        if len(self.variants) < self.variant_capacity:
            tensor = torch.empty_like(self.tensor)
            record_stream = getattr(tensor, "record_stream", None)
            if callable(record_stream):
                record_stream(stream)
        else:
            tensor = self.variants.pop(next(iter(self.variants)))
        self._encode_into(pointer, tensor, stream=stream)
        self.variants[pointer] = tensor
        self._activate(pointer, tensor)


@dataclass
class _TensorMapStagingSlot:
    host_buffer: object
    event: object


def _own_tensor_map_bindings(launches, bindings, *, stream):
    '''Clone cached descriptors and patch every launch to slot-owned storage.'''

    if not bindings:
        return ()
    owned_by_identity = {}
    owned_bindings = []
    with launch_stream_context(stream):
        for binding in bindings:
            original = binding.tensor
            owned = original.clone()
            owned._loom_tensor_map_recipe = binding.recipe
            metadata = getattr(original, "_loom_tma_metadata", None)
            if metadata is not None:
                owned._loom_tma_metadata = metadata
            owned_by_identity[id(original)] = owned
            pointer_carriers = tuple(
                launch._packed._prevent_gc[index]
                for launch in launches
                for index, arg in enumerate(launch._keepalive)
                if id(arg) == id(original)
            )
            if not pointer_carriers or any(
                type(carrier) is not ctypes.c_void_p for carrier in pointer_carriers
            ):
                raise RuntimeError("captured tensor-map binding has invalid pointer carriers")
            owned_bindings.append(
                _TensorMapBinding(
                    input_key=binding.input_key,
                    tensor=owned,
                    recipe=binding.recipe,
                    pointer=binding.pointer,
                    pointer_carriers=pointer_carriers,
                )
            )
        for launch in launches:
            replacements = {
                index: owned_by_identity[id(arg)]
                for index, arg in enumerate(launch._keepalive)
                if id(arg) in owned_by_identity
            }
            if replacements:
                launch.rebind_arguments(replacements, stream=stream)
    return tuple(owned_bindings)


def _public_tensor_input_identities(inputs):
    if inputs is None:
        return {}
    if not hasattr(inputs, "items"):
        raise TypeError("capture inputs must be a mapping")
    identities = {}
    for key, value in inputs.items():
        if (
            isinstance(key, str)
            and not key.startswith("_")
            and callable(getattr(value, "data_ptr", None))
        ):
            identities.setdefault(id(value), key)
    return identities


def _public_tensor_input_pointers(inputs):
    if inputs is None:
        return {}
    pointers = {}
    for key, value in inputs.items():
        if (
            isinstance(key, str)
            and not key.startswith("_")
            and callable(getattr(value, "data_ptr", None))
        ):
            pointers.setdefault(int(value.data_ptr()), key)
    return pointers


def _public_tensor_alias_topology(inputs, keys=None):
    '''Return the complete pointer-equality partition of public tensor inputs.'''

    if inputs is None:
        return ()
    if not hasattr(inputs, "items"):
        raise TypeError("capture inputs must be a mapping")
    if keys is None:
        selected = [
            key
            for key, value in inputs.items()
            if (
                isinstance(key, str)
                and not key.startswith("_")
                and callable(getattr(value, "data_ptr", None))
            )
        ]
    else:
        selected = list(keys)
        missing = sorted(set(selected) - set(inputs))
        if missing:
            raise KeyError(f"missing prepared semantic alias binding(s): {missing!r}")
        invalid = sorted(
            key
            for key in selected
            if not callable(getattr(inputs[key], "data_ptr", None))
        )
        if invalid:
            raise TypeError(
                f"prepared semantic alias binding(s) must be tensor-like: {invalid!r}"
            )
    groups = {}
    for key in selected:
        groups.setdefault(int(inputs[key].data_ptr()), []).append(key)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def _validate_public_tensor_alias_topology(inputs, expected):
    if not expected:
        return
    keys = tuple(key for group in expected for key in group)
    actual = _public_tensor_alias_topology(inputs, keys)
    if actual != expected:
        raise RuntimeError(
            "prepared semantic public tensor alias topology changed: "
            f"expected {expected!r}, got {actual!r}; capture a new sequence"
        )


def _capture_result_template(value, input_key_by_identity):
    key = input_key_by_identity.get(id(value))
    if key is not None:
        return _BoundInputResult(key)
    if isinstance(value, tuple):
        return tuple(_capture_result_template(item, input_key_by_identity) for item in value)
    if isinstance(value, list):
        return [_capture_result_template(item, input_key_by_identity) for item in value]
    if isinstance(value, dict):
        return {
            key: _capture_result_template(item, input_key_by_identity)
            for key, item in value.items()
        }
    return value


def _materialize_result_template(value, inputs):
    if isinstance(value, _BoundInputResult):
        return inputs[value.key]
    if isinstance(value, tuple):
        return tuple(_materialize_result_template(item, inputs) for item in value)
    if isinstance(value, list):
        return [_materialize_result_template(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: _materialize_result_template(item, inputs) for key, item in value.items()}
    return value


@contextmanager
def capture_kernel_launches(*, stream=None, arch=None, inputs=None, rebind=None):
    import torch

    with _launch_capture_prepare_lock:
        if _active_launch_capture.get() is not None:
            raise RuntimeError("nested kernel launch capture is not supported")
        capture = KernelLaunchCapture(stream=stream, arch=arch, inputs=inputs, rebind=rebind)
        token = _active_launch_capture.set(capture)
        # Captured launches are marshalled, not run, so any device read the
        # route performs mid-traversal (for example an ``overflow_flag.item()``
        # certification) observes memory its recorded kernels never wrote.
        # Interpose ``torch.Tensor.item`` for the capture's duration and count
        # CUDA-tensor reads; a nonzero count marks the capture
        # ``host_data_dependent`` so the plan builder keeps that signature on
        # the per-call launcher instead of freezing an unreproducible branch.
        # Captures are serialized by ``_launch_capture_prepare_lock``, so the
        # process-global interpose cannot nest. ``.item()`` is the only device
        # read the vendored dispatch modules perform on their launch paths.
        # Torch test doubles without a ``Tensor.item`` skip the interpose.
        original_tensor_item = getattr(getattr(torch, "Tensor", None), "item", None)

        def _observed_item(tensor):
            if getattr(tensor, "is_cuda", False):
                capture.host_data_reads += 1
            return original_tensor_item(tensor)

        if original_tensor_item is not None:
            torch.Tensor.item = _observed_item
        try:
            if stream is None:
                yield capture
            else:
                with torch.cuda.stream(stream):
                    yield capture
        finally:
            if original_tensor_item is not None:
                torch.Tensor.item = original_tensor_item
            capture.release_route_caches()
            _active_launch_capture.reset(token)


class DispatchKernel:
    def __init__(self, alias, symbol=None):
        self.exported = get_kernel(alias)
        self.symbol = symbol or self.exported.spec.symbol

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def launch(self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        capture = _active_launch_capture.get()
        if capture is not None:
            capture.add_kernel_launch(
                self.exported,
                mode="regular",
                grid=grid,
                block=block,
                args=args,
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem,
                stream=stream,
            )
            return
        self.exported.launch(
            *args, grid=grid, block=block, shared_mem=shared_mem, stream=stream,
            timeout_ms=timeout_ms, options=["--use_fast_math"],
        )

    def launch_cluster(
        self, *, grid, block, args, cluster_dims, shared_mem=0, stream=None,
        timeout_ms=None, **kwargs
    ):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        capture = _active_launch_capture.get()
        if capture is not None:
            capture.add_kernel_launch(
                self.exported,
                mode="cluster",
                grid=grid,
                block=block,
                args=args,
                arg_types=self.exported.arg_types,
                cluster_dims=cluster_dims,
                shared_mem=shared_mem,
                stream=stream,
            )
            return
        arch, stream, timeout_ms = resolve_launch_defaults(
            arch=None,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        with launch_stream_context(stream):
            kernel = self.exported.compile(arch=arch, options=["--use_fast_math"])
            kernel.launch_cluster(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                cluster_dims=cluster_dims, shared_mem=shared_mem, stream=stream,
                timeout_ms=timeout_ms,
            )

    def launch_cooperative(
        self, *, grid, block, args, shared_mem=0, stream=None, timeout_ms=None, **kwargs
    ):
        stream, timeout_ms = _resolved_launch_options(stream, timeout_ms)
        capture = _active_launch_capture.get()
        if capture is not None:
            capture.add_kernel_launch(
                self.exported,
                mode="cooperative",
                grid=grid,
                block=block,
                args=args,
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem,
                stream=stream,
            )
            return
        arch, stream, timeout_ms = resolve_launch_defaults(
            arch=None,
            stream=stream,
            timeout_ms=timeout_ms,
        )
        with launch_stream_context(stream):
            kernel = self.exported.compile(arch=arch, options=["--use_fast_math"])
            kernel.launch_cooperative(
                grid=grid, block=block, args=tuple(args),
                arg_types=self.exported.arg_types,
                shared_mem=shared_mem, stream=stream, timeout_ms=timeout_ms,
            )

    def prepare_launch(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch, kwargs)

    def prepare_launch_cluster(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch_cluster, kwargs)

    def prepare_launch_cooperative(self, **kwargs):
        return _PreparedDispatchLaunch(self.launch_cooperative, kwargs)


class _PreparedDispatchLaunch:
    def __init__(self, launch, kwargs):
        self._launch = launch
        self._kwargs = dict(kwargs)

    def launch(self, timeout_ms=None):
        kwargs = dict(self._kwargs)
        if timeout_ms is not None:
            kwargs["timeout_ms"] = timeout_ms
        return self._launch(**kwargs)


CUDAKernel = DispatchKernel


def compile_cuda(source, **kwargs):
    return source


def detect_gpu_arch():
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def _cuda_include_dirs():
    return []


def arch_flag_for_cc(major, minor):
    sm = int(major) * 10 + int(minor)
    return f"sm_{sm}a" if sm >= 90 else f"sm_{sm}"


def _capture_cuTensorMapEncodeTiled(*arguments):
    '''Encode a tensor map while retaining a pointer-rebind recipe.'''
    from cuda.bindings import driver

    result = driver.cuTensorMapEncodeTiled(*arguments)
    if result[0] == 0:
        _pending_tensor_map_recipe.set(tuple(arguments))
    else:
        _pending_tensor_map_recipe.set(None)
    return result


def _tmap_to_device(tmap, metadata=None):
    import torch
    del metadata
    recipe = _pending_tensor_map_recipe.get()
    _pending_tensor_map_recipe.set(None)
    host_ptr = tmap.getPtr()
    raw = bytes((ctypes.c_ubyte * 128).from_address(host_ptr))
    host = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    device = torch.empty(128, dtype=torch.uint8, device="cuda")
    device.copy_(host)
    if recipe is not None:
        device._loom_tensor_map_recipe = recipe
    return device


class Swizzle:
    # Standalone spellings used only by stripped TMA metadata helpers.
    SZ_128B = "128B"
    SZ_64B = "64B"
    SZ_32B = "32B"
    NONE = "none"


class TensorMapMetadata:
    # Compatibility carrier; frozen launch packing needs no metadata.
    def __init__(self, **values):
        self.__dict__.update(values)


def attach_tma_metadata(tensor, metadata):
    tensor._loom_tensor_map_metadata = metadata
    return tensor


def create_tensor_map(data_ptr, dim0, dim1, box0, box1, stride1_bytes):
    '''Create a rank-2 BF16 128B-swizzled map with a rebind recipe.'''
    from cuda.bindings import driver

    err, tmap = _capture_cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        data_ptr,
        [driver.cuuint64_t(dim0), driver.cuuint64_t(dim1)],
        [driver.cuuint64_t(stride1_bytes)],
        [driver.cuuint32_t(box0), driver.cuuint32_t(box1)],
        [driver.cuuint32_t(1), driver.cuuint32_t(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled (2D BF16) failed: CUresult={err}")
    return _tmap_to_device(tmap)


def _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, swizzle):
    from cuda.bindings import driver
    atoms = {"128B": 64, "64B": 32, "32B": 16}
    swizzles = {
        "128B": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        "64B": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        "32B": driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    }
    try:
        atom = atoms[swizzle]
        swizzle_value = swizzles[swizzle]
    except KeyError as exc:
        raise ValueError(f"unsupported 3D tensor-map swizzle: {swizzle}") from exc
    err, tmap = _capture_cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        data_ptr,
        [driver.cuuint64_t(atom), driver.cuuint64_t(global_height), driver.cuuint64_t(width // atom)],
        [driver.cuuint64_t(width * 2), driver.cuuint64_t(atom * 2)],
        [driver.cuuint32_t(atom), driver.cuuint32_t(shared_height), driver.cuuint32_t(block_width // atom)],
        [driver.cuuint32_t(1), driver.cuuint32_t(1), driver.cuuint32_t(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_value,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: CUresult={err}")
    return _tmap_to_device(tmap)


def create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "128B")


def create_tensor_map_3d_64b(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "64B")


def create_tensor_map_3d_32b(data_ptr, global_height, shared_height, width, block_width):
    return _create_tensor_map_3d(data_ptr, global_height, shared_height, width, block_width, "32B")


def generate_kernel(ir, **kwargs):
    request_key = json.dumps(
        {
            "ir_name": ir.symbol,
            "constants": [[str(name), value] for name, value in ir.constants],
            "threads": int(ir.threads),
            "computed_smem_bytes": int(ir.computed_smem_bytes),
            "kwargs": kwargs,
        },
        sort_keys=True, separators=(",", ":"), default=repr,
    )
    alias = _KERNEL_ALIAS_BY_REQUEST.get(request_key)
    if alias is None:
        alias = _KERNEL_ALIAS_BY_IR_NAME.get(ir.symbol)
    if alias is None:
        raise RuntimeError(f"uncaptured dispatcher specialization for {ir.symbol}")
    return alias


def generate_kernel_bundle(*args, **kwargs):
    raise RuntimeError("uncaptured dispatcher bundle specialization")


def _all_shapes():
    package = __package__ or __name__.rpartition(".")[0]
    text = resources.files(package).joinpath("_dispatch_shapes.json").read_text(encoding="utf-8")
    return json.loads(text)


class _CanonicalShapes:
    '''Lazy contract-shape view backed by the exported plan ledger.'''

    def __iter__(self):
        return iter(_all_shapes())

    def __len__(self):
        return len(_all_shapes())

    def __getitem__(self, index):
        return _all_shapes()[index]


CANONICAL_SHAPES = _CanonicalShapes()


def select_named_shapes(labels):
    labels = [labels] if isinstance(labels, str) else list(labels)
    by_label = {row["label"]: row for row in _all_shapes()}
    return [by_label[label] for label in labels]


def evaluate(*args, **kwargs):
    raise RuntimeError("Cake eval harness is not part of the standalone runtime")
