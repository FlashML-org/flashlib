"""Weave-only 24-row floor-1.40 kNN portfolio.

Minimum target architecture: sm_100a for the exact tcgen05 routes.  This
dispatcher preserves the inherited Weave portfolio for every guard miss.  The
four expanded-coverage repairs are exact seed guards; the forced route remains
a fallback-class route and only handles its explicit ``force_fallback`` input.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0630_d15e_af19_portfolio_floor14_7b75_v1 as parent
from . import knn_search_coverage_guard_d4096_q4_m32767_q4tail239_v1 as q4_tail
from . import knn_search_coverage_tail_d4096_q5_m32769_d510_v1 as q5_tail
from . import knn_search_forced_fallback_d4096_q4_m32768_b173cb3b0b43_v1 as forced
from . import knn_search_d1024q8_blockm256_retune_shared_q8_v1 as q8_v2
from . import knn_search_target0627_d768_q16_m65536_k10_4950_split2_v1 as d768_q16_split2
from . import knn_search_target0630_d256_q24_m49152_k10_d256specializedseed_v2 as d256
from . import knn_search_target0627_d256_q128_m262144_k64_fanin8cta_blockm64_891a_v1 as d256_q128_eb35
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0630_floor14_expanded_coverage_synthesis_6675_v1:launch_for_eval'
CONSUMED_SEEDS = ('weave-evolve-knn-search-6f4b', 'weave-evolve-knn-search-cf0a', 'weave-evolve-knn-search-q4tail239', 'weave-evolve-knn-search-d510-q5-tail', 'weave-evolve-knn-search-d256specializedseed-v2', 'weave-evolve-knn-search-eb35', 'weave-evolve-knn-search-d15e-forced-fallback')
PROFILE_ALL, TARGET_LABELS, TARGET_SHAPES = (parent.PROFILE_ALL, parent.TARGET_LABELS, parent.TARGET_SHAPES)
ir, parent_ir = (parent.ir, parent.parent_ir)
_ENTRIES = ({'shape_key': d256_q128_eb35.TARGET_LABELS[0], 'route': d256_q128_eb35.ROUTE, 'entrypoint': d256_q128_eb35.ENTRYPOINT, 'seed': 'weave-evolve-knn-search-eb35', 'guard': 'B == 1 and Q == 128 and M == 262144 and D == 256 and K == 64 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}'}, {'shape_key': q8_v2.TARGET_LABELS[0], 'route': q8_v2.ROUTE, 'entrypoint': q8_v2.ENTRYPOINT, 'seed': 'weave-evolve-knn-search-6f4b', 'guard': 'B == 1 and Q == 8 and M == 65536 and D == 1024 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}'}, {'shape_key': d768_q16_split2.TARGET_LABELS[0], 'route': d768_q16_split2.ROUTE, 'entrypoint': 'loom.examples.weave.knn_search_target0627_d768_q16_m65536_k10_4950_split2_v1:launch_for_eval', 'seed': 'weave-evolve-knn-search-cf0a', 'guard': 'B == 1 and Q == 16 and M == 65536 and D == 768 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}'}, {'shape_key': q4_tail.TARGET_SHAPE_KEY, 'route': q4_tail.ROUTE, 'entrypoint': q4_tail.ENTRYPOINT, 'seed': 'weave-evolve-knn-search-q4tail239', 'guard': 'exact q4/m32767 nonself, non-forced tcgen05'}, {'shape_key': q5_tail.TARGET_SHAPE_KEY, 'route': q5_tail.ROUTE, 'entrypoint': q5_tail.ENTRYPOINT, 'seed': 'weave-evolve-knn-search-d510-q5-tail', 'guard': 'exact q5/m32769 nonself, non-forced tcgen05'}, {'shape_key': d256.TARGET_LABEL, 'route': d256.ROUTE, 'entrypoint': d256.ENTRYPOINT, 'seed': d256.SELECTED_SEED, 'guard': d256.SHAPE_DISPATCH_REGISTRY[0]['guard']}, {'shape_key': forced.TARGET_SHAPE_KEY, 'route': forced.ROUTE, 'entrypoint': forced.ENTRYPOINT, 'seed': 'weave-evolve-knn-search-d15e-forced-fallback', 'guard': forced._ENTRY['guard']})
SHAPE_DISPATCH_REGISTRY = (*_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)

def __getattr__(name: str) -> Any:
    """Preserve the parent dispatcher contract entrypoints under this wrapper."""
    return getattr(parent, name)

def _selected(inputs: dict[str, Any]) -> tuple[dict[str, str], Any] | None:
    for entry, module in ((_ENTRIES[6], forced), (_ENTRIES[0], d256_q128_eb35), (_ENTRIES[1], q8_v2), (_ENTRIES[2], d768_q16_split2), (_ENTRIES[3], q4_tail), (_ENTRIES[4], q5_tail), (_ENTRIES[5], d256)):
        if module in {q8_v2, d768_q16_split2} and bool(inputs.get('force_fallback', False)):
            continue
        selector = getattr(module, 'selected_route', None)
        if selector is None:
            selector = module.selected_route_name
        if selector(inputs) == entry['route']:
            return (entry, module)
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    selected = _selected(inputs)
    return selected[0]['route'] if selected is not None else parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    selected = _selected(inputs)
    if selected is None:
        info = dict(parent.route_info(inputs))
        info['dispatcher_entrypoint'] = ENTRYPOINT
        info['guard_order'] = [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY]
        return info
    entry, module = selected
    info = dict(module.route_info(inputs))
    info.update({'dispatcher_entrypoint': ENTRYPOINT, 'selected_entrypoint': entry['entrypoint'], 'selected_seed': entry['seed'], 'expected_seed': entry['seed'], 'guard_id': entry['shape_key'], 'guard_condition': entry['guard'], 'selected_guard': entry['guard'], 'guard_order': [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY], 'production_policy': 'weave_only', 'external_fallback': None})
    if module is forced:
        info.update({'route_kind': 'fallback', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'forced_fallback': True})
    else:
        info.update({'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    selected = _selected(inputs)
    return selected[1].launch_for_eval(inputs) if selected is not None else parent.launch_for_eval(inputs)
