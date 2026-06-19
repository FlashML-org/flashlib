"""CuteDSL (Hopper SM90) backend for CAGRA search.

The Triton single-CTA search kernel is fundamentally register-bound: its
sorted candidate buffer + O(itopk^2) dedup live in registers (ncu shows
121-255 regs/thread, 12-23% occupancy), and Triton exposes no explicit
shared memory to move them off-chip. This CuteDSL path keeps the candidate
buffer + a visited hashmap in SMEM and computes neighbour distances with
warp-teams (coalesced row loads), targeting the gather-bandwidth /
compute rooflines a graph walk can actually reach.

Public probe: :func:`cutedsl_available`. The search entry point falls back
to the Triton kernel for any unsupported shape or a first-call compile
failure, so this is always safe to attempt.
"""
