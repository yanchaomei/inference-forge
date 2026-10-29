"""Reduce external request traces while rechecking the original failure."""

from copy import deepcopy
from math import ceil


def reduce_trace(events, probe, signature, *, repeats=2, max_probes=32):
    if repeats < 1 or max_probes < repeats:
        raise ValueError("budget must cover at least one repeated confirmation")
    current = deepcopy(events)
    checks = []
    probe_count = 0

    def confirm(candidate):
        nonlocal probe_count
        if probe_count + repeats > max_probes:
            return None
        observations = []
        for _ in range(repeats):
            result = probe(deepcopy(candidate))
            probe_count += 1
            observations.append(result)
        matching = sum(r.get("status") == "fail" and r.get("signature") == signature for r in observations)
        checks.append({"request_ids": [e.get("request_id") for e in candidate],
                       "attempts": repeats, "failures": matching, "observations": observations})
        return matching == repeats

    original_ok = confirm(current)
    if not original_ok:
        return {"status": "not_stably_reproduced", "events": current,
                "probe_count": probe_count, "confirmations": checks, "signature": signature}
    partitions = 2
    exhausted = False
    while len(current) > 1:
        width = ceil(len(current) / partitions)
        reduced = False
        for start in range(0, len(current), width):
            candidate = current[:start] + current[start + width:]
            if not candidate:
                continue
            matched = confirm(candidate)
            if matched is None:
                exhausted = True
                break
            if matched:
                current = candidate
                partitions = max(2, partitions - 1)
                reduced = True
                break
        if exhausted:
            break
        if not reduced:
            if partitions >= len(current):
                break
            partitions = min(len(current), partitions * 2)
    final_status = "budget_exhausted" if exhausted else "reduced"
    if not exhausted:
        final_match = confirm(current)
        if final_match is None:
            final_status = "budget_exhausted"
        elif not final_match:
            final_status = "not_stably_reproduced"
    return {"status": final_status, "events": current,
            "probe_count": probe_count, "confirmations": checks, "signature": signature,
            "scope": "external request deletion; no internal scheduler control or minimality proof"}
