from forge.replay import reduce_trace


def test_preserves_same_order_failure_and_removes_unrelated_requests():
    events = [{"request_id": x, "max_tokens": 4} for x in ["noise", "open", "resume", "other"]]

    def probe(trace):
        names = [x["request_id"] for x in trace]
        if "open" in names and "resume" in names and names.index("open") < names.index("resume"):
            return {"status": "fail", "signature": "resume_mismatch"}
        return {"status": "pass"}

    result = reduce_trace(events, probe, "resume_mismatch", repeats=2)
    assert result["status"] == "reduced"
    assert [e["request_id"] for e in result["events"]] == ["open", "resume"]
    assert result["confirmations"][-1]["failures"] == 2


def test_different_failure_cannot_replace_original_failure():
    events = [{"request_id": "required"}, {"request_id": "irrelevant"}]

    def probe(trace):
        if any(e["request_id"] == "required" for e in trace):
            return {"status": "fail", "signature": "original"}
        return {"status": "fail", "signature": "different"}

    result = reduce_trace(events, probe, "original")
    assert [e["request_id"] for e in result["events"]] == ["required"]


def test_unstable_original_is_reported_without_claiming_reduction():
    calls = 0

    def probe(trace):
        nonlocal calls
        calls += 1
        return {"status": "fail", "signature": "wanted"} if calls == 1 else {"status": "pass"}

    events = [{"request_id": "a"}, {"request_id": "b"}]
    result = reduce_trace(events, probe, "wanted", repeats=2)
    assert result["status"] == "not_stably_reproduced"
    assert result["events"] == events


def test_probe_budget_is_a_hard_limit_and_does_not_fabricate_confirmation():
    seen = []

    def probe(trace):
        seen.append(trace)
        return {"status": "fail", "signature": "wanted"}

    result = reduce_trace([{"request_id": str(i)} for i in range(8)], probe, "wanted", max_probes=3)
    assert len(seen) <= 3
    assert result["status"] == "budget_exhausted"
    assert result["probe_count"] == len(seen)


def test_blocked_probe_is_not_a_matching_failure():
    def probe(trace):
        return {"status": "blocked", "signature": "wanted"}

    result = reduce_trace([{"request_id": "a"}], probe, "wanted")
    assert result["status"] == "not_stably_reproduced"


def test_failure_disappearing_at_final_confirmation_is_not_reported_reduced():
    full_calls = 0

    def probe(trace):
        nonlocal full_calls
        if len(trace) == 2:
            full_calls += 1
            if full_calls <= 2:
                return {"status": "fail", "signature": "wanted"}
        return {"status": "pass"}

    result = reduce_trace([{"request_id": "a"}, {"request_id": "b"}], probe, "wanted")
    assert result["status"] == "not_stably_reproduced"
