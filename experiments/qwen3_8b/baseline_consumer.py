"""Fixed generic session consumer used as the joint-construction seed/control.

This hand-written baseline uses only put/get/delete. Model execution stays in
the previously generated Qwen3 engine. GPU budget covers cached session payload
at completed-turn boundaries, not transient attention working memory.
"""
from collections import OrderedDict
from dataclasses import dataclass

from engines.qwen3_8b_kv.engine import RequestState


@dataclass
class Session:
    cached_ids: list
    pending_ids: list
    state: object = None
    stored: bool = False


def payload_bytes(state):
    return sum(t.numel()*t.element_size() for pair in state.kv for t in pair)+state.logits.numel()*state.logits.element_size()


class SessionRuntime:
    def __init__(self, engine, store, gpu_budget_bytes):
        self.engine, self.store = engine, store
        self.gpu_budget_bytes = gpu_budget_bytes
        self.sessions = {}
        self.resident = OrderedDict()
        self.counters = {"turns": 0, "resident_hits": 0, "restores": 0, "spills": 0,
                         "restore_payload_bytes": 0, "spill_payload_bytes": 0}

    def _identity(self, session):
        return {"model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "config_sha256": self.engine.config_sha256, "dtype": "bfloat16",
                "adapter": None, "position_start": 0, "prefix": list(session.cached_ids)}

    def _resident_bytes(self):
        return sum(payload_bytes(self.sessions[key].state) for key in self.resident)

    def _spill_to_budget(self):
        while self.resident and self._resident_bytes() > self.gpu_budget_bytes:
            key, _ = self.resident.popitem(last=False)
            session = self.sessions[key]
            state = session.state
            size = payload_bytes(state)
            self.store.put(key, state.kv, state.logits, self._identity(session))
            session.stored = True
            state.kv = state.logits = None
            session.state = None
            self.counters["spills"] += 1
            self.counters["spill_payload_bytes"] += size

    def turn(self, session_id, input_ids, max_new_tokens, observer=None):
        if not input_ids or max_new_tokens < 1:
            raise ValueError("nonempty input and positive output length required")
        session = self.sessions.get(session_id)
        if session is None:
            session = Session([], [])
            self.sessions[session_id] = session
        prefix = session.cached_ids+session.pending_ids+list(input_ids)
        if len(prefix)+max_new_tokens > getattr(self.engine, "max_context", 4096):
            raise ValueError("session context exceeds model limit")
        if session.state is not None:
            past = session.state.kv
            self.counters["resident_hits"] += 1
        elif session.stored:
            past, previous_logits = self.store.get(session_id, self._identity(session), self.engine.device)
            self.counters["restore_payload_bytes"] += sum(t.numel()*t.element_size() for pair in past for t in pair)+previous_logits.numel()*previous_logits.element_size()
            self.counters["restores"] += 1
            self.store.delete(session_id)
            session.stored = False
            del previous_logits
        else:
            past = None
        logits, kv = self.engine.forward(session.pending_ids+list(input_ids), past)
        state = RequestState(session_id, prefix, max_new_tokens, True, False, kv=kv, logits=logits)
        # Release old local references before enforcing the completed-turn budget.
        del past, kv, logits
        session.state = state
        for index in range(max_new_tokens):
            if observer is not None:
                observer(session_id, index, state.logits)
            self.engine.next_token(state)
        result = {"token_ids": list(state.generated), "token_logprobs": list(state.generated_logprobs)}
        session.cached_ids = prefix+state.generated[:-1]
        session.pending_ids = state.generated[-1:]
        self.resident[session_id] = None
        self.resident.move_to_end(session_id)
        self._spill_to_budget()
        self.counters["turns"] += 1
        return result

    def close(self, session_id):
        session = self.sessions.pop(session_id)
        if session.stored:
            self.store.delete(session_id)
        self.resident.pop(session_id, None)
        if session.state is not None:
            session.state.kv = session.state.logits = None

    def stats(self):
        return {**self.counters, "resident_payload_bytes": self._resident_bytes(),
                "host_payload_bytes": getattr(self.store, "used_bytes", None), "sessions": len(self.sessions)}
