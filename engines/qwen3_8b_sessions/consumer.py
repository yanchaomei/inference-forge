"""Budgeted multi-session consumer for the frozen Qwen3 engine.

An optional retained-snapshot store avoids retransmitting immutable KV prefixes;
plain put/get/delete stores remain fully supported as a fallback.
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
    lease: object = None
    retained_length: int = 0


def payload_bytes(state):
    return sum(t.numel()*t.element_size() for pair in state.kv for t in pair)+state.logits.numel()*state.logits.element_size()


class SessionRuntime:
    def __init__(self, engine, store, gpu_budget_bytes):
        self.engine, self.store = engine, store
        self.gpu_budget_bytes = gpu_budget_bytes
        self.sessions = {}
        self.resident = OrderedDict()
        self._retained_protocol = all(
            callable(getattr(store, name, None))
            for name in ("get_retained", "put_retained", "extend_retained")
        )
        self.counters = {"turns": 0, "resident_hits": 0, "restores": 0, "spills": 0,
                         "restore_payload_bytes": 0, "spill_payload_bytes": 0,
                         "host_full_snapshots": 0, "host_suffix_appends": 0,
                         "host_sync_payload_bytes": 0}

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
            if self._retained_protocol:
                identity = self._identity(session)
                if session.stored:
                    old_length = session.retained_length
                    new_length = int(state.kv[0][0].shape[2])
                    suffix_bytes = sum(
                        k[:, :, old_length:new_length].numel() * k.element_size()
                        + v[:, :, old_length:new_length].numel() * v.element_size()
                        for k, v in state.kv
                    ) + state.logits.numel() * state.logits.element_size()
                    session.lease = self.store.extend_retained(
                        key, state.kv, state.logits, identity, session.lease
                    )
                    self.counters["host_suffix_appends"] += 1
                    self.counters["host_sync_payload_bytes"] += suffix_bytes
                    session.retained_length = new_length
                else:
                    session.lease = self.store.put_retained(
                        key, state.kv, state.logits, identity
                    )
                    session.stored = True
                    self.counters["host_full_snapshots"] += 1
                    self.counters["host_sync_payload_bytes"] += size
                    session.retained_length = int(state.kv[0][0].shape[2])
            else:
                self.store.put(key, state.kv, state.logits, self._identity(session))
                session.stored = True
                self.counters["spill_payload_bytes"] += size
            state.kv = state.logits = None
            session.state = None
            self.counters["spills"] += 1

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
            if self._retained_protocol:
                past, previous_logits, session.lease = self.store.get_retained(
                    session_id, self._identity(session), self.engine.device
                )
                session.retained_length = int(past[0][0].shape[2])
            else:
                past, previous_logits = self.store.get(
                    session_id, self._identity(session), self.engine.device
                )
            self.counters["restore_payload_bytes"] += sum(t.numel()*t.element_size() for pair in past for t in pair)+previous_logits.numel()*previous_logits.element_size()
            self.counters["restores"] += 1
            if session.lease is None:
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
