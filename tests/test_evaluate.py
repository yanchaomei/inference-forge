from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest

from forge.evaluate import StreamFailure, assess_interleaving, check_numerics, evaluate_trace, stream_request


@contextmanager
def wire_server(messages):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for item in messages:
                self.wfile.write(("data: " + json.dumps(item) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


EVENT = {"request_id": "case", "input_ids": [10, 20, 30], "max_tokens": 2}


def messages():
    return [
        {"request_id": "case", "index": 0, "token_id": 42, "text": "a"},
        {"request_id": "case", "index": 1, "token_id": 43, "text": "b"},
        {"request_id": "case", "finish_reason": "length", "token_ids": [42, 43],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    ]


def test_stream_preserves_exact_ids_and_real_receipt_times():
    with wire_server(messages()) as url:
        result = stream_request(url, EVENT)
    assert result["token_ids"] == [42, 43]
    assert len(result["token_timestamps_ns"]) == 2
    assert result["started_ns"] <= result["token_timestamps_ns"][0] <= result["token_timestamps_ns"][1] <= result["ended_ns"]
    assert result["raw_sse_lines"]


@pytest.mark.parametrize("change,error", [
    ("request_id", "identity"), ("index", "non-sequential"),
    ("usage", "accounting"), ("prompt_tokens", "prompt token"),
])
def test_invalid_stream_cannot_be_accepted_as_a_measurement(change,error):
    rows = messages()
    if change == "request_id":
        rows[0]["request_id"] = "another-request"
    elif change == "index":
        rows[1]["index"] = 3
    elif change == "usage":
        rows[-1]["usage"]["completion_tokens"] = 100
    else:
        rows[-1]["usage"]["prompt_tokens"] = 1
    with wire_server(rows) as url, pytest.raises(RuntimeError, match=error):
        stream_request(url, EVENT)


def test_missing_reference_cases_is_failure_not_vacuous_pass():
    assert check_numerics("http://unused", {"cases": []}, {})["status"] == "fail"


def test_malformed_stream_retains_received_lines_and_partial_tokens():
    rows = messages()
    rows[-1]["usage"]["completion_tokens"] = 999
    with wire_server(rows) as url, pytest.raises(StreamFailure) as caught:
        stream_request(url, EVENT)
    assert caught.value.evidence["request_id"] == "case"
    assert caught.value.evidence["token_ids"] == [42, 43]
    assert "999" in "".join(caught.value.evidence["raw_sse_lines"])


def test_trace_report_keeps_failed_request_evidence():
    rows = messages()[:1]
    golden = {"cases":[{"request_id":"case","token_ids":[42,43]}]}
    with wire_server(rows) as url:
        report = evaluate_trace(url, [EVENT], golden)
    assert report["status"] == "blocked"
    assert report["execution_errors"][0]["token_ids"] == [42]
    assert report["execution_errors"][0]["raw_sse_lines"]


def test_serial_processing_cannot_satisfy_interleaving_gate():
    serial = [{"event":"accepted","request_id":"a"}, {"event":"token","request_id":"a"},
              {"event":"finished","request_id":"a"}, {"event":"accepted","request_id":"b"},
              {"event":"token","request_id":"b"}, {"event":"finished","request_id":"b"}]
    assert assess_interleaving(serial,["a","b"])["status"] == "fail"


def test_real_overlapping_token_rotation_satisfies_interleaving_gate():
    rows = [{"event":"accepted","request_id":"a"}, {"event":"accepted","request_id":"b"},
            {"event":"token","request_id":"a"}, {"event":"token","request_id":"b"},
            {"event":"token","request_id":"a"}]
    result = assess_interleaving(rows,["a","b"])
    assert result["status"] == "pass"
    assert result["max_inflight"] == 2


def test_correct_token_ids_with_corrupt_stream_text_are_rejected():
    rows = messages()
    rows[0]["text"] = "�"
    rows[1]["text"] = "�"
    golden = {"cases":[{"request_id":"case","token_ids":[42,43],"decoded_prefixes":["", "字"]}]}
    with wire_server(rows) as url:
        report = evaluate_trace(url,[EVENT],golden)
    assert report["status"] == "fail"
    assert report["signature"] == "text_mismatch:case:0"
