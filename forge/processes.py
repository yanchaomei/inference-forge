"""Bound an owned process group, including its inherited child processes."""

import os
import signal
import subprocess
import time


def run_bounded(command, timeout_seconds, *, cwd=None, env=None):
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    start = time.monotonic()
    child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    timed_out = False

    def send_group(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    try:
        try:
            child.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            send_group(signal.SIGTERM)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                send_group(signal.SIGKILL)
                child.wait(timeout=2)
    finally:
        # The group was created exclusively for this invocation. No global
        # process search or GPU-occupant termination is used.
        send_group(signal.SIGKILL)
    return {"returncode":child.returncode, "timed_out":timed_out,
            "wall_seconds":time.monotonic()-start, "owned_process_group":child.pid,
            "timeout_seconds":timeout_seconds, "termination_grace_seconds":2}
