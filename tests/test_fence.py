import concurrent.futures
import threading

import pytest

from kv_ssd_sim.online.native import HostFence


def test_save_cannot_pass_unreleased_compute_fence():
    fence = HostFence()
    started = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

        def save():
            started.set()
            fence.synchronize()
            return "saved"

        future = executor.submit(save)
        assert started.wait(1)
        assert not future.done()
        fence.release()
        assert future.result(timeout=1) == "saved"


def test_compute_failure_is_propagated_to_saver():
    fence = HostFence()
    fence.fail(ValueError("invalid state"))
    with pytest.raises(RuntimeError, match="compute failed"):
        fence.synchronize()
