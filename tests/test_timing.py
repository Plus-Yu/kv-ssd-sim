import asyncio

from kv_ssd_sim.online.timing import wait_until


def test_early_wakeup_is_rechecked_and_copy_time_consumes_budget():
    now = [0.004]  # 4 ms of a 10 ms step already used by copying.
    waits = []

    async def sleep(duration):
        waits.append(duration)
        now[0] += duration / 2 if len(waits) == 1 else duration

    asyncio.run(wait_until(0.010, clock=lambda: now[0], sleep=sleep))
    assert abs(waits[0] - 0.006) < 1e-9
    assert len(waits) == 2
    assert now[0] >= 0.010


def test_overrun_does_not_add_another_full_wait():
    async def forbidden_sleep(_):
        raise AssertionError("deadline has already passed")

    asyncio.run(wait_until(0.010, clock=lambda: 0.012, sleep=forbidden_sleep))
