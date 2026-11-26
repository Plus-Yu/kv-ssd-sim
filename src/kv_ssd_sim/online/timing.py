"""Wait for a compute deadline, including placeholder copying in its budget."""

import asyncio
import time


async def wait_until(deadline, *, clock=time.perf_counter, sleep=asyncio.sleep):
    # Recheck the monotonic clock in case an event loop timer wakes early.
    while (remaining := deadline - clock()) > 0:
        await sleep(remaining)
