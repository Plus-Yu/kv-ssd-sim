"""Host side of an opt-in request/response sampler for a disposable FEMU VM."""

import json
import socket
import subprocess
import time
from contextlib import ExitStack


def receive(stream):
    while True:
        line = stream.readline()
        if not line:
            raise ConnectionError("FEMU closed QMP")
        response = json.loads(line)
        if "event" not in response:
            return response


def execute(stream, command, **arguments):
    stream.write((json.dumps({"execute": command, "arguments": arguments}) + "\n").encode())
    response = receive(stream)
    if "error" in response:
        raise RuntimeError(f"{command}: {response['error']}")
    return response["return"]


def run(command, output, log, env, timeout, sockpath):
    deadline = time.monotonic() + timeout
    with (
        subprocess.Popen(
            command,
            cwd=output,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as vm,
        ExitStack() as stack,
    ):
        try:
            seen = set()
            stream = None
            while vm.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("FEMU trial timed out")
                for request in sorted(output.glob("nand-*.request")):
                    if request.name in seen:
                        continue
                    if stream is None:
                        sock = stack.enter_context(socket.socket(socket.AF_UNIX))
                        sock.settimeout(20)
                        sock.connect(str(sockpath))
                        stream = stack.enter_context(sock.makefile("rwb", buffering=0))
                        if "QMP" not in receive(stream):
                            raise ValueError("Bad FEMU QMP greeting")
                        execute(stream, "qmp_capabilities")
                    data = execute(stream, "query-femu", nsid=1, kind="lines", limit=4096)
                    temp = request.with_suffix(".qmp.tmp")
                    temp.write_text(json.dumps(data) + "\n")
                    temp.replace(request.with_suffix(".qmp.json"))
                    seen.add(request.name)
                time.sleep(0.05)
            return vm.returncode
        finally:
            if vm.poll() is None:
                vm.kill()
