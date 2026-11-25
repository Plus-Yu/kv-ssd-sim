#!/usr/bin/env python3
"""Wait for a recorded 5-hour quota reset, then resume one explicit Codex thread.

No credentials, polling API, shell interpolation, or quota bypass is used.
Accepts a Codex rollout JSONL containing session_meta and token_count events.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path


def reset_from_event(event):
    payload = event.get("payload", {})
    if event.get("type") != "event_msg" or payload.get("type") != "token_count":
        return None
    limits = payload.get("rate_limits") or {}
    if limits.get("limit_id") not in (None, "codex"):
        return None
    primary = limits.get("primary") or {}
    if primary.get("window_minutes") != 300 or primary.get("used_percent", 0) < 100:
        return None
    reset = primary.get("resets_at")
    if not isinstance(reset, (int, float)) or reset <= 0:
        return None
    # A longer exhausted window takes precedence over the five-hour window.
    secondary = limits.get("secondary") or {}
    if secondary.get("used_percent", 0) >= 100:
        longer = secondary.get("resets_at")
        if not isinstance(longer, (int, float)) or longer <= 0:
            return None
        reset = max(reset, longer)
    return reset


def read_events(path, offset):
    with path.open("rb") as stream:
        stream.seek(offset)
        events = []
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break  # The active writer has not finished this record yet.
            offset = stream.tell()
            try:
                events.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                continue
    return events, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--codex", default=shutil.which("codex"))
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-wait-hours", type=float, default=168)
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Exit without sending when this task-completion marker exists",
    )
    args = parser.parse_args()
    if not args.codex:
        parser.error("codex is not on PATH; provide --codex /absolute/path/to/codex")
    events, offset = read_events(args.rollout, 0)
    ids = [e.get("payload", {}).get("id") for e in events if e.get("type") == "session_meta"]
    if ids != [args.session_id]:
        parser.error("rollout session_meta must match the explicit --session-id")
    project = args.project.resolve(strict=True)
    state_dir = project / "outputs/quota-resume"
    state_dir.mkdir(parents=True, exist_ok=True)
    # Safe file names: CLI accepts arbitrary thread names, this watcher uses UUIDs only.
    import uuid

    session_id = str(uuid.UUID(args.session_id))
    lock_path = state_dir / f"{session_id}.lock"
    receipt = state_dir / f"{session_id}.json"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists() and not args.dry_run:
            parser.error(f"Already attempted once; inspect {receipt} before rearming")
        prompt = (
            args.prompt_file.read_text()
            if args.prompt_file
            else "继续完成 projects/kv-ssd-sim 的已授权任务：先阅读 docs/项目维护/续接状态.md，"
            "完成可配置 MLA＋KDA、FEMU SSD 闭环和验证报告。检查现有成果后继续，"
            "不要重复已完成实验，不要启动第二个额度恢复监视器。"
        )
        started = time.monotonic()
        target = None
        # Ignore old exhausted windows; only a future reset may arm this watcher.
        for event in events:
            candidate = reset_from_event(event)
            if candidate and candidate > time.time():
                target = max(target or 0, candidate)
        while True:
            if args.stop_file is not None and args.stop_file.exists():
                print("Task completion marker found; no continuation sent", flush=True)
                return 0
            new, offset = read_events(args.rollout, offset)
            for event in new:
                candidate = reset_from_event(event)
                if candidate and candidate > time.time():
                    target = max(target or 0, candidate)
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "session": session_id,
                            "reset_epoch": target,
                            "command": [
                                args.codex,
                                "exec",
                                "resume",
                                "--json",
                                "--skip-git-repo-check",
                                session_id,
                                "-",
                            ],
                            "cwd": str(project),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            if target is not None and time.time() >= target + 60:
                break
            if time.monotonic() - started > args.max_wait_hours * 3600:
                raise TimeoutError("No eligible reset within watcher lifetime; no command sent")
            time.sleep(15)
        record = {
            "session": session_id,
            "reset_epoch": target,
            "started_utc": dt.datetime.now(dt.UTC).isoformat(),
            "status": "attempting",
        }
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        with (state_dir / f"{session_id}.log").open("a") as log:
            result = subprocess.run(
                [args.codex, "exec", "resume", "--json", "--skip-git-repo-check", session_id, "-"],
                input=prompt,
                text=True,
                cwd=project,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update(status="finished", exit_code=result.returncode)
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
