"""Subprocess orchestration for smart-run.

The :class:`CommandRunner` spawns the wrapped command, pumps its stdout and
stderr through two background threads, keeps a rolling *tail* buffer of the
last N lines (used later for crash analysis), optionally mirrors the output
back to the terminal so the developer can still watch progress, and
optionally persists the full transcript to a log file.

Design notes
------------
* Two pump threads read line-by-line from each pipe. Lines are appended to a
  shared :class:`collections.deque` (bounded, thread-safe enough under the
  GIL, but we still guard it with a lock for cleanliness).
* We do **not** use ``shell=True`` by default -- the command is split into a
  list, which is both safer and avoids surprising glob expansion. A
  ``--shell`` escape hatch exists for people who really need it.
* On timeout we escalate ``terminate`` -> ``kill`` and flag the result so the
  analyzer can explain it as a timeout rather than a crash.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence

from .config import Config
from .hooks import HookContext, LifecycleEvent, PluginManager


@dataclass
class RunResult:
    """Everything the analyzer/notifier need to know about a finished run."""

    command: List[str]
    exit_code: Optional[int]
    success: bool
    duration: float
    tail: List[str] = field(default_factory=list)
    killed_by_timeout: bool = False
    timed_out: bool = False
    started_at: float = 0.0
    ended_at: float = 0.0
    signal: Optional[int] = None

    @property
    def failed(self) -> bool:
        return not self.success


class CommandRunner:
    def __init__(self, command: Sequence[str], config: Config, plugin_manager: Optional[PluginManager] = None) -> None:
        self.command: List[str] = list(command)
        self.config = config
        self.plugin_manager = plugin_manager or PluginManager()
        self._buffer: Deque[str] = deque(maxlen=max(1, config.tail_lines))
        self._lock = threading.Lock()
        self._log_file = None
        self._timed_out = False
        self._killed_by_timeout = False
        self._ctx: Optional[HookContext] = None

    # ------------------------------------------------------------------ run ---
    def run(self) -> RunResult:
        cfg = self.config
        if not self.command:
            raise ValueError("no command given to smart-run")

        started = time.time()
        self._ctx = HookContext(
            event=LifecycleEvent.START,
            command=list(self.command),
            started_at=started,
        )
        self.plugin_manager.dispatch(LifecycleEvent.START, self._ctx)

        self._open_log()
        self._log_header()

        # --- early failure: can't launch -------------------------------------
        result: Optional[RunResult] = None
        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                cwd=cfg.cwd or None,
            )
            if cfg.shell:
                popen_kwargs["shell"] = True
                command_arg = " ".join(self.command)
            else:
                command_arg = list(self.command)
                popen_kwargs["env"] = os.environ.copy()

            proc = subprocess.Popen(command_arg, **popen_kwargs)
        except FileNotFoundError as exc:
            result = RunResult(
                command=self.command,
                exit_code=127,
                success=False,
                duration=0.0,
                tail=[f"smart-run: command not found: {exc}"],
                started_at=started,
                ended_at=time.time(),
            )
        except OSError as exc:
            result = RunResult(
                command=self.command,
                exit_code=126,
                success=False,
                duration=0.0,
                tail=[f"smart-run: failed to launch: {exc}"],
                started_at=started,
                ended_at=time.time(),
            )

        if result is not None:
            self._close_log()
            return self._finalize(result)

        # --- child is running -------------------------------------------------
        threads = [
            threading.Thread(
                target=self._pump,
                args=(proc.stdout, sys.stdout, "stdout"),
                name="smart-run-stdout",
            ),
            threading.Thread(
                target=self._pump,
                args=(proc.stderr, sys.stderr, "stderr"),
                name="smart-run-stderr",
            ),
        ]
        for t in threads:
            t.start()

        timer: Optional[threading.Timer] = None
        if cfg.timeout and cfg.timeout > 0:
            timer = threading.Timer(cfg.timeout, self._timeout_kill, args=(proc,))
            timer.daemon = True
            timer.start()

        exit_code = proc.wait()
        self._killed_by_timeout = self._timed_out
        if timer is not None:
            timer.cancel()

        for t in threads:
            t.join(timeout=10)

        ended = time.time()
        with self._lock:
            tail = list(self._buffer)

        success = exit_code == 0 and not self._killed_by_timeout
        result = RunResult(
            command=self.command,
            exit_code=exit_code,
            success=success,
            duration=ended - started,
            tail=tail,
            killed_by_timeout=self._killed_by_timeout,
            timed_out=self._killed_by_timeout,
            started_at=started,
            ended_at=ended,
            signal=proc.returncode if proc.returncode and proc.returncode < 0 else None,
        )
        self._log_footer(exit_code, ended - started)
        self._close_log()
        return self._finalize(result)

    # --------------------------------------------------------------- finalize --
    def _finalize(self, result: RunResult) -> RunResult:
        """Patch result onto the shared context and dispatch lifecycle events."""
        assert self._ctx is not None
        self._ctx.result = result

        # Timeout path: already dispatched TIMEOUT from the timer thread.
        if result.timed_out:
            self._ctx.event = LifecycleEvent.END
            self.plugin_manager.dispatch(LifecycleEvent.END, self._ctx)
            return result

        # Always emit END (process has exited, one way or another).
        self._ctx.event = LifecycleEvent.END
        self.plugin_manager.dispatch(LifecycleEvent.END, self._ctx)

        # Additionally emit ERROR if the run failed.
        if not result.success:
            self._ctx.event = LifecycleEvent.ERROR
            self.plugin_manager.dispatch(LifecycleEvent.ERROR, self._ctx)

        return result

    # --------------------------------------------------------------- pumping --
    def _pump(self, stream, sink, stream_name: str) -> None:
        if stream is None:
            return
        prefix = ""  # keep terminal output verbatim, no prefix noise
        for raw in iter(stream.readline, ""):
            if raw == "":
                break
            line = raw if raw.endswith("\n") else raw + "\n"
            with self._lock:
                self._buffer.append(line)
            self._write_log(stream_name, line)
            if self.config.passthrough:
                try:
                    sink.write(prefix + line)
                    sink.flush()
                except (OSError, ValueError):
                    pass
            # Fire OUTPUT event -- PluginManager serializes this internally
            if self._ctx is not None:
                self._ctx.event = LifecycleEvent.OUTPUT
                self._ctx.stream_name = stream_name
                self._ctx.line = line
                self.plugin_manager.dispatch(LifecycleEvent.OUTPUT, self._ctx)
        try:
            stream.close()
        except OSError:
            pass

    # --------------------------------------------------------------- timeout --
    def _timeout_kill(self, proc: subprocess.Popen) -> None:
        self._timed_out = True
        self._log("stderr", "smart-run: timeout reached, terminating child\n")
        if self._ctx is not None:
            self._ctx.event = LifecycleEvent.TIMEOUT
            self.plugin_manager.dispatch(LifecycleEvent.TIMEOUT, self._ctx)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except OSError:
            pass

    # ------------------------------------------------------------------ log ---
    def _open_log(self) -> None:
        path = self.config.log_file
        if not path:
            return
        try:
            self._log_file = open(path, "a", encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"smart-run: cannot open log file {path}: {exc}\n")
            self._log_file = None

    def _write_log(self, stream_name: str, line: str) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.write(line)
            self._log_file.flush()
        except OSError:
            pass

    def _log(self, stream_name: str, text: str) -> None:
        self._write_log(stream_name, text)

    def _log_header(self) -> None:
        self._log("meta", f"\n==== smart-run start ====\n")
        self._log("meta", f"command: {' '.join(self.command)}\n")
        self._log("meta", f"cwd: {self.config.cwd or os.getcwd()}\n")
        self._log("meta", f"pid: {os.getpid()}\n")
        self._log("meta", f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log("meta", "---- output ----\n")

    def _log_footer(self, exit_code: Optional[int], duration: float) -> None:
        self._log("meta", "---- end output ----\n")
        self._log("meta", f"exit_code: {exit_code}\n")
        self._log("meta", f"duration: {duration:.2f}s\n")
        self._log("meta", f"ended: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log("meta", "==== smart-run end ====\n\n")

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
