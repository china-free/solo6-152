"""Plugin and lifecycle hook system for smart-run.

The core execution model is simple: :class:`CommandRunner` emits events at
well-defined lifecycle points (start / output / end / error / timeout), and
any registered :class:`Plugin` subclass can react to them.

Plugin ordering
---------------
Plugins are dispatched in **registration order**. Analyzer plugins typically
register early because they produce results (e.g. ``RunResult.analysis``)
that later plugins (notifiers, loggers) consume.

Best effort
-----------
A crashing plugin is caught, logged, and never prevents the command itself
from running or other plugins from executing. That means a misbehaving LLM
plugin or webhook can't mask a training run.

Public API
----------
* :class:`LifecycleEvent` -- enum of all events that a plugin can react to.
* :class:`Plugin` -- base class for all plugins. Subclass and override the
  ``on_*`` methods you care about.
* :class:`PluginManager` -- registry & dispatcher. Pass one instance to
  :class:`CommandRunner` and it will drive the whole chain.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


log = logging.getLogger("smart_run.hooks")


class LifecycleEvent(str, Enum):
    """Events emitted by :class:`CommandRunner` during execution."""

    START = "start"
    OUTPUT = "output"
    END = "end"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class HookContext:
    """Uniform payload delivered to every plugin callback.

    The context is the same object passed through the whole chain, so
    plugins can attach computed artifacts to it for downstream plugins to
    consume. Two conventions:

    * ``result`` -- populated only on ``end``/``error``/``timeout`` events.
    * ``analysis`` -- populated by :class:`AnalyzerPlugin` after the run
      ends. Notifiers read it.
    * ``extra`` -- a free-form dict for anything else.
    """

    event: LifecycleEvent
    command: List[str]
    started_at: float = 0.0
    stream_name: Optional[str] = None
    line: Optional[str] = None
    result: Optional[Any] = None
    analysis: Optional[Any] = None
    extra: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class Plugin:
    """Base class for all smart-run plugins.

    All methods are no-ops by default; override only what you need.

    Each callback receives a :class:`HookContext` with the relevant payload.
    For ``on_end``/``on_error``/``on_timeout``, ``context.result`` is a
    :class:`RunResult` with the tail, exit code, duration etc.

    If a plugin needs to communicate with later plugins in the chain, it can
    write directly onto ``context`` (e.g. setting ``context.analysis``). The
    same context object flows through every registered plugin.

    Ordering contract: analyzer plugins register first, so by the time a
    notifier runs, ``context.analysis`` is guaranteed to exist if the run
    has ended.
    """

    name: str = "plugin"

    # --------------------------------------------------------------- events ---
    def on_start(self, ctx: HookContext) -> None:
        """Command is about to start."""

    def on_output(self, ctx: HookContext) -> None:
        """A line of output (stdout or stderr) was captured.

        ``ctx.stream_name`` is "stdout" or "stderr", ``ctx.line`` is the raw
        line including newline. Fires synchronously from the pump thread, so
        keep it fast.
        """

    def on_end(self, ctx: HookContext) -> None:
        """Command finished naturally (exit code 0 or otherwise)."""

    def on_error(self, ctx: HookContext) -> None:
        """Command failed (non-zero exit code / signal / launch failure)."""

    def on_timeout(self, ctx: HookContext) -> None:
        """Command was killed because it exceeded the timeout."""

    # ---------------------------------------------------------- safe dispatch
    def handle(self, event: LifecycleEvent, ctx: HookContext) -> None:
        """Dispatch one event; catches and logs any exception."""
        handler = {
            LifecycleEvent.START: self.on_start,
            LifecycleEvent.OUTPUT: self.on_output,
            LifecycleEvent.END: self.on_end,
            LifecycleEvent.ERROR: self.on_error,
            LifecycleEvent.TIMEOUT: self.on_timeout,
        }.get(event)
        if handler is None:
            return
        try:
            handler(ctx)
        except Exception as exc:  # pylint: disable=broad-except
            # A plugin crash must never take down the whole run.
            log.warning("plugin %s: %s handler failed: %s", self.name, event, exc, exc_info=True)
            try:
                sys.stderr.write(f"[smart-run] plugin {self.name} warning: {event} handler failed: {exc}\n")
                sys.stderr.flush()
            except (OSError, ValueError):
                pass


class PluginManager:
    """Ordered registry of plugins, plus a dispatcher.

    Usage::

        pm = PluginManager()
        pm.register(AnalyzerPlugin(config))
        pm.register(NotifierPlugin(config))
        runner = CommandRunner(cmd, config, plugin_manager=pm)
        result = runner.run()  # all events dispatched automatically

    Thread-safety: ``dispatch`` is guarded by a lock because :meth:`on_output`
    fires from two pump threads concurrently. All other events fire from the
    main thread after ``proc.wait()`` returns.
    """

    def __init__(self) -> None:
        self._plugins: List[Plugin] = []
        self._lock = threading.Lock()

    # -------------------------------------------------------------- register ---
    def register(self, plugin: Plugin) -> None:
        """Append a plugin to the dispatch chain."""
        if not isinstance(plugin, Plugin):
            raise TypeError(f"expected Plugin, got {type(plugin).__name__}")
        self._plugins.append(plugin)

    def register_all(self, plugins: Sequence[Plugin]) -> None:
        for p in plugins:
            self.register(p)

    # ------------------------------------------------------------- dispatch ---
    def dispatch(self, event: LifecycleEvent, ctx: HookContext) -> None:
        """Send one event to every registered plugin in order.

        Output events are serialized with a lock since they come from two
        threads concurrently.
        """
        if event == LifecycleEvent.OUTPUT:
            with self._lock:
                for p in self._plugins:
                    p.handle(event, ctx)
        else:
            for p in self._plugins:
                p.handle(event, ctx)

    # ----------------------------------------------------------- introspection
    def __iter__(self):
        return iter(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, plugin: Plugin) -> bool:
        return plugin in self._plugins
