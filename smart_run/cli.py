"""Command-line interface for smart-run.

Usage
-----
    smart-run [options] -- <command> [args...]
    smart-run [options] <command> [args...]

Put smart-run's own options first; everything from the first positional token
onwards (or everything after ``--``) is treated as the wrapped command and is
passed through verbatim -- including its own flags like ``--epochs 5``.

Examples
--------
    smart-run python train.py
    smart-run --tail-lines 80 --log-file run.log -- python train.py --epochs 5
    smart-run --feishu-webhook $URL python clean_data.py

Exit code: smart-run mirrors the wrapped command's exit code, so wrapping a
script in a CI pipeline keeps exit-code semantics intact.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .analyzer import Analyzer, format_for_terminal
from .config import build_config
from .notifier import Notifier
from .runner import CommandRunner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smart-run",
        description=(
            "Wrap a long-running command: capture stdout/stderr, explain "
            "crashes in plain language, and push a Webhook notification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    p.add_argument("--version", action="version", version=f"smart-run {__version__}")

    # --- capture / analysis ---
    p.add_argument("--tail-lines", type=int, default=None, metavar="N",
                   help="number of trailing lines to keep for analysis (default 60)")
    p.add_argument("--analyze-on-success", action="store_true", default=None,
                   help="also produce an analysis summary when the command succeeds")
    p.add_argument("--no-llm", dest="use_llm", action="store_false", default=None,
                   help="disable the LLM explanation even if credentials exist")
    p.add_argument("--use-llm", dest="use_llm", action="store_true", default=None,
                   help="enable the LLM explanation (requires base_url + api_key)")

    # --- LLM config ---
    p.add_argument("--llm-base-url", default=None, metavar="URL",
                   help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
    p.add_argument("--llm-api-key", default=None, metavar="KEY",
                   help="API key for the LLM endpoint")
    p.add_argument("--llm-model", default=None, metavar="NAME",
                   help="model name (default gpt-4o-mini)")

    # --- notification ---
    p.add_argument("--feishu-webhook", default=None, metavar="URL",
                   help="Feishu (Lark) bot webhook URL")
    p.add_argument("--wecom-webhook", default=None, metavar="URL",
                   help="WeCom (企业微信) bot webhook URL")
    p.add_argument("--notify-on-success", dest="notify_on_success",
                   action="store_true", default=None,
                   help="notify when the command succeeds (default: on when any webhook is set)")
    p.add_argument("--no-notify-on-success", dest="notify_on_success",
                   action="store_false", default=None,
                   help="do NOT notify when the command succeeds, only on failure")
    p.add_argument("--no-notify", action="store_true", default=False,
                   help="disable all Webhook notifications for this run")
    p.add_argument("--mention", default=None, metavar="TEXT",
                   help="text/mention to append to the notification body")

    # --- runtime ---
    p.add_argument("--log-file", default=None, metavar="PATH",
                   help="append the full transcript to this log file")
    p.add_argument("--no-passthrough", dest="passthrough", action="store_false",
                   default=None, help="do not mirror output to the terminal")
    p.add_argument("--shell", action="store_true", default=None,
                   help="run the command through the shell (expands globs/pipes)")
    p.add_argument("--cwd", default=None, metavar="PATH",
                   help="working directory for the wrapped command")
    p.add_argument("--timeout", type=float, default=None, metavar="SEC",
                   help="kill the command after SEC seconds")

    # --- the wrapped command: everything after the options ---
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="command to run, e.g. `python train.py`")
    return p


def _split_command(argv: Sequence[str]) -> tuple[List[str], List[str]]:
    """Separate smart-run flags from the wrapped command.

    If ``--`` is present, everything after it is the command verbatim and the
    part before is smart-run's flags. Otherwise we let argparse's REMAINDER
    handle it (it grabs the first positional onward).
    """
    argv = list(argv)
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def _cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a config-override dict from only the explicitly-set CLI flags."""
    overrides: Dict[str, Any] = {}

    mapping = {
        "tail_lines": args.tail_lines,
        "analyze_on_success": args.analyze_on_success,
        "use_llm": args.use_llm,
        "feishu_webhook": args.feishu_webhook,
        "wecom_webhook": args.wecom_webhook,
        "notify_on_success": args.notify_on_success,
        "mention": args.mention,
        "log_file": args.log_file,
        "passthrough": args.passthrough,
        "shell": args.shell,
        "cwd": args.cwd,
        "timeout": args.timeout,
    }
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value

    llm: Dict[str, Any] = {}
    if args.llm_base_url is not None:
        llm["base_url"] = args.llm_base_url
    if args.llm_api_key is not None:
        llm["api_key"] = args.llm_api_key
    if args.llm_model is not None:
        llm["model"] = args.llm_model
    if args.use_llm is not None:
        llm["enabled"] = args.use_llm
    if llm:
        overrides["llm"] = llm

    if args.no_notify:
        overrides["feishu_webhook"] = ""
        overrides["wecom_webhook"] = ""

    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    smart_argv, forced_command = _split_command(argv)
    parser = _build_parser()
    args = parser.parse_args(smart_argv)

    command: List[str] = forced_command if forced_command else list(args.command or [])

    while command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.print_help(sys.stderr)
        sys.stderr.write("\nerror: no command given.\n")
        return 2

    config = build_config(_cli_overrides(args))

    runner = CommandRunner(command, config)
    result = runner.run()

    analyzer = Analyzer(config)
    analysis = analyzer.analyze(result)

    if config.passthrough is False:
        sys.stdout.write(format_for_terminal(analysis))
        sys.stdout.flush()
    else:
        sys.stderr.write(format_for_terminal(analysis))
        sys.stderr.flush()

    notifier = Notifier(config)
    if notifier.should_notify(analysis):
        outcomes = notifier.notify(result, analysis)
        for outcome in outcomes:
            stream = sys.stdout if outcome.ok else sys.stderr
            tag = "ok" if outcome.ok else "FAILED"
            stream.write(f"[smart-run] notify {outcome.channel}: {tag}"
                         + (f" ({outcome.detail})" if not outcome.ok and outcome.detail else "")
                         + "\n")
            stream.flush()

    return result.exit_code if result.exit_code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
