"""smart-run: a smart command-line wrapper.

Wraps a long-running command, captures its stdout/stderr into a rolling
buffer, and when the command finishes (or crashes) it explains the failure
in plain language and pushes a Webhook notification (Feishu / WeCom).
"""

__version__ = "0.1.0"
