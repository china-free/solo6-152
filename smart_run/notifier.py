"""Webhook notifications for Feishu (Lark) and WeCom (企业微信).

Both platforms accept a JSON body posted to a webhook URL. We build a compact
card / markdown message summarising the run: status, command, exit code,
duration, the plain-language explanation, top suggestions, and a small
snippet of the offending output. All HTTP is done with the standard library.

* Feishu: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/interactive-card
  ``msg_type=interactive`` with a coloured header + lark_md body.
* WeCom:   https://developer.work.weixin.qq.com/document/path/91770
  ``msgtype=markdown``.

Both are best-effort: a failed push is logged but never raises, so a flaky
webhook can't mask the real exit code of smart-run itself.

Plugin adapter
--------------
:class:`NotifierPlugin` wraps :class:`Notifier` as a :class:`Plugin`. It
reacts to ``on_end``, reads ``ctx.analysis`` (produced by
:class:`AnalyzerPlugin`, which must be registered first), and pushes.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Tuple

from .analyzer import AnalysisResult
from .config import Config
from .hooks import HookContext, Plugin
from .runner import RunResult


_SEVERITY = {
    "success": "green",
    "failed": "red",
    "timeout": "orange",
    "crashed": "red",
}


@dataclass
class PushResult:
    channel: str
    ok: bool
    detail: str = ""


class Notifier:
    def __init__(self, config: Config) -> None:
        self.config = config

    def should_notify(self, analysis: AnalysisResult) -> bool:
        if not self.config.has_notifier:
            return False
        if analysis.status == "success":
            return self.config.notify_on_success
        return True

    def notify(self, result: RunResult, analysis: AnalysisResult) -> List[PushResult]:
        if not self.should_notify(analysis):
            return []
        outcomes: List[PushResult] = []
        if self.config.feishu_webhook:
            outcomes.append(self._push_feishu(result, analysis))
        if self.config.wecom_webhook:
            outcomes.append(self._push_wecom(result, analysis))
        return outcomes

    # ------------------------------------------------------------- Feishu ---
    def _push_feishu(self, result: RunResult, analysis: AnalysisResult) -> PushResult:
        color = _SEVERITY.get(analysis.status, "red")
        emoji = "❌" if analysis.status == "failed" else ("⏰" if analysis.status == "timeout" else "✅")

        md_lines: List[str] = []
        md_lines.append(f"**{emoji} {analysis.title}**")
        md_lines.append("")
        md_lines.append(f"**命令**：`{' '.join(result.command)}`")
        md_lines.append(f"**退出码**：{analysis.exit_code}　**耗时**：{_fmt_duration(analysis.duration)}　**状态**：{analysis.status}")
        md_lines.append("")
        md_lines.append(f"**原因**：{analysis.explanation}")
        if analysis.suggestions:
            md_lines.append("")
            md_lines.append("**建议**：")
            for i, s in enumerate(analysis.suggestions, 1):
                md_lines.append(f"{i}. {s}")
        if analysis.llm_text:
            md_lines.append("")
            md_lines.append("**LLM 解释**：")
            md_lines.append(analysis.llm_text)
        if analysis.snippet:
            md_lines.append("")
            md_lines.append("**最后输出**：")
            md_lines.append("```")
            for line in analysis.snippet.splitlines()[-8:]:
                md_lines.append(line)
            md_lines.append("```")
        if self.config.mention:
            md_lines.append("")
            md_lines.append(self.config.mention)

        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": f"smart-run 通知 · {analysis.status.upper()}"},
                    "template": color,
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": "\n".join(md_lines)},
                    }
                ],
            },
        }
        return self._post("feishu", self.config.feishu_webhook, payload)

    # --------------------------------------------------------------- WeCom ---
    def _push_wecom(self, result: RunResult, analysis: AnalysisResult) -> PushResult:
        emoji = "❌" if analysis.status == "failed" else ("⏰" if analysis.status == "timeout" else "✅")
        lines: List[str] = []
        lines.append(f"{emoji} **smart-run 通知**（{analysis.status}）")
        lines.append("")
        lines.append(f"**{analysis.title}**")
        lines.append("")
        lines.append(f"> 命令：`{' '.join(result.command)}`")
        lines.append(f"> 退出码：{analysis.exit_code}  耗时：{_fmt_duration(analysis.duration)}")
        lines.append("")
        lines.append(f"**原因**：{analysis.explanation}")
        if analysis.suggestions:
            lines.append("")
            lines.append("**建议**：")
            for i, s in enumerate(analysis.suggestions, 1):
                lines.append(f"{i}. {s}")
        if analysis.llm_text:
            lines.append("")
            lines.append("**LLM 解释**：")
            lines.append(analysis.llm_text)
        if analysis.snippet:
            lines.append("")
            lines.append("**最后输出**：")
            lines.append("```")
            for line in analysis.snippet.splitlines()[-8:]:
                lines.append(line)
            lines.append("```")
        if self.config.mention:
            lines.append("")
            lines.append(self.config.mention)

        payload = {"msgtype": "markdown", "markdown": {"content": "\n".join(lines)}}
        return self._post("wecom", self.config.wecom_webhook, payload)

    # -------------------------------------------------------------- posting ---
    def _post(self, channel: str, url: str, payload: dict) -> PushResult:
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
                code = parsed.get("code", parsed.get("errcode"))
                if code not in (None, 0, "0"):
                    return PushResult(channel, False, f"webhook returned: {body[:300]}")
            except json.JSONDecodeError:
                pass
            return PushResult(channel, True, body[:200])
        except urllib.error.HTTPError as exc:
            return PushResult(channel, False, f"HTTP {exc.code}: {exc.reason}")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return PushResult(channel, False, str(exc))
        except Exception as exc:  # never let notification mask the real exit
            return PushResult(channel, False, str(exc))


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class NotifierPlugin(Plugin):
    """Lifecycle adapter that pushes webhook notifications on run end.

    Ordering: register this plugin **after** :class:`AnalyzerPlugin` because
    it reads ``ctx.analysis``, which the analyzer produces.

    Best-effort: a failed push logs a warning but never raises or modifies
    the exit code.
    """

    name = "notifier"

    def __init__(self, config: Config) -> None:
        self.notifier = Notifier(config)
        self.config = config

    def on_end(self, ctx: HookContext) -> None:
        result = ctx.result
        analysis = ctx.analysis
        if result is None or analysis is None:
            return
        if not self.notifier.should_notify(analysis):
            return
        outcomes = self.notifier.notify(result, analysis)
        for outcome in outcomes:
            stream = sys.stdout if outcome.ok else sys.stderr
            tag = "ok" if outcome.ok else "FAILED"
            detail = f" ({outcome.detail})" if not outcome.ok and outcome.detail else ""
            try:
                stream.write(f"[smart-run] notify {outcome.channel}: {tag}{detail}\n")
                stream.flush()
            except (OSError, ValueError):
                pass
