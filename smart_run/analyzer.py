"""Crash analysis: turn a captured tail into a plain-language explanation.

The analyzer has two layers:

1. A **local regex library** (see :mod:`smart_run.patterns`) that needs no
   network and no credentials. It covers the common ML / data-pipeline failure
   modes (CUDA OOM, segfault, OOM-killer, NCCL, disk full, missing files /
   modules, shape mismatch, ...). This always runs.

2. An **optional LLM pass**. If an OpenAI-compatible endpoint is configured
   (``base_url`` + ``api_key``), the analyzer sends the tail + exit code and
   asks the model for a short, developer-friendly explanation. The LLM result
   augments -- never replaces -- the local diagnosis, so a network failure or a
   missing key never breaks the tool.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Config
from .patterns import PATTERNS
from .runner import RunResult


@dataclass
class AnalysisResult:
    status: str  # "success" | "failed" | "timeout" | "crashed"
    title: str
    explanation: str
    suggestions: List[str] = field(default_factory=list)
    matched_pattern_ids: List[str] = field(default_factory=list)
    llm_text: str = ""
    snippet: str = ""
    exit_code: Optional[int] = None
    duration: float = 0.0

    @property
    def used_llm(self) -> bool:
        return bool(self.llm_text)


class Analyzer:
    def __init__(self, config: Config) -> None:
        self.config = config

    # ----------------------------------------------------------------- main
    def analyze(self, result: RunResult) -> AnalysisResult:
        tail_text = "".join(result.tail)
        snippet = self._extract_snippet(tail_text)

        if result.timed_out or result.killed_by_timeout:
            return AnalysisResult(
                status="timeout",
                title="任务超时被强制终止",
                explanation=(
                    "smart-run 在设定的超时时间内没有等到任务结束，已主动终止子进程。"
                    "可能是任务卡死、等待外部资源，或单纯跑得太久需要更长的超时。"
                ),
                suggestions=[
                    "通过 --timeout 调大超时阈值，或去掉超时让它跑到自然结束。",
                    "检查是否卡在网络请求、分布式 barrier、或死锁。",
                    "查看日志确认最后停在哪一步。",
                ],
                snippet=snippet,
                exit_code=result.exit_code,
                duration=result.duration,
            )

        if result.success:
            if not self.config.analyze_on_success:
                return AnalysisResult(
                    status="success",
                    title="任务正常结束",
                    explanation="命令以退出码 0 正常结束，未检测到错误。",
                    snippet=snippet,
                    exit_code=result.exit_code,
                    duration=result.duration,
                )

        # --- local regex layer ------------------------------------------------
        matched: List[Dict[str, Any]] = []
        for rule in PATTERNS:
            try:
                if re.search(rule["regex"], tail_text, re.IGNORECASE | re.MULTILINE):
                    matched.append(rule)
            except re.error:
                continue

        primary = matched[0] if matched else None

        if primary is not None:
            title = primary["title"]
            explanation = primary["explanation"]
            suggestions = list(primary["suggestions"])
            matched_ids = [r["id"] for r in matched]
        else:
            exit_code = result.exit_code
            title = self._fallback_title(exit_code)
            explanation = self._fallback_explanation(exit_code, tail_text)
            suggestions = self._fallback_suggestions(exit_code)
            matched_ids = []

        llm_text = ""
        if self._llm_available():
            llm_text = self._call_llm(result, tail_text, title, explanation)

        return AnalysisResult(
            status="failed" if result.failed else "success",
            title=title,
            explanation=explanation,
            suggestions=suggestions,
            matched_pattern_ids=matched_ids,
            llm_text=llm_text,
            snippet=snippet,
            exit_code=result.exit_code,
            duration=result.duration,
        )

    # --------------------------------------------------------------- helpers
    def _extract_snippet(self, tail_text: str, max_lines: int = 12) -> str:
        lines = tail_text.splitlines()
        snippet_lines = lines[-max_lines:] if lines else []
        snippet = "\n".join(snippet_lines).strip()
        cap = 1500
        if len(snippet) > cap:
            snippet = "..." + snippet[-cap:]
        return snippet

    def _fallback_title(self, exit_code: Optional[int]) -> str:
        if exit_code is None:
            return "任务异常退出"
        if exit_code == 137:
            return "任务被杀死（退出码 137，疑似 OOM Killer）"
        if exit_code == 139:
            return "任务段错误（退出码 139）"
        return f"任务以非零退出码结束（{exit_code}）"

    def _fallback_explanation(self, exit_code: Optional[int], tail_text: str) -> str:
        if exit_code == 137:
            return (
                "进程退出码 137 = 128 + 9(SIGKILL)，通常是被系统 OOM Killer 杀掉，"
                "也可能是手动 kill。没有 Python 异常堆栈往往正是因为是被外部信号杀死的。"
            )
        if exit_code == 139:
            return (
                "进程退出码 139 = 128 + 11(SIGSEGV)，发生了段错误，"
                "通常是 C/C++/CUDA 扩展崩溃。"
            )
        if not tail_text.strip():
            return "没有捕获到任何输出，进程就结束了，可能启动即失败或被外部 kill。"
        return (
            "本地规则库没有匹配到已知错误模式。请查看下方最后几行输出，"
            "通常最后一行 `XXXError: ...` 就是根因。"
        )

    def _fallback_suggestions(self, exit_code: Optional[int]) -> List[str]:
        if exit_code == 137:
            return ["检查内存占用，降低 num_workers / 分块读数据。", "用 free -h 观察 RSS 峰值。"]
        if exit_code == 139:
            return ["打开 faulthandler：python -X faultholder train.py。", "核对 CUDA / 驱动 / 算子版本。"]
        return ["查看日志最后几行定位根因。", "把关键报错行贴出来进一步分析。"]

    # ------------------------------------------------------------------- LLM
    def _llm_available(self) -> bool:
        llm = self.config.llm
        want = self.config.use_llm or llm.enabled
        return want and bool(llm.base_url) and bool(llm.api_key)

    def _call_llm(
        self,
        result: RunResult,
        tail_text: str,
        local_title: str,
        local_explanation: str,
    ) -> str:
        llm = self.config.llm
        url = llm.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"

        tail_for_prompt = tail_text[-3000:]
        system = (
            "你是一名资深 AI 基础设施工程师，擅长解释训练/数据处理脚本的崩溃原因。"
            "请用通俗、面向开发者的中文回答：1) 一句话点出最可能的根因；"
            "2) 给出 2-4 条可立刻执行的排查/修复建议。不要复述整段堆栈，不要编造没出现的信息。"
        )
        user = (
            f"命令: {' '.join(result.command)}\n"
            f"退出码: {result.exit_code}\n"
            f"耗时: {result.duration:.1f}s\n"
            f"本地规则初步判断: {local_title} —— {local_explanation}\n\n"
            f"以下是程序最后输出的日志（可能含报错堆栈）：\n```\n{tail_for_prompt}\n```"
        )

        payload = {
            "model": llm.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {llm.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=llm.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            choices = parsed.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                return content.strip()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, TimeoutError):
            return ""
        except Exception:
            return ""
        return ""


def format_for_terminal(analysis: AnalysisResult) -> str:
    """Pretty-print an analysis for the terminal (ANSI-free, portable)."""
    parts = []
    parts.append(f"\n===== smart-run 分析 =====")
    parts.append(f"状态: {analysis.status} | 退出码: {analysis.exit_code} | 耗时: {analysis.duration:.1f}s")
    parts.append(f"标题: {analysis.title}")
    parts.append(f"原因: {analysis.explanation}")
    if analysis.suggestions:
        parts.append("建议:")
        for i, s in enumerate(analysis.suggestions, 1):
            parts.append(f"  {i}. {s}")
    if analysis.matched_pattern_ids:
        parts.append(f"匹配规则: {', '.join(analysis.matched_pattern_ids)}")
    if analysis.llm_text:
        parts.append("LLM 解释:")
        parts.append(analysis.llm_text)
    if analysis.snippet:
        parts.append("最后输出片段:")
        for line in analysis.snippet.splitlines():
            parts.append(f"  | {line}")
    parts.append("=========================\n")
    return "\n".join(parts)
