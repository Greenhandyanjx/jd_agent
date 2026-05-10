"""
ChatMemory — 三层记忆统一接口

整合 jd_agent 现有的记忆组件，提供统一的门面：
1. 短期记忆 → Session.messages（由 session/manager.py 管理）
2. 中期记忆 → history.jsonl（Consolidator 压缩摘要，HistoryJsonlStore 管理）
3. 长期记忆 → MEMORY.md（Dream 整理归档，MemoryStore 管理）

使用方式：
    memory = ChatMemory(workspace=Path("."))
    
    # 构建 LLM 上下文（含长期记忆 + 历史摘要 + 近期消息）
    ctx = memory.build_context(session)
    
    # 触发 consolidation（消息数/时间超限时）
    await memory.maybe_consolidate(session)
    
    # 触发 Dream 整理（可定期调用）
    dream_result = memory.run_dream()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from agent.memory.memory_store import MemoryStore, HistoryJsonlStore, MemoryConsolidator
from agent.memory.dream import Dream
from agent.core.types import LLMResponse


class ChatMemory:
    """多轮对话记忆的封装门面。

    整合：
    - Session 对话历史 → 短期
    - history.jsonl Consolidator 摘要 → 中期
    - MEMORY.md Dream 归档 → 长期
    """

    def __init__(
        self,
        workspace: Path,
        max_history_entries: int = 500,
    ):
        self.workspace = workspace
        self.store = MemoryStore(workspace)
        self.history_jsonl = self.store.history_jsonl  # 直接引用

        # Consolidator 和 Dream 的 LLM 回调（外部注入）
        self._summarizer_fn: Callable[[str], str] | None = None
        self._dream_analyzer_fn: Callable[[str], str] | None = None
        self._dream_dedup_fn: Callable[[str], str] | None = None

        # Consolidator 实例（给外部用，不替换 MemoryConsolidator）
        self.dream = Dream(
            store=self.store,
            history_jsonl=self.history_jsonl,
            summarizer_fn=None,  # 稍后注入
            dedup_fn=None,
        )

    # ─── LLM 回调注入 ─────────────────────────────────

    def set_summarizer(self, fn: Callable[[str], str]) -> None:
        """注入 Consolidator 用的 LLM 摘要函数"""
        self._summarizer_fn = fn

    def set_dream_fn(self, analyze_fn: Callable[[str], str], dedup_fn: Callable[[str], str]) -> None:
        """注入 Dream 用的 LLM 分析+去重函数"""
        self.dream.summarizer_fn = analyze_fn
        self.dream.dedup_fn = dedup_fn

    # ─── 上下文构建 ──────────────────────────────────

    def build_context(
        self,
        session_memory_context: str = "",
        max_memory_chars: int = 8000,
        max_history_entries: int = 5,
    ) -> str:
        """构建完整的记忆上下文（用于注入 system prompt）。

        组装顺序：
        1. 长期记忆：MEMORY.md
        2. 中期记忆：history.jsonl 最近几条摘要
        3. 短期记忆：由外部调用者传入（session.get_history() 已含）

        Args:
            session_memory_context: 可选的 session 特定上下文（如 ContextBuilder.get()）
            max_memory_chars: MEMORY.md 截断字符数
            max_history_entries: 从 history.jsonl 读取的最新条目数

        Returns:
            格式化的上下文文本（可为空）
        """
        parts = []
        memory = self.store.get_memory_context()
        if memory:
            parts.append(memory[:max_memory_chars])

        # 中期记忆
        entries = self.history_jsonl.read_all()
        if entries:
            recent = entries[-max_history_entries:]
            summary_lines = []
            for i, e in enumerate(recent, 1):
                summary = e.get("summary", "")
                if summary:
                    # 截断到 200 字
                    summary_lines.append(f"## 历史阶段 {i}\n{summary[:200]}")
            if summary_lines:
                parts.append("## 对话历史摘要\n" + "\n\n".join(summary_lines))

        if session_memory_context:
            parts.append(session_memory_context)

        return "\n\n---\n\n".join(parts)

    # ─── Consolidation ──────────────────────────────

    async def maybe_consolidate(
        self,
        consolidator: MemoryConsolidator,
        session: Any,  # Session
    ) -> None:
        """检查并执行 consolidation（委托给 MemoryConsolidator）"""
        await consolidator.maybe_consolidate(session)

    # ─── Dream ──────────────────────────────────────

    def run_dream(self) -> dict[str, Any]:
        """执行一轮 Dream 处理"""
        result = self.dream.run()
        return result

    def dream_status(self) -> dict[str, Any]:
        """获取 Dream 状态"""
        return self.dream.status()

    def tidy_memory(self) -> int:
        """清理 MEMORY.md 过时条目"""
        return self.dream.tidy_memory()

    # ─── 状态查询 ──────────────────────────────────

    def status(self) -> dict[str, Any]:
        """获取记忆系统概览"""
        history_entries = len(self.history_jsonl.read_all())
        memory_size = len(self.store.read_long_term())
        dream_info = self.dream.status()

        return {
            "history_jsonl_entries": history_entries,
            "memory_size_chars": memory_size,
            "dream": dream_info,
        }
