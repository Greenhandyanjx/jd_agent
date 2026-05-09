"""
Agent Core: 记忆系统（三层记忆）
参考 nanobot 的 agent/memory.py 设计

记忆系统分为三层：
1. MEMORY.md（长期事实）— 跨会话持久化的用户偏好、关键事实
2. HISTORY.md（历史日志）— grep 可搜索的行为日志
3. Session Messages（短期记忆）— 当前会话上下文（由 session/manager.py 管理）

关键特点：
- MemoryStore 负责文件级别的读写
- MemoryConsolidator 负责在合适的时机（按时间/消息数）自动 consolidation
- Consolidation 通过 LLM tool call 让模型自己决定写什么
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from agent.core.types import LLMResponse, ToolCallRequest

if TYPE_CHECKING:
    from agent.core.llm_provider import LLMProvider
    from agent.session.manager import Session, SessionManager


# ─── MemoryConsolidator 使用的 LLM Tool 定义 ─────
# 这是一个"假的" tool 定义，实际上我们不调用 tool，
# 而是让 LLM 直接返回我们需要的 JSON 格式

_MEMORY_SAVE_PROMPT = """## 记忆系统指令

请根据当前对话内容，更新记忆文件。
请以 JSON 格式返回以下两个字段（不要包含其他内容）：

```json
{
  "history_entry": "[YYYY-MM-DD HH:MM] 一段描述本次对话关键事件/决策的段落...",
  "memory_update": "完整的更新后的 long-term memory（Markdown 格式）..."
}
```

- history_entry: 添加到 HISTORY.md 的一条可 grep 搜索的摘要
- memory_update: 完整的 MEMORY.md 内容（包含所有已有事实 + 新学习到的信息）
"""


class MemoryStore:
    """
    Memory 文件存储层。
    
    管理两个文件：
    - memory/MEMORY.md: 长期事实（跨会话持久）
    - memory/HISTORY.md: 历史日志（可 grep 搜索）
    
    参考 nanobot 的 MemoryStore，但简化了 history.jsonl 的迁移逻辑。
    """

    def __init__(self, workspace: Path):
        self.memory_dir = _ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    # ─── MEMORY.md 读写 ────────────────────────

    def read_long_term(self) -> str:
        """读取长期记忆"""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        """写入长期记忆"""
        self.memory_file.write_text(content, encoding="utf-8")

    def get_memory_context(self) -> str:
        """获取用于 system prompt 的记忆上下文"""
        content = self.read_long_term()
        return f"## Long-term Memory\n{content}" if content else ""

    # ─── HISTORY.md 操作 ───────────────────────

    def append_history(self, entry: str) -> None:
        """追加一条历史日志"""
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class MemoryConsolidator:
    """
    记忆 Consolidation 引擎。
    
    在合适的时机调用 LLM 来总结对话并更新 MEMORY.md / HISTORY.md。
    
    触发时机：
    - 每 N 条消息后（按消息数）
    - 每 T 分钟（按时间间隔）
    
    参考 nanobot 的 MemoryConsolidator，但更简洁。
    """

    def __init__(
        self,
        workspace: Path,
        provider: "LLMProvider",
        model: str,
        memory_store: MemoryStore,
        context_builder: Any,  # ContextBuilder, 用于 build_messages
        sessions: "SessionManager",
        consolidate_every_n: int = 20,  # 每 N 条消息后触发
        consolidate_interval_min: int = 15,  # 最短间隔 15 分钟
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.memory = memory_store
        self.context_builder = context_builder
        self.sessions = sessions
        self._consolidate_every_n = consolidate_every_n
        self._consolidate_interval_min = consolidate_interval_min
        self._last_consolidation_time: float = 0.0  # epoch 时间戳

    async def maybe_consolidate(self, session: "Session") -> None:
        """
        检查是否需要进行记忆 consolidation。
        
        条件：
        1. 自上次 consolidation 以来有新消息
        2. 达到消息数阈值或时间间隔
        """
        new_count = len(session.messages) - session.last_consolidated
        if new_count == 0:
            return

        # 按消息数检查
        hit_count = new_count >= self._consolidate_every_n

        # 按时间检查
        now = datetime.now().timestamp()
        time_since_last = now - self._last_consolidation_time
        hit_time = time_since_last >= self._consolidate_interval_min * 60

        if not hit_count and not hit_time:
            return

        logger.info(f"触发记忆 consolidation "
                     f"(消息数阈值: {new_count}/{self._consolidate_every_n}, "
                     f"时间间隔: {int(time_since_last)}s/{self._consolidate_interval_min * 60}s)")

        await self._consolidate(session)
        self._last_consolidation_time = now

    async def _consolidate(self, session: "Session") -> None:
        """
        执行记忆 consolidation。
        
        流程：
        1. 获取未 consolidated 的对话历史
        2. 用 LLM 分析对话并生成 history_entry + memory_update
        3. 写入 HISTORY.md 和 MEMORY.md
        4. 更新 session 的 last_consolidated 指针
        """
        # 获取未 consolidated 的历史
        unconsolidated = session.messages[session.last_consolidated:]
        if not unconsolidated:
            return

        # 读取当前的 MEMORY.md 以获得上下文
        current_memory = self.memory.read_long_term()

        # 构建 consolidation prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个记忆管理系统。请分析以下对话历史，提取关键信息。"
                    "返回 JSON 格式：\n\n"
                    '{"history_entry": "时间戳 摘要", "memory_update": "完整记忆文件"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## 当前长期记忆\n{current_memory or '（空）'}\n\n"
                    f"## 这段对话历史（共 {len(unconsolidated)} 条消息）\n"
                    + json.dumps([
                        {"role": m["role"], "content": str(m.get("content", ""))[:500]}
                        for m in unconsolidated[-10:]  # 只取最近 10 条，避免 token 爆炸
                    ], ensure_ascii=False, indent=2)
                    + "\n\n请返回 JSON 格式的 history_entry 和 memory_update。"
                ),
            },
        ]

        try:
            response = await self.provider.chat(
                messages=messages,
                model=self.model,
                max_tokens=2048,
                temperature=0.3,
            )

            if response.content:
                result = self._parse_consolidation_result(response.content)
                if result:
                    history_entry, memory_update = result
                    # 写入 HISTORY.md
                    if history_entry:
                        self.memory.append_history(history_entry)
                    # 更新 MEMORY.md
                    if memory_update:
                        self.memory.write_long_term(memory_update)
                    # 更新 session 指针
                    session.last_consolidated = len(session.messages)
                    logger.info("记忆 consolidation 完成")
        except Exception as e:
            logger.error(f"记忆 consolidation 失败: {e}")

    @staticmethod
    def _parse_consolidation_result(content: str) -> tuple[str, str] | None:
        """解析 LLM 返回的 consolidation 结果"""
        # 尝试提取 JSON
        import re
        json_match = re.search(r'\{[^{}]*"history_entry"[^{}]*"memory_update"[^{}]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("history_entry", ""), data.get("memory_update", "")
            except (json.JSONDecodeError, KeyError):
                pass
        return None
