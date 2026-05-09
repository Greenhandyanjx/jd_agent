"""
Agent Orchestrator: Agent 编排器
参考 nanobot 的 bus/queue.py + agent/loop.py 设计

编排器是系统的统一入口：
1. 初始化 MessageBus + AgentLoop + Provider + Tools
2. 提供 chat() / chat_stream() 接口给外部调用
3. 管理 Agent 生命周期

核心改变（参考 nanobot 后）：
- 使用 MessageBus 解耦输入/输出
- AgentLoop 处理所有消息
- Provider 抽象化，支持切换
- 新的三层记忆系统
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

from loguru import logger

from agent.core.types import InboundMessage, OutboundMessage
from agent.core.llm_provider import LLMProvider
from agent.bus.queue import MessageBus
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry


class AgentOrchestrator:
    """
    Agent 编排器。
    
    负责：
    1. 初始化所有组件（总线、循环、Provider、工具）
    2. 提供同步/异步的 chat() 接口
    3. 管理工具注册
    """

    _instance = None

    def __init__(self, provider: LLMProvider | None = None, workspace: str | None = None):
        self.bus = MessageBus()
        self.provider = provider
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self.loop: AgentLoop | None = None
        self._task: asyncio.Task | None = None

    # ─── 初始化 ────────────────────────────────

    async def initialize(self) -> None:
        """
        初始化 Agent 系统。
        
        必须先调用此方法才能使用 Agent。
        """
        self.loop = AgentLoop(
            bus=self.bus,
            provider=self.provider,
            workspace=self.workspace,
        )
        # 启动消息处理循环（后台任务）
        self._task = asyncio.create_task(self.loop.run())
        logger.info("[Orchestrator] Agent 初始化完成")

    # ─── 同步接口 ──────────────────────────────

    def chat(self, user_query: str) -> str:
        """
        同步聊天接口。
        
        适用于 CLI、Streamlit 等简单调用场景。
        内部实际走异步路径。
        """
        return asyncio.run(self.chat_async(user_query))

    async def chat_async(self, user_query: str) -> str:
        """
        异步聊天接口。
        
        构造消息 → 直接处理（不走队列） → 返回结果。
        """
        if self.loop is None:
            await self.initialize()

        response = await self.loop.process_direct(
            content=user_query,
            session_key="cli:direct",
            channel="cli",
            chat_id="direct",
        )
        return response.content if response else ""

    async def chat_stream_async(self, user_query: str) -> AsyncGenerator[str, None]:
        """
        异步流式聊天接口。
        
        每次 yield 一个文本块。
        """
        if self.loop is None:
            await self.initialize()

        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=user_query,
        )

        # 构造流式 progress 回调
        stream_buffer = []

        async def on_stream(delta: str) -> None:
            stream_buffer.append(delta)

        class StreamProxy:
            """用闭包收集流式内容"""
            pass

        # 处理消息
        response = await self.loop._process_message(msg)

        if stream_buffer:
            for chunk in stream_buffer:
                yield chunk
        elif response and response.content:
            yield response.content

    # ─── 工具管理 ──────────────────────────────

    def get_tool_registry(self) -> ToolRegistry:
        """获取工具注册中心（用于注册外部工具）"""
        if self.loop:
            return self.loop.tools
        return None

    def get_all_tools(self) -> list[dict]:
        """获取所有已注册工具的 Schema"""
        if self.loop:
            return self.loop.tools.get_definitions()
        return []

    def get_tool_names(self) -> list[str]:
        """获取所有工具名称"""
        if self.loop:
            return self.loop.tools.tool_names
        return []

    # ─── 会话管理 ──────────────────────────────

    def clear_session(self) -> None:
        """清空当前会话"""
        if self.loop:
            session = self.loop.sessions.get_or_create("cli:direct")
            session.clear()

    def get_session_history(self) -> list[dict]:
        """获取当前会话历史"""
        if self.loop:
            session = self.loop.sessions.get_or_create("cli:direct")
            return session.get_history()
        return []

    # ─── 生命周期 ──────────────────────────────

    async def shutdown(self) -> None:
        """关闭 Agent"""
        if self.loop:
            self.loop.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Orchestrator] Agent 已关闭")
