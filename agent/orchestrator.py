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
from typing import AsyncGenerator

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
    4. 管理技能系统（Skill 自动发现与注册）
    """

    _instance = None

    def __init__(
        self,
        provider: LLMProvider | None = None,
        workspace: str | None = None,
        skills_dir: str | None = None,
    ):
        self.bus = MessageBus()
        self.provider = provider
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self.skills_dir = Path(skills_dir).expanduser().resolve() if skills_dir else None
        self.loop: AgentLoop | None = None
        self._task: asyncio.Task | None = None

    # ─── 初始化 ────────────────────────────────

    async def initialize(self) -> None:
        """
        初始化 Agent 系统。
        
        必须先调用此方法才能使用 Agent。
        初始化后，技能系统也会自动发现并注册。
        """
        self.loop = AgentLoop(
            bus=self.bus,
            provider=self.provider,
            workspace=self.workspace,
            skills_dir=self.skills_dir,
        )
        self._task = asyncio.create_task(self.loop.run())

        # 初始化技能系统：扫描 skills/ 目录，自动发现并注册 Skill
        await self.loop.initialize_skills()

        logger.info("[Orchestrator] Agent 初始化完成")

    # ─── 同步接口 ──────────────────────────────

    def chat(self, user_query: str) -> str:
        """
        同步聊天接口。
        
        适用于 CLI、Streamlit 等简单调用场景。
        内部实际走异步路径。
        """
        return asyncio.run(self.chat_async(user_query))

    async def chat_async(self, user_query: str, session_key: str = "cli:direct") -> str:
        """
        异步聊天接口（非流式）。
        
        构造消息 → 直接处理 → 返回结果。
        """
        if self.loop is None:
            await self.initialize()

        response = await self.loop.process_direct(
            content=user_query,
            session_key=session_key,
            channel="cli",
            chat_id="direct",
        )
        return response.content if response else ""

    async def chat_stream_async(self, user_query: str, session_key: str = "cli:direct") -> AsyncGenerator[str, None]:
        """
        异步流式聊天接口。
        
        真正的流式实现：
        - 直接在 _process_message 内部收集流式 chunk
        - 不需要队列或事件同步（Streamlit 中逐块 yield）
        """
        if self.loop is None:
            await self.initialize()

        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=user_query,
        )

        # 使用简单的 buffer + 同步粒度：收集所有流式块
        # 由于 Streamlit 中 _asyncio.run 会阻塞直到完成，
        # 先全部收集再一次性 yield
        stream_buffer = []

        async def on_stream(delta: str) -> None:
            stream_buffer.append(delta)

        # 临时替换 loop._process_message 中的流式回调
        # _process_message -> _run_agent_loop 接受 on_stream 回调
        # 但我们不修改 _process_message 签名，改用内部模式
        
        # 方法：构造消息后，手动走 _run_agent_loop 并注入回调
        session = self.loop.sessions.get_or_create(msg.session_key)
        history = session.get_history(max_messages=0)
        initial_messages = self.loop.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        final_content, tools_used, all_msgs = await self.loop._run_agent_loop(
            initial_messages,
            on_stream=on_stream,
        )

        if final_content is None:
            final_content = "处理完成，但没有生成回复。"

        # 持久化（_process_message 中的部分逻辑）
        self.loop._save_turn(session, all_msgs, 1 + len(history))
        self.loop.sessions.save(session)

        # yield 流式内容
        if stream_buffer:
            for chunk in stream_buffer:
                yield chunk
        else:
            yield final_content

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

    # ─── 技能管理 ──────────────────────────────

    def get_all_skills(self) -> list[dict]:
        """获取所有已注册技能的字典表示"""
        if self.loop:
            return self.loop.skill_manager.to_dict()
        return []

    def get_skill_count(self) -> int:
        """获取技能数量"""
        if self.loop:
            return self.loop.skill_manager.count
        return 0

    def get_skill_names(self) -> list[str]:
        """获取所有技能名称"""
        if self.loop:
            return [s.name for s in self.loop.skill_manager.get_all()]
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
