"""
Agent Core: ReAct 循环（核心处理引擎）
参考 nanobot 的 agent/loop.py 设计

这是 Agent 的核心：Think → Act → Observe 循环。

流程：
1. 接收用户消息
2. 构建上下文（System Prompt + 历史 + 当前消息）
3. 调用 LLM
4. 如果 LLM 返回工具调用 → 执行工具 → 结果注入 → 回到步骤3
5. 如果 LLM 返回文本 → 作为最终回复返回

关键设计（参考 nanobot）：
- 线程/会话隔离：每个 session 有自己的锁，保证串行处理
- 并发控制：跨 session 可以并发，同 session 串行
- Streaming 支持：LLM 回复可以流式输出给用户
- Progress 通知：工具调用时通知用户"正在做什么"
- 自动重试：调用 LLM 失败时自动重试
- MCP 支持：支持外部 MCP 服务器连接（预留）
"""

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from agent.core.types import InboundMessage, OutboundMessage, LLMResponse
from agent.core.llm_provider import LLMProvider
from agent.bus.queue import MessageBus
from agent.session.manager import Session, SessionManager
from agent.tools.registry import ToolRegistry
from agent.tools.filesystem import ReadFileTool, WriteFileTool
from agent.tools.web import WebFetchTool
from agent.memory.context_builder import ContextBuilder
from agent.memory.memory_store import MemoryConsolidator
from agent.memory.dream import Dream
from agent.memory.chat_memory import ChatMemory
from agent.skills.manager import SkillManager
from agent.skills.loader import SkillLoader

if TYPE_CHECKING:
    pass


class AgentLoop:
    """
    Agent 循环：核心处理引擎。
    
    对应 nanobot 的 AgentLoop：
    1. 从 MessageBus 接收消息
    2. Builder 上下文
    3. 调用 LLM
    4. 执行工具调用
    5. 发送响应
    """

    # 工具结果最大长度（防止 LLM 被撑爆）
    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        skills_dir: str | Path | None = None,
    ):
        """初始化 Agent Loop。

        Args:
            bus: 消息总线（解耦通道和 Agent）
            provider: LLM Provider
            workspace: 工作区路径
            model: 模型名，默认使用 provider 的默认模型
            max_iterations: 最大 ReAct 循环轮数
            context_window_tokens: 上下文窗口大小（用于记忆压缩）
            skills_dir: 技能目录路径（可选，默认 workspace/skills）
        """
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self._start_time = time.time()

        # 核心组件
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()

        # 技能系统（对标 OpenClaw 的 skills/ 机制）
        # 扫描 skills/ 目录，自动发现 Skill → 注册底层 Tool
        skills_path = Path(skills_dir) if skills_dir else (workspace / "skills")
        self.skill_manager = SkillManager(skills_path)

        # 记忆 Consolidation（已有）
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            memory_store=self.context.memory,
            context_builder=self.context,
            sessions=self.sessions,
        )

        # 新增：ChatMemory 门面和 Dream（对标 nanobot 的 Dream）
        self.chat_memory = ChatMemory(workspace=workspace)
        self.dream = self.chat_memory.dream

        # 运行状态
        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

        # 并发门控：NANOBOT_MAX_CONCURRENT_REQUESTS（<=0 不限, 默认 3）
        _max = int(os.environ.get("JDAGENT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )

        # 注册默认工具
        self._register_default_tools()

    async def initialize_skills(self) -> None:
        """
        初始化技能系统：扫描 skills/ 目录，发现并加载所有技能，
        然后将技能下的 Tool 注册到 ToolRegistry。
        """
        try:
            result = await self.skill_manager.discover()
            if result.success_count > 0:
                # 自动注册技能下的 Tool 到 ToolRegistry
                count = self.skill_manager.register_all_tools(self.tools)
                logger.info(
                    f"[Skill] 技能初始化完成: {result.success_count} 个技能, "
                    f"{count} 个工具注册"
                )
            if result.failed_count > 0:
                for f in result.failed:
                    logger.warning(f"[Skill] 技能加载失败: {f}")
        except Exception as e:
            logger.warning(f"[Skill] 技能系统初始化失败: {e}")

    def _register_default_tools(self) -> None:
        """注册 Agent 的默认工具集"""
        self.tools.register(ReadFileTool(workspace=self.workspace))
        self.tools.register(WriteFileTool(workspace=self.workspace))
        self.tools.register(WebFetchTool())

        # 注意：外部工具（如 get_weather、query_order 等）在 tool_definitions.py 中注册
        # 它们会被 orchestrator 加载后追加到这里

    # ─── 工具上下文设置 ─────────────────────────

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """为需要路由信息的工具设置上下文"""
        pass  # 预留：在有 MessageTool 等时需要

    # ─── 核心：ReAct 循环 ─────────────────────

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Any] | None = None,
        on_stream: Callable[[str], Any] | None = None,
        on_stream_end: Callable[..., Any] | None = None,
        skill_context: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """
        运行 Agent ReAct 循环。

        流程：
        while 迭代次数 < max_iterations:
            1. 获取工具定义
            2. 调用 LLM（流式或非流式）
            3. 如果 LLM 返回工具调用:
                a. 执行工具（并行执行所有工具调用）
                b. 将结果注入消息列表
                c. 继续循环
            4. 如果 LLM 返回文本:
                a. 作为最终回复返回

        Args:
            initial_messages: 初始消息列表（system + history + user）
            on_progress: 进度回调（非流式模式下的文字进度）
            on_stream: 流式内容回调（每个文本块）
            on_stream_end: 流式结束回调（resuming=False = 最终回复）
            skill_context: 技能上下文文本，注入到 system message

        Returns:
            (final_content, tools_used, all_messages)
        """
        messages = initial_messages

        # 注入技能上下文到 system message
        if skill_context:
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = (msg["content"] or "") + "\n\n" + skill_context
                    break
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            # 获取工具定义（每次循环都重新获取，因为工具可能动态变更）
            tool_defs = self.tools.get_definitions()

            # === LLM 调用 ===
            if on_stream:
                # 流式模式
                response = await self.provider.chat_stream_with_retry(
                    messages=messages,
                    tools=tool_defs,
                    model=self.model,
                    on_content_delta=on_stream,
                )
            else:
                # 非流式模式
                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tool_defs,
                    model=self.model,
                )

            # 记录 token 用量
            usage = response.usage or {}

            if response.has_tool_calls:
                # ─── Act: 执行工具调用 ───
                if on_stream and on_stream_end:
                    # 通知前端：流式结束，开始执行工具（可以显示 spinner）
                    await on_stream_end(resuming=True)

                if on_progress:
                    # 发送进度通知
                    for tc in response.tool_calls:
                        await on_progress(f"[调用工具] {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:80]}...)")

                # 将 assistant 消息（含 tool_calls）追加到消息列表
                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self._add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                # 记录工具调用
                for tc in response.tool_calls:
                    tools_used.append(tc.name)
                    logger.info(f"[Agent] Tool call: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:200]})")

                # ─── Observe: 执行所有工具并收集结果 ───
                # 注意：LLM 在一次响应中发出的多个工具调用是相互独立的，
                # 所以可以并发执行。return_exceptions=True 确保所有结果都被收集。
                results = await asyncio.gather(*(
                    self.tools.execute(tc.name, tc.arguments)
                    for tc in response.tool_calls
                ), return_exceptions=True)

                # 将工具结果追加到消息列表
                for tool_call, result in zip(response.tool_calls, results):
                    if isinstance(result, BaseException):
                        result = f"错误: {type(result).__name__}: {result}"
                    messages = self._add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

            else:
                # ─── LLM 给出最终回复 ───
                if on_stream and on_stream_end:
                    await on_stream_end(resuming=False)

                # 检查是否出错
                if response.finish_reason == "error":
                    logger.error(f"[Agent] LLM 返回错误: {(response.content or '')[:200]}")
                    final_content = response.content or "抱歉，调用 AI 模型时出错。"
                    break

                # 转换为最终回复
                messages = self._add_assistant_message(
                    messages, response.content,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = response.content
                break

        # 超过最大迭代次数
        if final_content is None and iteration >= self.max_iterations:
            logger.warning(f"[Agent] 达到最大迭代次数 {self.max_iterations}")
            final_content = (
                f"我已经尝试了 {self.max_iterations} 次工具调用仍未完成任务。"
                "您可以尝试将任务分解成更多步骤。"
            )

        return final_content, tools_used, messages

    # ─── 消息处理工具方法 ────────────────────

    @staticmethod
    def _add_assistant_message(
        messages: list[dict],
        content: str | None,
        tool_call_dicts: list[dict] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict]:
        """添加 assistant 消息到消息列表"""
        msg = {"role": "assistant", "content": content}
        if tool_call_dicts:
            msg["tool_calls"] = tool_call_dicts
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
        messages.append(msg)
        return messages

    @staticmethod
    def _add_tool_result(
        messages: list[dict],
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> list[dict]:
        """添加工具执行结果到消息列表"""
        content = str(result)
        # 截断过长的结果
        if len(content) > 16000:
            content = content[:16000] + "\n... (结果过长已截断)"
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        })
        return messages

    # ─── 主循环 ─────────────────────────────

    async def run(self) -> None:
        """
        运行 Agent 主循环。
        
        从 MessageBus 消费入站消息，为每个 session 启动独立任务。
        同 session 串行，跨 session 并发。
        """
        self._running = True
        logger.info("[Agent] Agent 循环启动")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[Agent] 消费消息出错: {e}")
                continue

            # 为每个消息创建独立任务
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(
                lambda t, k=msg.session_key: self._active_tasks.get(k, []) and
                self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """
        分发消息到对应 session。
        
        使用 per-session lock 保证同 session 串行处理。
        使用 concurrency gate 控制全局并发数。
        """
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)

            except asyncio.CancelledError:
                logger.info(f"[Agent] 会话 {msg.session_key} 的任务被取消")
                raise
            except Exception:
                logger.exception(f"[Agent] 处理会话 {msg.session_key} 出错")
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="抱歉，处理您的消息时出现错误。",
                ))

    async def _process_message(
        self,
        msg: InboundMessage,
    ) -> OutboundMessage | None:
        """
        处理单条入站消息。
        
        完整流程：
        1. 获取或创建 session
        2. 加载历史消息
        3. 构建消息列表
        4. 运行 ReAct 循环
        5. 持久化新消息
        6. 触发记忆 consolidation
        7. 返回出站消息
        """
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"[Agent] 处理来自 {msg.channel}:{msg.sender_id} 的消息: {preview}")

        # 获取或创建 session
        session = self.sessions.get_or_create(msg.session_key)

        # ── 技能匹配：根据用户意图匹配最相关的技能 ──
        skill_context = await self._match_skills(msg.content)

        # 构建初始消息
        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        # 运行 ReAct 循环（传入技能上下文）
        final_content, tools_used, all_msgs = await self._run_agent_loop(
            initial_messages,
            skill_context=skill_context,
        )

        if final_content is None:
            final_content = "处理完成，但没有生成回复。"

        # 持久化消息
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        # 触发记忆 consolidation
        await self.memory_consolidator.maybe_consolidate(session)

        # 触发 Dream 归档（积攒至少 3 条未处理的 consolidation 时执行）
        try:
            if self.dream.pending_count() >= 3:
                dream_result = self.chat_memory.run_dream()
                if dream_result.get('added', 0) > 0 or dream_result.get('replaced', 0) > 0:
                    logger.info(f"[Dream] 归档完成: {dream_result}")
        except Exception as exc:
            logger.warning(f"[Dream] 归档失败: {exc}")

        preview_rsp = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"[Agent] 回复 {msg.channel}:{msg.sender_id}: {preview_rsp}")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
        )

    # ─── 技能匹配 ────────────────────────────────

    async def _match_skills(self, query: str) -> str | None:
        """
        根据用户查询匹配合适的技能，返回技能上下文文本。
        
        流程：
          1. 调用 SkillManager.match() 获取 top-k 技能
          2. 调用 compute_context() 生成 LLM 能理解的上下文文本
          3. 将上下文注入到 system prompt 中
        
        如果没有匹配到任何技能，返回 None。
        """
        if self.skill_manager.count == 0:
            return None

        try:
            matches = await self.skill_manager.match(query, top_k=3)
            if not matches:
                return None

            context = self.skill_manager.compute_context(matches)
            if context:
                logger.info(
                    f"[技能匹配] 命中 {len(matches)} 个技能: "
                    + ", ".join(f"{m.skill.name}({m.score:.2f})" for m in matches)
                )
            return context
        except Exception as e:
            logger.warning(f"[技能匹配] 失败: {e}")
            return None

    # ─── 消息持久化 ────────────────────────────

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """
        将本轮消息保存到 session。
        
        skip: 要跳过的消息数（即历史消息的数量）
        只保存本轮新生成的消息。
        """
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")

            # 跳过空的 assistant 消息（可能由 tool-call-only 消息产生）
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue

            # 截断过长的 tool result
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (截断)"

            entry.setdefault("timestamp", time.time())
            session.messages.append(entry)

        session.updated_at = __import__("datetime").datetime.now()

    # ─── 直接处理（CLI/非总线模式）────────────

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> OutboundMessage | None:
        """
        直接处理消息（不走 MessageBus，适用于 CLI 模式）。
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
        )
        return await self._process_message(msg)

    # ─── 生命周期管理 ────────────────────────

    def stop(self) -> None:
        """停止 Agent 循环"""
        self._running = False
        logger.info("[Agent] Agent 循环停止")
