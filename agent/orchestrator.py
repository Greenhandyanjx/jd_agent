"""
Agent Orchestrator: Agent 编排器
作为系统的统一入口，管理 ReAct Loop + Memory + Task Planner + Tool System
"""
from typing import Optional, Generator
from utils.logger import logger
from agent.core.react_loop import ReactLoop
from agent.core.tool_registry import get_all_tools
from agent.memory.short_term import ShortTermMemory
from agent.memory.working_memory import WorkingMemory
from agent.memory.long_term import LongTermMemory
from agent.planner.task_planner import TaskPlanner
from agent.planner.task_graph import TaskExecutor


class AgentOrchestrator:
    """
    Agent 编排器
    统一调度 Agent 各模块
    """

    def __init__(self):
        self.react_loop = ReactLoop()
        self.short_term_memory = self.react_loop.short_term_memory
        self.working_memory = self.react_loop.working_memory
        self.long_term_memory = LongTermMemory()
        self.task_planner = TaskPlanner(self.working_memory)
        self.task_executor = TaskExecutor(self.task_planner, self.react_loop)

    def chat(self, user_query: str) -> str:
        """
        处理用户消息
        自动判断是简单问答还是复杂任务
        """
        logger.info(f"[AgentOrchestrator] 收到用户消息: {user_query[:50]}...")

        # 检查是否有相关长期记忆
        memories = self.long_term_memory.recall(user_query)
        if memories:
            context = "\n".join([m["content"] for m in memories])
            enriched_query = f"{user_query}\n\n[相关历史记忆]:\n{context}"
        else:
            enriched_query = user_query

        # 判断任务复杂度（简单 vs 复杂）
        if self._is_complex_task(enriched_query):
            logger.info("[AgentOrchestrator] 识别为复杂任务，启动 Task Planner")
            return self.task_executor.execute_plan(enriched_query)
        else:
            logger.info("[AgentOrchestrator] 简单任务，直接走 ReAct")
            return self.react_loop.execute(enriched_query)

    def chat_stream(self, user_query: str) -> Generator[str, None, str]:
        """
        流式处理用户消息
        """
        logger.info(f"[AgentOrchestrator] 流式处理: {user_query[:50]}...")

        memories = self.long_term_memory.recall(user_query)
        if memories:
            context = "\n".join([m["content"] for m in memories])
            enriched_query = f"{user_query}\n\n[相关历史记忆]:\n{context}"
        else:
            enriched_query = user_query

        if self._is_complex_task(enriched_query):
            yield "[检测到复杂任务，正在分解...]\n"
            result = self.task_executor.execute_plan(enriched_query)
            yield result
            return result
        else:
            return self.react_loop.execute_stream(enriched_query)

    def _is_complex_task(self, query: str) -> bool:
        """
        简单判断任务复杂度
        如果涉及多个动作、分析、对比等，认为是复杂任务
        """
        complex_keywords = [
            "分析", "对比", "比较", "整理", "总结", "报告",
            "所有", "全部", "各个", "分别", "同时",
            "然后", "之后", "先", "再",
            "每月", "每周", "统计", "趋势",
        ]
        query_lower = query
        keyword_count = sum(1 for kw in complex_keywords if kw in query_lower)
        return keyword_count >= 2 or len(query) > 100

    def get_available_tools(self) -> list[dict]:
        """获取可用工具列表"""
        return [t.schema.to_dict() for t in get_all_tools()]

    def get_session_history(self) -> list[dict]:
        """获取当前会话历史"""
        return self.short_term_memory.get_all()

    def clear_session(self):
        """清空当前会话"""
        self.short_term_memory.clear()
        self.working_memory.clear()
