"""
RAG 检索增强系统（2026-05-11 Agent 集成版）
===========================================
设计原则：RAG 回归"检索"角色，不生成回答。
Agent 通过 RAGQueryTool 调用 RAG 获取知识素材，自主生成回答。

功能模块：
① schema_validator.py  — 工具参数 Pydantic 校验
② retry.py            — 重试 + 降级机制
③ optimizer/          — 查询改写（LLM + 规则）
④ reranker/           — 重排序（关键词 + LLM 交叉 + 混合）
⑤ retrieval/          — 多路召回（向量 + BM25 + RRF 融合）

全链路入口：RagRetrievalService（rag_service.py）
           ↓ 被 RAGQueryTool 调用（agent/tools/tool_definitions.py）
           ↓ AgentLoop 调度（agent/loop.py）
"""

from rag.rag_service import RagRetrievalService
from rag.schema_validator import validate_params
from rag.retry import retry_with_fallback, retry_async_with_fallback, build_fallback_chain

__all__ = [
    "RagRetrievalService",
    "validate_params",
    "retry_with_fallback",
    "retry_async_with_fallback",
    "build_fallback_chain",
]
