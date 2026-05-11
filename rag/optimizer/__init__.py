"""
RAG 优化器：查询改写、HyDE 等

QueryRewriter（增强版）：
- 支持 search/expand/decompose 三种改写策略
- LLM 改写 + 规则降级 + 重试机制
- 多查询版本生成（rewrite_multi）
- Pydantic 参数校验
"""

from rag.optimizer.query_rewrite import QueryRewriter

__all__ = ["QueryRewriter"]
