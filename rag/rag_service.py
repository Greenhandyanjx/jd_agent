"""
RAG Service: RAG 检索增强生成服务（Agent 集成版）
==================================================
设计变更（2026-05-11）：RAG 回归纯检索角色

之前的问题：
═══════════
RagSummarizeService.rag_summarize() 内部直接调 LLM 生成答案，
和 AgentLoop 形成了"双重生成"：Agent 生成一次，RAG 又生成一次。

现在的设计：
══════════
RAG 只做"检索"——不再自己生成回答。
Agent 通过 RAGQueryTool 调用 RAG 获取知识素材，
然后 Agent 自己思考怎么用这些素材生成最终回答。

全链路：
  用户输入
    ↓
  AgentLoop (ReAct 循环)
    ↓
  LLM 决定调用 rag_search / rag_search_enhanced 工具
    ↓
  RAGQueryTool / RAGEnhancedTool
    │  ① Pydantic 参数校验 (schema_validator.py)
    │  ② Query Rewrite (query_rewrite.py)
    │  ③ 多路召回 (hybrid_retriever.py) · 带重试保护 (retry.py)
    │  ④ Reranker 重排序 (reranker.py)
    ↓
  原始文档内容 + 元数据 → 返回给 Agent
    ↓
  Agent + LLM 自主生成最终回答

功能模块整合（2026-05-11）：
① 工具参数 Schema 校验（Pydantic）✅  — schema_validator.py
② 重试机制（retry+fallback） ✅        — retry.py
③ Query Rewrite 模块 ✅                — optimizer/query_rewrite.py
④ Reranker 集成 ✅                     — reranker/reranker.py
⑤ 多路召回（BM25 + 向量检索 + RRF） ✅ — retrieval/hybrid_retriever.py
"""

from typing import Optional

from langchain_core.documents import Document

from utils.logger import logger
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model

from rag.schema_validator import validate_params, RetrieverParams
from rag.retry import retry_with_fallback
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.vector_store import VectorStoreService
from rag.optimizer.query_rewrite import QueryRewriter
from rag.reranker.reranker import Reranker


class RagRetrievalService:
    """
    RAG 检索服务（纯检索，不生成回答）
    
    ⚠️ 角色定位：Agent 的知识补给线
    只负责检索 → 返回文档，不负责回答。
    Agent 拿到文档后自己思考、自己生成。
    
    用法（被 RAGQueryTool 调用）：
        rag = RagRetrievalService()
        docs = rag.retrieve("什么是深度学习")
        # docs 是可读的文本，作为工具结果返回给 Agent
    """

    def __init__(
        self,
        enable_optimization: bool = True,
        rewrite_style: str = "auto",
        reranker_strategy: str = "hybrid",
        reranker_top_k: int = 5,
        multi_query: bool = False,
    ):
        """
        Args:
            enable_optimization: 是否启用优化（Rewrite + Reranker）
            rewrite_style: 改写风格（auto/search/expand/decompose）
            reranker_strategy: 重排序策略（keyword/hybrid/llm_cross）
            reranker_top_k: 排序后保留文档数
            multi_query: 是否启用多查询版本检索
        """
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

        # 优化组件
        self.query_rewriter = QueryRewriter(default_style=rewrite_style) if enable_optimization else None
        self.reranker = Reranker(top_k=reranker_top_k, strategy=reranker_strategy) if enable_optimization else None
        self.hybrid_retriever = HybridRetriever(enable_multi_query=multi_query) if enable_optimization else None
        self.enable_optimization = enable_optimization

        # 启动时初始化 BM25 索引（确保双路召回生效）
        self._init_bm25_index()

    # ─── 核心检索 ────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> str:
        """
        检索知识库，返回格式化的文档内容（纯文本，供 Agent 使用）。
        
        完整链路：
        ① Pydantic 参数校验
        ② Query Rewrite（LLM + 规则降级）
        ③ 多路召回（向量 + BM25 → RRF 融合）
        ④ Reranker（关键词/混合排序）
        
        Args:
            query: 用户查询
            k: 返回文档数
            
        Returns:
            格式化后的文档文本（Agent 可以直接用）
        """
        params = validate_params({"query": query, "k": k}, RetrieverParams)
        query, k = params["query"], params["k"]
        
        logger.info(f"[RAG] 开始检索: query='{query[:50]}...', k={k}")
        
        # ── Step 1: Query Rewrite ──
        final_query = query
        if self.query_rewriter:
            try:
                final_query = self.query_rewriter.rewrite(query)
                if final_query and final_query != query:
                    logger.info(f"[RAG] 查询改写: '{query[:30]}...' → '{final_query[:30]}...'")
            except Exception as e:
                logger.warning(f"[RAG] Query Rewrite 失败: {e}，使用原始查询")
                final_query = query
        
        # ── Step 2: 多路召回 ──
        context_docs: list[Document] = []
        if self.hybrid_retriever:
            try:
                context_docs = self.hybrid_retriever.retrieve(final_query, k=k)
            except Exception as e:
                logger.warning(f"[RAG] 多路召回失败: {e}，降级到基础向量检索")
        
        if not context_docs:
            # 降级到基础向量检索
            try:
                context_docs = self.retriever.invoke(final_query, config={"configurable": {"k": k}})
            except Exception as e:
                logger.error(f"[RAG] 所有检索方式均失败: {e}")
                return "知识库检索未返回结果，当前知识库中可能没有相关内容。"
        
        logger.info(f"[RAG] 检索到 {len(context_docs)} 篇文档")
        
        # ── Step 3: Reranker ──
        if self.reranker and len(context_docs) > 1:
            try:
                context_docs = self.reranker.rerank(query, context_docs)
                logger.info(f"[RAG] 重排序完成: {len(context_docs)} 篇")
            except Exception as e:
                logger.warning(f"[RAG] Reranker 失败: {e}，使用原始排序")
        
        # ── Step 4: 格式化为纯文本（给 Agent 用）──
        return self._format_docs_for_agent(context_docs)
    
    def retrieve_with_reranker(self, query: str, k: int = 5) -> tuple[list[Document], str]:
        """
        检索并返回 Document 对象 + 格式化文本（供需要原始文档的调用方使用）。
        
        Returns:
            (documents_list, formatted_text)
        """
        params = validate_params({"query": query, "k": k}, RetrieverParams)
        query, k = params["query"], params["k"]
        
        # 同 retrieve() 的逻辑...
        final_query = query
        if self.query_rewriter:
            try:
                final_query = self.query_rewriter.rewrite(query)
            except Exception:
                pass
        
        context_docs: list[Document] = []
        if self.hybrid_retriever:
            try:
                context_docs = self.hybrid_retriever.retrieve(final_query, k=k)
            except Exception:
                pass
        
        if not context_docs:
            try:
                context_docs = self.retriever.invoke(final_query, config={"configurable": {"k": k}})
            except Exception:
                pass
        
        if self.reranker and len(context_docs) > 1:
            try:
                context_docs = self.reranker.rerank(query, context_docs)
            except Exception:
                pass
        
        return context_docs, self._format_docs_for_agent(context_docs)
    
    # ─── 格式化工具 ──────────────────────────
    
    @staticmethod
    def _format_docs_for_agent(docs: list[Document]) -> str:
        """将检索到的文档格式化为供 Agent 使用的纯文本"""
        if not docs:
            return "知识库检索未返回结果。"
        
        parts = []
        for i, doc in enumerate(docs, 1):
            score_info = ""
            if doc.metadata:
                scores = []
                for key in ("score", "rrf_score", "vector_score", "bm25_score"):
                    val = doc.metadata.get(key)
                    if val is not None:
                        scores.append(f"{key}={val:.3f}")
                if scores:
                    score_info = " [" + ", ".join(scores) + "]"
            
            parts.append(f"[文档 {i}]{score_info}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(parts)
    
    # ─── 知识库管理 ──────────────────────────
    
    def update_knowledge_base(self):
        """重新加载知识库"""
        self.vector_store.load_document()
        logger.info("[RAG] 知识库已更新")
        
        if self.hybrid_retriever:
            try:
                all_docs = self.vector_store.get_all_documents()
                if all_docs:
                    texts = [doc.page_content for doc in all_docs]
                    self.hybrid_retriever.bm25_retriever.add_documents(texts)
                    logger.info(f"[RAG] BM25 索引已更新: {len(texts)} 篇")
            except Exception as e:
                logger.warning(f"[RAG] BM25 索引更新失败: {e}")

    def _init_bm25_index(self) -> None:
        """初始化 BM25 索引（启动时调用，确保双路召回生效）"""
        if not self.hybrid_retriever:
            return
        try:
            all_docs = self.vector_store.get_all_documents()
            if all_docs:
                texts = [doc.page_content for doc in all_docs]
                self.hybrid_retriever.bm25_retriever.add_documents(texts)
                logger.info(f"[RAG] BM25 索引初始化完成: {len(texts)} 篇")
        except Exception as e:
            logger.warning(f"[RAG] BM25 索引初始化失败（首次使用无数据）: {e}")
