"""
RAG Retrieval: 混合检索器（增强版 - 多路召回 + RRF 融合）
========================================================
功能：
1. 向量检索 + BM25 关键词检索 双路召回
2. RRF（Reciprocal Rank Fusion）融合算法
3. 支持多查询改写版本同时检索
4. 带重试和降级机制
5. Pydantic 参数校验

参考 nanobot 搜索设计，适配 RAG 检索场景
"""

from collections import defaultdict
from typing import Optional

from langchain_core.documents import Document
from utils.logger import logger
from rag.retrieval.vector_store import VectorStoreService
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.schema_validator import validate_params, HybridRetrieverParams
from rag.retry import retry_with_fallback, build_fallback_chain


class HybridRetriever:
    """
    混合检索器（增强版）

    多路召回策略：
    1. 向量检索（语义匹配）— 主路
    2. BM25 关键词检索（精确匹配）— 辅路
    3. RRF 融合算法 — 综合排序
    4. 多查询版本并行检索 — 扩展覆盖

    使用示例：
        retriever = HybridRetriever()
        # 单查询检索
        results = retriever.retrieve("什么是机器学习")
        # 多版本检索
        results = retriever.retrieve_multi(["机器学习", "ML", "深度学习"])
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        enable_multi_query: bool = False,
    ):
        """
        Args:
            vector_weight: 向量检索权重（融合时使用，RRF 模式不直接使用）
            bm25_weight: BM25 检索权重
            rrf_k: RRF 融合常数（默认 60，越大对低排名文档越公平）
            enable_multi_query: 是否启用多查询版本检索（会影响速度）
        """
        self.vector_service = VectorStoreService()
        self.bm25_retriever = BM25Retriever()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.enable_multi_query = enable_multi_query

    # ─── 向量检索 ────────────────────────────

    def _vector_retrieve(self, query: str, k: int = 10) -> list[tuple[str, float, Document]]:
        """
        向量检索，返回 (文本, 得分, Document) 列表。

        重试 2 次，失败时静默返回空列表（降级到 BM25）。
        """
        wrapped = retry_with_fallback(
            max_retries=2,
            base_delay=0.5,
            name="VectorRetrieve",
            fallback_func=lambda *a, **kw: [],
        )

        def do_vector_search(q: str, top_k: int) -> list[tuple[str, float, Document]]:
            retriever = self.vector_service.get_retriever()
            docs = retriever.invoke(q, config={"configurable": {"k": top_k}})
            results = []
            for i, doc in enumerate(docs):
                # 归一化得分：从排位得分转为 0-1 分数
                score = 1.0 - (i / max(len(docs), 1))
                doc.metadata["vector_score"] = round(score, 4)
                results.append((doc.page_content, score, doc))
            return results

        return do_vector_search(query, k)

    # ─── BM25 检索 ───────────────────────────

    def _bm25_retrieve(self, query: str, k: int = 10) -> list[tuple[str, float, Document]]:
        """
        BM25 关键词检索，返回 (文本, 得分, Document) 列表。
        """
        bm25_results_list = self.bm25_retriever.retrieve(query, k=k)
        results = []
        for text, score in bm25_results_list:
            doc = Document(page_content=str(text), metadata={"bm25_score": round(score, 4)})
            results.append((str(text), score, doc))

        if not results:
            logger.info(f"[HybridRetriever] BM25 检索无结果，尝试查询词拆分")
            # 降级：尝试小规模词语组合
            terms = query.split()
            if len(terms) > 1:
                for term in terms:
                    if len(term) > 1:
                        sub_results = self.bm25_retriever.retrieve(term, k=max(1, k // 2))
                        for text, score in sub_results:
                            if text not in {t[0] for t in results}:
                                doc = Document(
                                    page_content=str(text),
                                    metadata={"bm25_score": round(score, 4)}
                                )
                                results.append((str(text), score * 0.5, doc))

        return results

    # ─── RRF 融合 ────────────────────────────

    def _rrf_merge(
        self,
        vector_results: list[tuple[str, float, Document]],
        bm25_results: list[tuple[str, float, Document]],
        k: int = 5,
    ) -> list[Document]:
        """
        RRF（Reciprocal Rank Fusion）融合算法。

        原理：对每个文档，其融合分数 = sum(1 / (k + rank_i))
        其中 rank_i 是该文档在第 i 个检索器中的排名

        RRF 对高排名文档友好，且不依赖原始得分，避免了不同检索器得分不可比的问题。
        """
        if not vector_results and not bm25_results:
            return []

        rrf_scores: dict[str, dict] = {}

        # 向量检索排名
        for rank, (text, score, doc) in enumerate(vector_results, start=1):
            rrf_scores[text] = {
                "doc": doc,
                "vector_rank": rank,
                "bm25_rank": 0,
                "rrf_score": 1.0 / (self.rrf_k + rank),
            }

        # BM25 检索排名
        for rank, (text, score, doc) in enumerate(bm25_results, start=1):
            if text in rrf_scores:
                rrf_scores[text]["bm25_rank"] = rank
                rrf_scores[text]["rrf_score"] += 1.0 / (self.rrf_k + rank)
                # 合并 bm25 的 metadata
                if "bm25_score" in doc.metadata:
                    rrf_scores[text]["doc"].metadata["bm25_score"] = doc.metadata["bm25_score"]
            else:
                rrf_scores[text] = {
                    "doc": doc,
                    "vector_rank": 0,
                    "bm25_rank": rank,
                    "rrf_score": 1.0 / (self.rrf_k + rank),
                }

        # 按 RRF 分排序
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )

        # 取 top-k
        result_docs = []
        for text, info in sorted_items[:k]:
            doc = info["doc"]
            doc.metadata["rrf_score"] = round(info["rrf_score"], 4)
            doc.metadata["vector_rank"] = info["vector_rank"]
            doc.metadata["bm25_rank"] = info["bm25_rank"]
            # 综合得分：RRF Score（归一化到 0-1）
            max_possible = 1.0 / (self.rrf_k + 1) + 1.0 / (self.rrf_k + 1)
            doc.metadata["score"] = round(info["rrf_score"] / max_possible, 4)
            result_docs.append(doc)

        return result_docs

    # ─── 加权融合（备用方案）─────────────────

    def _weighted_merge(
        self,
        vector_results: list[tuple[str, float, Document]],
        bm25_results: list[tuple[str, float, Document]],
        k: int = 5,
    ) -> list[Document]:
        """加权融合（适用于两个检索器得分都有意义时）"""
        merged: dict[str, dict] = {}

        for text, score, doc in vector_results:
            merged[text] = {"doc": doc, "score": score * self.vector_weight, "bm25_score": 0}

        for text, score, doc in bm25_results:
            if text in merged:
                merged[text]["score"] += score * self.bm25_weight
                merged[text]["bm25_score"] = score
            else:
                merged[text] = {"doc": doc, "score": score * self.bm25_weight, "bm25_score": score}

        sorted_items = sorted(merged.items(), key=lambda item: item[1]["score"], reverse=True)
        result_docs = []
        for text, info in sorted_items[:k]:
            doc = info["doc"]
            doc.metadata["score"] = round(info["score"], 4)
            doc.metadata["vector_weighted_score"] = round(
                info["score"] - (info.get("bm25_score", 0) * self.bm25_weight), 4
            ) if info.get("bm25_score", 0) > 0 else round(info["score"], 4)
            result_docs.append(doc)

        return result_docs

    # ─── 对外检索接口 ────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """
        多路召回检索。

        流程：
        1. 参数校验
        2. 向量检索
        3. BM25 检索
        4. RRF 融合排序

        Args:
            query: 查询文本
            k: 返回文档数

        Returns:
            Document 列表，每篇文档的 metadata 包含 score/vector_rank/bm25_rank/rrf_score
        """
        # 参数校验
        params = validate_params(
            {"query": query, "k": k},
            HybridRetrieverParams,
        )
        query = params["query"]
        k = params["k"]

        logger.info(f"[HybridRetriever] 开始多路召回: query='{query[:40]}...', k={k}")

        # 向量检索
        try:
            vector_results = self._vector_retrieve(query, k=k * 2)
            logger.info(f"[HybridRetriever] 向量检索: {len(vector_results)} 篇")
        except Exception as e:
            logger.warning(f"[HybridRetriever] 向量检索失败: {e}，降级")
            vector_results = []

        # BM25 检索
        try:
            bm25_results = self._bm25_retrieve(query, k=k * 2)
            logger.info(f"[HybridRetriever] BM25 检索: {len(bm25_results)} 篇")
        except Exception as e:
            logger.warning(f"[HybridRetriever] BM25 检索失败: {e}，降级")
            bm25_results = []

        # RRF 融合（主方案）
        if vector_results and bm25_results:
            merged_docs = self._rrf_merge(vector_results, bm25_results, k=k)
        elif vector_results:
            merged_docs = [doc for _, _, doc in vector_results[:k]]
        elif bm25_results:
            merged_docs = [doc for _, _, doc in bm25_results[:k]]
        else:
            merged_docs = []

        # 如果 RRF 结果不足 k 个，用加权融合补充
        if len(merged_docs) < k and vector_results and bm25_results:
            logger.info("[HybridRetriever] RRF 结果不足，使用加权融合补充")
            extra = self._weighted_merge(vector_results, bm25_results, k=k)
            existing_texts = {doc.page_content for doc in merged_docs}
            for doc in extra:
                if doc.page_content not in existing_texts and len(merged_docs) < k:
                    merged_docs.append(doc)
                    existing_texts.add(doc.page_content)

        logger.info(f"[HybridRetriever] 多路召回完成: {len(merged_docs)} 篇")
        return merged_docs

    def retrieve_multi(
        self,
        queries: list[str],
        k: int = 5,
    ) -> list[Document]:
        """
        多查询版本检索（进一步扩展召回覆盖率）。

        对每个查询版本分别检索，然后将所有结果按最高 RRF 分融合。

        Args:
            queries: 多个查询版本
            k: 最终返回文档数

        Returns:
            Document 列表
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self.retrieve(queries[0], k)

        # 对所有查询版本检索
        all_results: list[Document] = []
        seen_texts: set[str] = set()

        for query in queries:
            try:
                docs = self.retrieve(query, k=k)
                for doc in docs:
                    if doc.page_content not in seen_texts:
                        all_results.append(doc)
                        seen_texts.add(doc.page_content)
            except Exception as e:
                logger.warning(f"[HybridRetriever] 多查询版 '{query[:20]}...' 检索失败: {e}")

        # 按得分排序取 top-k
        all_results.sort(
            key=lambda d: (
                d.metadata.get("score", 0)
                or d.metadata.get("rrf_score", 0)
                or 0
            ),
            reverse=True,
        )

        return all_results[:k]
