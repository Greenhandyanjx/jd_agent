"""
RAG Retrieval: 混合检索（多路召回 + 融合）
"""
from typing import Optional
from utils.logger import logger
from rag.retrieval.vector_store import VectorStoreService
from rag.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    混合检索器
    结合向量检索和 BM25 关键词检索，实现多路召回
    """

    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3):
        self.vector_service = VectorStoreService()
        self.bm25_retriever = BM25Retriever()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """
        多路检索融合
        返回 [(文档文本, 融合得分), ...]
        """
        # 向量检索
        vector_docs = self.vector_service.get_retriever().invoke(query)
        vector_results = {
            doc.page_content: self.vector_weight * (1.0 - i / len(vector_docs))
            for i, doc in enumerate(vector_docs)
        }

        # BM25 检索
        bm25_results_list = self.bm25_retriever.retrieve(query, k=k)
        bm25_results = {
            doc: self.bm25_weight * (1.0 - i / len(bm25_results_list)) * score
            for i, (doc, score) in enumerate(bm25_results_list)
            if score > 0
        }

        # 融合（RRF 算法简化版）
        from collections import defaultdict
        merged = defaultdict(float)

        for doc, score in vector_results.items():
            merged[doc] += score
        for doc, score in bm25_results.items():
            merged[doc] += score

        sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
