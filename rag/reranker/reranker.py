"""
RAG Reranker: 重排序器
对检索结果进行精排，提高最终回答质量
"""
from typing import Optional
from langchain_core.documents import Document
from utils.logger import logger
from model.factory import chat_model as llm


class Reranker:
    """
    重排序器
    使用交叉编码器或LLM对检索结果重排序
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.model = llm

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        对检索文档重排序
        使用 LLM 进行相关性打分，保留 top_k 个
        """
        if not documents:
            return documents

        # 使用 LLM 进行相关性评分
        scored_docs = []
        for doc in documents:
            score = self._score_relevance(query, doc.page_content)
            doc.metadata["score"] = score
            scored_docs.append((doc, score))

        # 按得分排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 截取 top_k
        result = [doc for doc, _ in scored_docs[:self.top_k]]
        logger.info(f"[Reranker] 重排序完成: {len(documents)}→{len(result)} 篇")
        return result

    def _score_relevance(self, query: str, doc: str) -> float:
        """
        对单篇文档进行相关性打分（0-1）
        使用简单关键词匹配 + 长度归一化
        """
        query_terms = set(query.lower().split())
        doc_terms = set(doc.lower().split())

        if not query_terms:
            return 0.5

        # Jaccard 相似度
        intersection = query_terms & doc_terms
        union = query_terms | doc_terms
        jaccard = len(intersection) / len(union) if union else 0

        # 覆盖度
        coverage = len(intersection) / len(query_terms) if query_terms else 0

        # 综合得分
        score = 0.6 * coverage + 0.4 * jaccard
        return min(score, 1.0)
