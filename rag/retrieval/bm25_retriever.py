"""
RAG Retrieval: BM25 关键词检索
作为向量检索的补充，用于多路召回
"""
from typing import Optional
from utils.logger import logger


class BM25Retriever:
    """
    BM25 检索器
    使用关键词匹配方式检索文档
    """

    def __init__(self):
        self.documents: list[str] = []
        self.doc_freq: dict[str, int] = {}
        self.avgdl: float = 0
        self.k1: float = 1.5
        self.b: float = 0.75
        self._initialized = False

    def add_documents(self, documents: list[str]):
        """添加文档"""
        self.documents = documents
        self.avgdl = sum(len(d.split()) for d in documents) / max(len(documents), 1)

        # 计算文档频率
        from collections import Counter
        for doc in documents:
            terms = set(doc.split())
            for term in terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self._initialized = True
        logger.info(f"[BM25Retriever] 已加载 {len(documents)} 篇文档")

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """
        检索与查询最相关的文档
        返回 [(文档文本, 得分), ...]
        """
        if not self._initialized or not self.documents:
            return []

        query_terms = query.split()
        scores = []

        for doc in self.documents:
            score = self._bm25_score(query_terms, doc)
            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _bm25_score(self, query_terms: list[str], doc: str) -> float:
        """计算 BM25 得分"""
        import math
        doc_len = len(doc.split())
        score = 0.0
        n_docs = len(self.documents)

        for term in query_terms:
            tf = doc.count(term)
            if tf == 0:
                continue

            df = self.doc_freq.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score
