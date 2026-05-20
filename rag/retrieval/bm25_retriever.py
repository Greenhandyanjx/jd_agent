"""
RAG Retrieval: BM25 关键词检索（支持中文）
作为向量检索的补充，用于多路召回

中文分词策略：按单字切分（CJK 字符），英文按单词切分。
这是轻量级方案，无需依赖 jieba 等外部分词器。
更精确的场景可替换为 jieba 分词。
"""
import math
import re
from typing import Optional
from collections import Counter
from utils.logger import logger


# 汉字 Unicode 范围
_CJK_RANGE = re.compile(r'[一-鿿]|[a-zA-Z0-9]+')


def _tokenize(text: str) -> list[str]:
    """
    分词：中文按单字切分，英文/数字按单词切分。

    "扫地机器人 吸力3000Pa" → ["扫", "地", "机", "器", "人", "吸力", "3000", "Pa"]
    """
    return [m.group() for m in _CJK_RANGE.finditer(text.lower())]


class BM25Retriever:
    """
    BM25 检索器，支持混合中英文文本。

    核心改进（相比原版）：
    - 正确处理中文分词（原版用 str.split() 把整句中文字当成了一个 token）
    - 文档数和平均文档长度基于 token 而非 raw str
    """

    def __init__(self):
        self.documents: list[str] = []
        self.doc_freq: dict[str, int] = {}
        self.avgdl: float = 0
        self.k1: float = 1.5
        self.b: float = 0.75
        self._initialized = False

    def add_documents(self, documents: list[str]):
        """添加文档，构建 BM25 索引"""
        self.documents = documents
        # 使用 tokenize 计算文档长度
        doc_token_lens = [len(_tokenize(d)) for d in documents]
        self.avgdl = sum(doc_token_lens) / max(len(documents), 1)

        # 计算文档频率（一个 term 出现在多少篇文档中）
        for doc in documents:
            terms = set(_tokenize(doc))
            for term in terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self._initialized = True
        logger.info(f"[BM25Retriever] 已加载 {len(documents)} 篇文档，词典大小 {len(self.doc_freq)}")

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """
        检索与查询最相关的文档。
        返回 [(文档文本, BM25 得分), ...]
        """
        if not self._initialized or not self.documents:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scores = []
        for doc in self.documents:
            score = self._bm25_score(query_terms, doc)
            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _bm25_score(self, query_terms: list[str], doc: str) -> float:
        """计算 doc 在 query_terms 下的 BM25 得分"""
        doc_len = len(_tokenize(doc))
        if doc_len == 0:
            return 0.0

        score = 0.0
        n_docs = len(self.documents)

        for term in query_terms:
            tf = doc.count(term) if len(term) == 1 else doc.count(term)
            if tf == 0:
                continue

            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score
