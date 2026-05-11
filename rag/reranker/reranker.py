"""
RAG Reranker: 重排序器（增强版）
================================
功能：
1. 关键词匹配排序（Jaccard + 覆盖度 + TF加权）
2. LLM 交叉编码排序（两两比较）
3. 多策略融合排序
4. 参数 Pydantic 校验

参考 Cross-Encoder 设计思路，在无 GPU 环境下使用 LLM 模拟交叉编码
"""

import itertools
import re
from collections import Counter
from typing import Optional

from langchain_core.documents import Document
from utils.logger import logger
from model.factory import chat_model as llm
from rag.schema_validator import validate_params, RerankerParams

# 中英文停用词
_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "这个", "那个", "什么", "怎么", "如何",
    "为什么", "请", "请问", "能", "可以", "应该", "需要", "想", "知道",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "to", "of", "in", "for", "on", "with", "at",
    "by", "from", "as", "into", "through", "then", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "well",
}


class Reranker:
    """
    重排序器（增强版）

    提供三种排序策略：
    1. keyword: 关键词匹配排序（快速、无外部依赖）
    2. llm_cross: LLM 交叉编码排序（质量高、有模型调用开销）
    3. hybrid: 混合排序（先关键词初筛，再 LLM 精排 top-5）

    默认使用 hybrid 策略，在质量和效率间取得平衡。
    """

    def __init__(self, top_k: int = 3, strategy: str = "hybrid"):
        """
        Args:
            top_k: 排序后保留的文档数量
            strategy: 排序策略
                - "keyword": 纯关键词匹配（快速）
                - "llm_cross": LLM 交叉编码（高质量）
                - "hybrid": 先关键词初筛再 LLM 精排（默认，平衡模式）
        """
        self.top_k = top_k
        self.strategy = strategy
        self.model = llm
        self._pairwise_prompt = """你是一个相关性评分专家。请判断以下"文档"是否与"查询"相关。

请从0到10打分，只返回一个数字：
- 0 = 完全不相关
- 5 = 部分相关
- 10 = 高度相关

查询: {query}
文档: {document}
相关性评分:"""

    # ─── 关键词匹配评分 ──────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词，返回小写词列表"""
        # 中文分词：按字符拆分（简单方案，无外部分词器）
        # 英文分词：按空格和标点拆分
        tokens = []
        # 匹配中文字符和英文单词
        for match in re.finditer(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower()):
            tokens.append(match.group())
        return tokens

    def _keyword_score(self, query: str, document: str) -> float:
        """
        关键词相关性评分（0-1）
        使用 TF 加权 Jaccard 相似度
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        # 过滤停用词
        query_terms = [t for t in query_tokens if t not in _STOP_WORDS]
        doc_terms = [t for t in doc_tokens if t not in _STOP_WORDS]

        if not query_terms:
            return 0.5  # 无有效查询词，中性分数

        # 查询词在文档中的出现频率（TF 加权）
        query_set = set(query_terms)
        doc_counter = Counter(doc_terms)

        # 计算加权覆盖度：每个查询词最多加 1.0，TF>1 的加 0.2 bonus
        coverage_score = 0.0
        for term in query_set:
            tf = doc_counter.get(term, 0)
            if tf > 0:
                coverage_score += 1.0
                if tf > 1:
                    coverage_score += min(0.2 * (tf - 1), 0.5)  # bonus 上限 0.5

        max_score = len(query_set) * 1.5  # 每个词最多 1.5 分
        coverage = coverage_score / max_score if max_score > 0 else 0

        # Jaccard 相似度
        doc_set = set(doc_terms)
        intersection = query_set & doc_set
        union = query_set | doc_set
        jaccard = len(intersection) / len(union) if union else 0

        # 长文档惩罚（避免长文档因词更多而得分虚高）
        doc_length_penalty = min(1.0, 500.0 / max(len(doc_tokens), 1))

        # 综合得分
        score = 0.5 * coverage + 0.3 * jaccard + 0.2 * doc_length_penalty
        return min(score, 1.0)

    # ─── LLM 交叉编码评分 ────────────────────

    def _llm_pairwise_score(self, query: str, document: str) -> float:
        """
        使用 LLM 对 (query, document) 对进行相关性打分（0-10）。
        模拟 Cross-Encoder 的 pairwise 模式。
        """
        try:
            response = self.model.invoke([
                {"role": "system", "content": self._pairwise_prompt.format(
                    query=query[:300], document=document[:800]
                )}
            ])
            content = response.content if hasattr(response, 'content') else str(response)
            # 提取数字
            numbers = re.findall(r'\d+(?:\.\d+)?', content.strip())
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(10.0, score)) / 10.0  # 归一化到 0-1
        except Exception as e:
            logger.warning(f"[Reranker] LLM 评分失败: {e}")

        # 降级到关键词评分
        return self._keyword_score(query, document)

    # ─── 排序方法 ────────────────────────────

    def _keyword_rank(self, query: str, documents: list[Document]) -> list[Document]:
        """关键词匹配排序"""
        scored = []
        for doc in documents:
            score = self._keyword_score(query, doc.page_content)
            doc.metadata["score"] = round(score, 4)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:self.top_k]]

    def _llm_cross_rank(self, query: str, documents: list[Document]) -> list[Document]:
        """LLM 交叉编码排序（两两比较）"""
        scored = []
        for i, doc in enumerate(documents):
            score = self._llm_pairwise_score(query, doc.page_content)
            doc.metadata["score"] = round(score, 4)
            scored.append((doc, score))
            logger.debug(f"[Reranker] LLM 评分 [{i+1}/{len(documents)}]: {score:.4f}")

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:self.top_k]]

    def _hybrid_rank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        混合排序策略：
        1. 先用关键词初筛保留 top-10
        2. 再用 LLM 精排得到最终 top_k
        """
        if len(documents) <= self.top_k:
            # 文档数少，直接关键词排序即可
            return self._keyword_rank(query, documents)

        # 关键词初筛（保留 top-10 或 2x top_k）
        keyword_top_k = max(10, self.top_k * 2)
        scored = []
        for doc in documents:
            score = self._keyword_score(query, doc.page_content)
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = [doc for doc, _ in scored[:keyword_top_k]]

        # 对候选文档用 LLM 精排
        return self._llm_cross_rank(query, candidates)

    # ─── 对外接口 ────────────────────────────

    def rerank(self, query: str, documents: list[Document], top_k: Optional[int] = None) -> list[Document]:
        """
        对检索结果重排序。

        Args:
            query: 用户查询
            documents: 待排序文档列表
            top_k: （可选）覆盖实例 top_k

        Returns:
            排序后的文档列表（top_k 个）
        """
        # 参数校验
        k = top_k if top_k is not None else self.top_k
        params = validate_params({"top_k": k}, RerankerParams)
        k = params["top_k"]

        if not documents:
            return documents

        if len(documents) <= 1:
            for doc in documents:
                doc.metadata["score"] = doc.metadata.get("score", 0.5)
            return documents

        # 按策略排序
        if self.strategy == "llm_cross":
            result = self._llm_cross_rank(query, documents)
        elif self.strategy == "hybrid":
            result = self._hybrid_rank(query, documents)
        else:
            result = self._keyword_rank(query, documents)

        logger.info(f"[Reranker] {self.strategy.upper()} 排序: "
                     f"{len(documents)}→{len(result)} 篇")
        return result
