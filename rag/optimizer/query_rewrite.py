"""
RAG Optimizer: 查询改写模块（增强版）
=====================================
功能：
1. 多种改写策略：search（搜索优化）、expand（语义扩展）、decompose（分解复杂查询）
2. 去停用词 + 关键词补全
3. 重试机制（LLM 调用失败时降级到规则改写）
4. Pydantic 参数校验

参考 nanobot 的 query rewrite 思路，适配 RAG 检索场景
"""

import re
from typing import Optional
from utils.logger import logger
from model.factory import chat_model as llm
from rag.schema_validator import validate_params, QueryRewriteParams
from rag.retry import retry_with_fallback


# 中英文停用词集合（业务常用词）
_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "那", "这个", "那个", "什么", "怎么",
    "如何", "为什么", "请", "请问", "能", "可以", "应该", "需要", "想", "知道",
    "他", "她", "它", "们", "其", "中", "等", "与", "及", "或", "但", "而",
    "且", "因为", "所以", "虽然", "但是", "如果", "那么", "然后", "之后",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "shall", "need", "dare", "used", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just",
}


class QueryRewriter:
    """
    查询改写器（增强版）

    支持三种改写策略：
    - search: 搜索优化（默认），将自然语言问题改写为关键词查询
    - expand: 语义扩展，补充同义词和相关概念
    - decompose: 复杂查询分解，将一个问题拆成多个子查询

    策略选择：
    - 短查询（<10字）→ expand 扩展
    - 长查询（>50字）→ decompose 分解
    - 一般查询 → search 优化
    """

    def __init__(self, default_style: str = "auto"):
        """
        Args:
            default_style: 默认改写风格
                - "auto": 自动选择
                - "search": 搜索优化
                - "expand": 语义扩展
                - "decompose": 查询分解
        """
        self.model = llm
        self.default_style = default_style
        self._prompts = self._init_prompts()

    # ─── 提示词模板 ──────────────────────────

    def _init_prompts(self) -> dict:
        return {
            "search": """你是一个查询改写专家。请将用户的问题改写成更适合"向量检索"的查询语句。

改写要求：
1. 保持原始意图不变
2. 补充缺失的关键领域术语和关键词
3. 去除冗余修饰词和口语化表达
4. 若有具体名词（人名、产品名、专有名词）必须保留
5. 直接返回改写后的查询语句，不要任何解释或前缀

原始问题: {query}
改写后:""",

            "expand": """你是一个查询扩展专家。请对用户的查询进行语义扩展，补充相关关键词。

改写要求：
1. 保留原始查询的所有内容
2. 在最后补充 3-5 个相关同义词、近义词或关联概念（用空格分隔）
3. 扩展词应与原查询属于同一领域
4. 直接返回扩展后的完整查询，不要解释

原始查询: {query}
扩展后:""",

            "decompose": """你是一个复杂查询分解专家。请将用户的复杂问题分解成多个独立的子查询。

改写要求：
1. 识别问题中的不同子主题
2. 每个子查询用换行符分隔
3. 每个子查询应能独立作为检索查询
4. 保持每个子查询的语义完整性
5. 直接返回分解后的子查询列表，不要解释

原始问题: {query}
分解后:""",
        }

    # ─── 核心改写方法 ───────────────────────

    @retry_with_fallback(
        max_retries=2,
        base_delay=0.5,
        name="QueryRewrite",
        fallback_func=None,  # 由 rewrite() 自己处理降级
    )
    def _llm_rewrite(self, query: str, style: str) -> str:
        """调用 LLM 改写查询（带重试）"""
        prompt_text = self._prompts.get(style, self._prompts["search"])
        response = self.model.invoke([
            {"role": "system", "content": prompt_text.format(query=query)}
        ])
        rewritten = response.content if hasattr(response, 'content') else str(response)
        rewritten = rewritten.strip().strip('"').strip("'").strip('"')
        return rewritten

    def _rule_rewrite(self, query: str) -> str:
        """
        纯规则改写（LLM 不可用时的降级方案）。

        规则：
        1. 去除停用词
        2. 补充常见领域前缀
        3. 压缩连续空格
        """
        # 分词（简单空格+标点分割）
        # 循环替换各类标点为空格，避免正则 escape 问题
        for p in [',', '.', '!', '?', ':', ';', '，', '。', '！', '？',
                  '、', '；', '：', '"', '"', '\'', '(', ')', '【', '】',
                  '[', ']', '{', '}']:
            query = query.replace(p, ' ')
        tokens = query.split()
        # 去停用词
        filtered = [t for t in tokens if t and t.lower() not in _STOP_WORDS]
        if not filtered:
            return query  # 全被滤掉了，返回原文

        result = " ".join(filtered)

        # 如果改写后太短，尝试补充
        if len(result) < 5 and len(query) > 5:
            # 保留原查询中的名词、专有名词
            return query

        return result

    def _auto_select_style(self, query: str) -> str:
        """自动选择改写策略"""
        query_len = len(query)
        if query_len < 10:
            return "expand"
        elif query_len > 50:
            return "decompose"
        else:
            return "search"

    # ─── 对外接口 ────────────────────────────

    def rewrite(self, query: str, style: Optional[str] = None) -> str:
        """
        改写用户查询，提高检索命中率。

        Args:
            query: 原始用户查询
            style: 改写风格（search/expand/decompose/auto）
                    默认为初始化时设置的 default_style

        Returns:
            改写后的查询字符串

        Raises:
            ValueError: 参数校验失败
        """
        # 参数校验
        params = validate_params(
            {"query": query, "style": style or self.default_style},
            QueryRewriteParams,
        )
        query = params["query"]
        style = params["style"]

        # 自动选择策略
        if style == "auto":
            style = self._auto_select_style(query)

        try:
            rewritten = self._llm_rewrite(query, style)
            if rewritten and rewritten != query:
                logger.info(f"[QueryRewriter] {style.upper()} 改写: "
                            f"'{query[:40]}...' → '{rewritten[:40]}...'")
                return rewritten
        except Exception as e:
            logger.warning(f"[QueryRewriter] LLM 改写失败 ({style}): {e}，降级到规则改写")

        # LLM 改写失败或退化 → 规则改写
        rule_result = self._rule_rewrite(query)
        if rule_result != query:
            logger.info(f"[QueryRewriter] 规则改写: '{query[:40]}...' → '{rule_result[:40]}...'")
        return rule_result

    def rewrite_multi(self, query: str, style: str = "auto") -> list[str]:
        """
        返回多个改写版本（多路改写），用于多路召回进一步提升覆盖率。

        Args:
            query: 原始查询
            style: 改写风格

        Returns:
            改写版本列表（至少包含原始查询）
        """
        versions = [query]  # 保留原始版本

        try:
            # search 风格改写
            search_q = self.rewrite(query, "search")
            if search_q != query and search_q not in versions:
                versions.append(search_q)
        except Exception:
            pass

        try:
            # expand 风格改写
            expand_q = self.rewrite(query, "expand")
            if expand_q != query and expand_q not in versions:
                versions.append(expand_q)
        except Exception:
            pass

        # 规则改写（无异常风险）
        rule_q = self._rule_rewrite(query)
        if rule_q != query and rule_q not in versions:
            versions.append(rule_q)

        return versions
