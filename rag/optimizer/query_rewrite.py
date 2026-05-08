"""
RAG Optimizer: 查询改写
改写用户原始问题以提高检索命中率
"""
from typing import Optional
from utils.logger import logger
from model.factory import chat_model as llm


class QueryRewriter:
    """
    查询改写器
    将用户的模糊问题改写成更明确的检索查询
    """

    def __init__(self):
        self.model = llm
        self.rewrite_prompt = """你是一个查询改写专家。用户的提问可能比较模糊，请将其改写成更适合检索的清晰查询。

要求:
1. 保持原始意图不变
2. 补充缺失的关键词
3. 去除冗余的修饰词
4. 使其更像一个"搜索引擎查询"
5. 直接返回改写后的查询语句，不要解释

原始提问: {query}
改写后:"""

    def rewrite(self, query: str) -> str:
        """
        改写用户查询
        """
        try:
            response = self.model.invoke([
                {"role": "system", "content": self.rewrite_prompt.format(query=query)}
            ])
            rewritten = response.content if hasattr(response, 'content') else str(response)
            rewritten = rewritten.strip().strip('"').strip("'")
            return rewritten
        except Exception as e:
            logger.warning(f"[QueryRewriter] 改写失败: {e}，使用原始查询")
            return query
