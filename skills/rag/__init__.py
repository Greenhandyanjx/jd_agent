"""
JD-Agent 示例技能：RAG 知识库检索
====================================
提供知识库检索能力，从向量数据库和 BM25 引擎中检索相关文档。
"""

from agent.skills.base import Skill, SkillAbility
from agent.tools.base import Tool
from rag.rag_service import RagRetrievalService


class RAGSearchTool(Tool):
    """知识库检索（纯检索，不生成回答）"""

    def __init__(self):
        self._rag = RagRetrievalService()

    @property
    def name(self) -> str:
        return "rag_query"

    @property
    def description(self) -> str:
        return (
            "从知识库中检索与用户问题相关的文档内容。"
            "当需要产品知识、技术文档、操作手册等信息时调用此工具。"
            "返回原始文档内容，不生成回答。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "query": {
                "type": "string",
                "description": "检索查询，如：'如何安装扫地机器人'",
                "required": True,
            },
        }

    async def execute(self, query: str, **kwargs) -> str:
        """从 RAG 知识库中检索相关内容"""
        try:
            result = self._rag.retrieve(query)
            if not result or result.strip() == "":
                return f"未在知识库中找到与 '{query}' 相关的内容。"
            return result
        except Exception as e:
            return f"RAG 检索失败: {e}"


class RAGSkill(Skill):
    """
    RAG 知识库检索技能。
    
    允许 Agent 从产品文档、技术手册等知识库中检索相关内容。
    """

    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return "知识库检索技能：从产品文档/技术手册中检索相关知识和信息"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def categories(self) -> list[str]:
        return ["知识检索", "企业服务"]

    @property
    def abilities(self) -> list[SkillAbility]:
        return [
            SkillAbility(
                name="rag_query",
                description="从知识库中检索相关文档内容",
                keywords=[
                    "知识", "文档", "手册", "指南", "教程", "说明", "介绍",
                    "什么是", "如何", "怎么", "用法", "功能", "规格",
                    "产品", "技术", "查询", "搜索", "查找",
                ],
                examples=[
                    "如何安装扫地机器人？",
                    "产品有哪些功能？",
                    "什么是向量数据库？",
                    "说明书上关于XXX的内容",
                ],
                tool=RAGSearchTool(),
            ),
        ]


skill_instance = RAGSkill()
