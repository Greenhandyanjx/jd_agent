"""
RAG Service: RAG 服务入口
继承原始项目的 RAG 实现，增加多路召回和重排序
"""
from typing import Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils.logger import logger
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from rag.retrieval.vector_store import VectorStoreService
from rag.optimizer.query_rewrite import QueryRewriter
from rag.reranker.reranker import Reranker


class RagSummarizeService:
    """
    RAG 检索增强生成服务
    在原始项目基础上增加了 Query Rewrite + 多路召回 + Reranker
    """

    def __init__(self, enable_optimization: bool = True):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.prompt_template | self.model | StrOutputParser()
        self.query_rewriter = QueryRewriter() if enable_optimization else None
        self.reranker = Reranker() if enable_optimization else None
        self.enable_optimization = enable_optimization

    def retriever_docs(self, query: str, k: int = 5) -> list[Document]:
        """检索文档"""
        return self.retriever.invoke(query, config={"configurable": {"k": k}})

    def rag_summarize(self, query: str) -> str:
        """
        RAG 查询主流程
        1. 查询改写（可选）
        2. 向量检索 / 多路召回
        3. 重排序（可选）
        4. LLM 生成
        """
        # Step 1: Query Rewrite
        final_query = query
        if self.query_rewriter:
            rewritten = self.query_rewriter.rewrite(query)
            if rewritten and rewritten != query:
                final_query = rewritten
                logger.info(f"[RAG] 查询改写: '{query[:30]}...' → '{final_query[:30]}...'")

        # Step 2: 检索
        context_docs = self.retriever_docs(final_query)
        logger.info(f"[RAG] 检索到 {len(context_docs)} 篇文档")

        if not context_docs:
            return "抱歉，知识库中没有找到相关的内容。请换个问题试试。"

        # Step 3: Reranker（重排序）
        if self.reranker and len(context_docs) > 1:
            reranked = self.reranker.rerank(query, context_docs)
            if reranked:
                context_docs = reranked
                logger.info(f"[RAG] 重排序完成")

        # Step 4: 构建上下文 & 生成
        context = ""
        for i, doc in enumerate(context_docs, 1):
            score_str = f" | 相关度: {doc.metadata.get('score', 'N/A')}" if doc.metadata else ""
            context += f"参考文档{i}: {doc.page_content}{score_str}\n"

        result = self.chain.invoke({
            "input": query,
            "context": context,
        })

        return result

    def update_knowledge_base(self):
        """重新加载知识库"""
        self.vector_store.load_document()
        logger.info("[RAG] 知识库已更新")
