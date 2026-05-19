---
name: rag
description: "知识库检索：从产品文档/技术手册中检索相关知识和信息。Use when: user needs to query product documentation, technical manuals, or knowledge base."
version: "1.0.0"
categories: ["知识检索", "企业服务"]
metadata:
  openclaw:
    emoji: "📚"
    requires:
      python: ["chromadb", "sentence-transformers"]
---

# RAG Skill

从知识库中检索与用户问题相关的文档内容。支持多路召回（向量+BM25+RRF融合）、Query Rewrite 和 Reranker 重排序。

## When to Use

✅ **USE this skill when:**

- "如何安装XXX？"
- "文档中有关于XXX的描述吗？"
- "这个产品的功能有哪些？"
- 需要从知识库中获取事实信息

## When NOT to Use

❌ **DON'T use this skill when:**

- 需要实时外部信息（用 web_fetch）
- 用户闲聊（直接用 LLM 回复）

## Abilities

### rag_query

从知识库中检索与查询相关的文档内容。

**参数：**
- `query` (string, 必填): 检索查询字符串

**检索方式：**
- 向量检索（ChromaDB + Sentence Embedding）
- BM25 关键词检索
- RRF 融合排序
- Reranker 重排序
