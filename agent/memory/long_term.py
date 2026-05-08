"""
Memory: 长期记忆
持久化跨会话的记忆，使用向量数据库进行语义检索
"""
import json
from typing import Optional
from datetime import datetime
from utils.logger import logger
from utils.config import chroma_conf
from langchain_chroma import Chroma
from model.factory import embed_model


class LongTermMemory:
    """
    长期记忆：跨会话持久化
    保存用户偏好、历史行为、重要结论
    通过语义相似度检索相关记忆
    """

    def __init__(self, collection_name: str = "long_term_memory"):
        self.collection_name = collection_name
        try:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embed_model,
                persist_directory=chroma_conf.get("persist_directory", "db/chroma_db"),
            )
            logger.info(f"[LongTermMemory] 初始化长期记忆存储: {collection_name}")
        except Exception as e:
            logger.warning(f"[LongTermMemory] 初始化失败（可能是首次使用）: {e}")
            self.vector_store = None

    def remember(self, key: str, content: str, metadata: Optional[dict] = None) -> bool:
        """
        记住一条信息
        key: 记忆的唯一标识
        content: 记忆内容
        metadata: 附加元数据（如时间、来源等）
        """
        if self.vector_store is None:
            logger.warning("[LongTermMemory] 存储未初始化，无法保存")
            return False

        try:
            meta = metadata or {}
            meta["key"] = key
            meta["timestamp"] = datetime.now().isoformat()

            from langchain_core.documents import Document
            doc = Document(page_content=content, metadata=meta)
            self.vector_store.add_documents([doc])
            logger.info(f"[LongTermMemory] 记住: {key}")
            return True
        except Exception as e:
            logger.error(f"[LongTermMemory] 保存失败: {e}")
            return False

    def recall(self, query: str, k: int = 3) -> list[dict]:
        """
        根据语义检索相关记忆
        """
        if self.vector_store is None:
            return []

        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", None),
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"[LongTermMemory] 检索失败: {e}")
            return []

    def forget(self, key: str) -> bool:
        """
        忘记一条信息（根据key删除）
        """
        if self.vector_store is None:
            return False
        try:
            # Chroma 通过 metadata 中的 key 来过滤删除
            self.vector_store.delete(filter={"key": key})
            logger.info(f"[LongTermMemory] 遗忘: {key}")
            return True
        except Exception as e:
            logger.error(f"[LongTermMemory] 删除失败: {e}")
            return False

    def get_stats(self) -> dict:
        """获取长期记忆统计"""
        if self.vector_store is None:
            return {"status": "uninitialized", "count": 0}
        try:
            count = len(self.vector_store.get()["ids"])
            return {"status": "ready", "count": count}
        except:
            return {"status": "error", "count": 0}
