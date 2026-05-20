"""
RAG Retrieval: 向量存储服务（FAISS 实现，替代原 ChromaDB）
"""
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger import logger


class _EmptyRetriever(BaseRetriever):
    """当向量索引不存在时返回空结果的兜底检索器。"""
    def _get_relevant_documents(self, query: str, **kwargs):
        return []


class VectorStoreService:
    """向量存储服务（基于 FAISS）"""

    def __init__(self):
        raw_dir = chroma_conf.get("persist_directory", "db/faiss_index")
        abs_dir = os.path.normpath(get_abs_path(raw_dir))
        os.makedirs(abs_dir, exist_ok=True)
        # FAISS C++ fopen 无法处理含中文的绝对路径（Windows），
        # 使用 relpath 确保传递给 FAISS 的是 ASCII 路径。
        self.persist_dir = os.path.relpath(abs_dir)

        self.vector_store: Optional[FAISS] = None
        try:
            index_file = os.path.join(self.persist_dir, "index.faiss")
            if os.path.exists(index_file):
                self.vector_store = FAISS.load_local(
                    self.persist_dir,
                    embed_model,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"[VectorStore] 加载 FAISS 索引: {self.persist_dir}")
        except Exception as e:
            logger.warning(f"[VectorStore] 加载 FAISS 索引失败（将新建）: {e}")
            self.vector_store = None

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf.get("chunk_size", 200),
            chunk_overlap=chroma_conf.get("chunk_overlap", 20),
            separators=chroma_conf.get("separators", ["\n\n", "\n", ".", "!", "?", " ", ""]),
            length_function=len,
        )

    def get_retriever(self):
        if self.vector_store is None:
            return _EmptyRetriever()
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf.get("k", 3)}
        )

    def get_all_documents(self) -> list[Document]:
        """获取索引中所有文档（用于 BM25 同步等场景）"""
        if self.vector_store is None:
            return []
        return list(self.vector_store.docstore._dict.values())

    def _save(self):
        """持久化 FAISS 索引到磁盘"""
        if self.vector_store is not None:
            self.vector_store.save_local(self.persist_dir)

    def load_document(self):
        """加载知识库文档到向量存储"""
        def check_md5(md5_for_check: str) -> bool:
            md5_path = get_abs_path(chroma_conf.get("md5_hex_store", "md5.text"))
            if not os.path.exists(md5_path):
                open(md5_path, "w", encoding="utf-8").close()
                return False
            with open(md5_path, "r", encoding="utf-8") as f:
                return md5_for_check in [line.strip() for line in f.readlines()]

        def save_md5(md5_for_check: str):
            with open(get_abs_path(chroma_conf.get("md5_hex_store", "md5.text")), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        allowed_files = listdir_with_allowed_type(
            get_abs_path(chroma_conf.get("data_path", "data/knowledge")),
            tuple(chroma_conf.get("allow_knowledge_file_type", ["txt", "pdf"])),
        )

        for path in allowed_files:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                continue
            if check_md5(md5_hex):
                logger.info(f"[VectorStore] 跳过已加载的: {path}")
                continue

            try:
                documents = []
                if path.endswith(".txt"):
                    documents = txt_loader(path)
                elif path.endswith(".pdf"):
                    documents = pdf_loader(path)

                if not documents:
                    continue

                split_docs = self.spliter.split_documents(documents)
                if split_docs:
                    if self.vector_store is None:
                        self.vector_store = FAISS.from_documents(split_docs, embed_model)
                    else:
                        self.vector_store.add_documents(split_docs)
                    self._save()
                    save_md5(md5_hex)
                    logger.info(f"[VectorStore] 加载知识库: {path}")
            except Exception as e:
                logger.error(f"[VectorStore] 加载失败 {path}: {e}")
