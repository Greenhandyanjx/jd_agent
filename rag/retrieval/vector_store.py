"""
RAG Retrieval: 向量存储服务（迁移自原始项目）
"""
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger import logger
import os


class VectorStoreService:
    """向量存储服务"""

    def __init__(self):
        persist_dir = get_abs_path(chroma_conf.get("persist_directory", "db/chroma_db"))
        os.makedirs(persist_dir, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=chroma_conf.get("collection_name", "agent"),
            embedding_function=embed_model,
            persist_directory=persist_dir,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf.get("chunk_size", 200),
            chunk_overlap=chroma_conf.get("chunk_overlap", 20),
            separators=chroma_conf.get("separators", ["\n\n", "\n", ".", "!", "?", " ", ""]),
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf.get("k", 3)}
        )

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
                    self.vector_store.add_documents(split_docs)
                    save_md5(md5_hex)
                    logger.info(f"[VectorStore] 加载知识库: {path}")
            except Exception as e:
                logger.error(f"[VectorStore] 加载失败 {path}: {e}")
