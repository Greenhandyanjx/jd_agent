"""
Migration: 从 ChromaDB 迁移数据到 FAISS
=======================================
将已有 ChromaDB 中的数据（RAG + LongTermMemory）迁移到 FAISS 索引。
"""
import os
import sys
import shutil

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from utils.logger import logger
from utils.config import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path


def migrate_rag_collection():
    """迁移 RAG 知识库集合 (collection: agent)"""
    chroma_persist = get_abs_path("db/chroma_db")
    faiss_persist = get_abs_path(chroma_conf.get("persist_directory", "db/faiss_index"))

    if not os.path.exists(os.path.join(chroma_persist, "chroma.sqlite3")):
        logger.info("[迁移] 未找到 ChromaDB 数据，跳过 RAG 集合迁移")
        return False

    logger.info(f"[迁移] 从 ChromaDB 读取 RAG 集合...")
    chroma_store = Chroma(
        collection_name="agent",
        embedding_function=embed_model,
        persist_directory=chroma_persist,
    )
    all_data = chroma_store.get()

    if not all_data or not all_data.get("documents"):
        logger.info("[迁移] ChromaDB RAG 集合为空，跳过")
        return True

    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(all_data["documents"], all_data.get("metadatas", []))
    ]
    logger.info(f"[迁移] 读取到 {len(docs)} 篇文档，写入 FAISS...")

    os.makedirs(faiss_persist, exist_ok=True)
    faiss_store = FAISS.from_documents(docs, embed_model)
    faiss_store.save_local(faiss_persist)

    logger.info(f"[迁移] RAG 集合迁移完成 → {faiss_persist}")
    return True


def migrate_long_term_memory():
    """迁移长期记忆集合 (collection: long_term_memory)"""
    chroma_persist = get_abs_path("db/chroma_db")
    base_faiss = get_abs_path(chroma_conf.get("persist_directory", "db/faiss_index"))
    faiss_persist = os.path.join(base_faiss, "long_term_memory")

    if not os.path.exists(os.path.join(chroma_persist, "chroma.sqlite3")):
        logger.info("[迁移] 未找到 ChromaDB 数据，跳过长期记忆迁移")
        return False

    logger.info(f"[迁移] 从 ChromaDB 读取长期记忆集合...")
    chroma_store = Chroma(
        collection_name="long_term_memory",
        embedding_function=embed_model,
        persist_directory=chroma_persist,
    )
    all_data = chroma_store.get()

    if not all_data or not all_data.get("documents"):
        logger.info("[迁移] ChromaDB 长期记忆为空，跳过")
        return True

    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(all_data["documents"], all_data.get("metadatas", []))
    ]
    logger.info(f"[迁移] 读取到 {len(docs)} 条记忆，写入 FAISS...")

    os.makedirs(faiss_persist, exist_ok=True)
    faiss_store = FAISS.from_documents(docs, embed_model)
    faiss_store.save_local(faiss_persist)

    logger.info(f"[迁移] 长期记忆迁移完成 → {faiss_persist}")
    return True


def clean_chromadb():
    """可选：清理旧的 ChromaDB 数据"""
    chroma_persist = get_abs_path("db/chroma_db")
    if os.path.exists(chroma_persist):
        logger.info(f"[迁移] 删除旧 ChromaDB 数据: {chroma_persist}")
        shutil.rmtree(chroma_persist)


if __name__ == "__main__":
    print("=" * 50)
    print("  开始从 ChromaDB → FAISS 数据迁移")
    print("=" * 50)

    rag_ok = migrate_rag_collection()
    mem_ok = migrate_long_term_memory()

    if rag_ok and mem_ok:
        print("\n✅ 迁移完成！")
        print("   旧 ChromaDB 数据保留在 db/chroma_db/")
        print("   如需清理，运行: python scripts/migrate_to_faiss.py --clean")
        print("\n   然后可以删除依赖:")
        print("   pip uninstall chromadb langchain-chroma")
        print("   pip install faiss-cpu")
    else:
        print("\n⚠️  迁移部分完成或无数据可迁移")

    if "--clean" in sys.argv:
        clean_chromadb()
        print("✅ 旧 ChromaDB 数据已清理")
