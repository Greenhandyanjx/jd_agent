"""
初始化脚本：加载知识库到向量数据库
"""
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.retrieval.vector_store import VectorStoreService
from utils.logger import logger


def init_knowledge_base():
    """初始化知识库"""
    logger.info("[Init] 开始初始化知识库...")
    vs = VectorStoreService()
    vs.load_document()
    logger.info("[Init] 知识库初始化完成")


def init_environment():
    """初始化环境"""
    # 创建必要的目录
    dirs = ["db/chroma_db", "logs", "data/external"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("[Init] 目录初始化完成")


if __name__ == "__main__":
    print("=" * 40)
    print("JD-Agent 初始化工具")
    print("=" * 40)

    init_environment()
    init_knowledge_base()

    print("✅ 初始化完成！")
    print("运行方式:")
    print("  API服务:    python -m api.app")
    print("  Streamlit UI: streamlit run main.py")
