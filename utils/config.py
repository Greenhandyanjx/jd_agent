"""
配置加载：读取 YAML 配置文件
（迁移自原始项目的 config_handler.py，适配新路径）
"""
import os
import yaml
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str = None, encoding: str = "utf-8"):
    path = config_path or get_abs_path("config/rag.yml")
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str = None, encoding: str = "utf-8"):
    # 优先读取 vector_store.yml（新名称），不存在则回退 chroma.yml
    path = config_path or get_abs_path("config/vector_store.yml")
    if not os.path.exists(path):
        path = config_path or get_abs_path("config/chroma.yml")
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str = None, encoding: str = "utf-8"):
    path = config_path or get_abs_path("config/prompts.yml")
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = None, encoding: str = "utf-8"):
    path = config_path or get_abs_path("config/agent.yml")
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
