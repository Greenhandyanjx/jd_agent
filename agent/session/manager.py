"""
Agent Core: Session 会话管理
参考 nanobot 的 session/manager.py 设计

会话管理负责：
1. 持久化会话消息到磁盘（JSONL 格式）
2. 按 session_key 分会话（每个 channel:chat_id 一个独立会话）
3. 消息对齐保护（防止 orphan tool results）
4. 会话压缩（超出上限时丢弃旧消息）

JSONL 格式好处：
- 追加写，高性能
- 每行一条独立 JSON，原子性好
- 无需数据库依赖
"""

import json
import os
import shutil
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """将字符串转为安全的文件名（替换非法字符）"""
    safe = ""
    for c in name:
        if c.isalnum() or c in ('-', '_', '.'):
            safe += c
        else:
            safe += '_'
    return safe


def find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    """
    找到第一个合法的消息起始位置。
    
    如果开头有孤立的 tool 消息（没有对应的 assistant tool_call），
    这些消息会被跳过，因为大多数 Provider 不接受这样的消息序列。
    """
    declared: set[str] = set()
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    declared.add(str(tc["id"]))
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid and str(tid) not in declared:
                # 存在孤儿 tool 消息，跳过直到找到合法的起始位置
                start = i + 1
                declared.clear()
                for prev in messages[start:i + 1]:
                    if prev.get("role") == "assistant":
                        for tc in prev.get("tool_calls") or []:
                            if isinstance(tc, dict) and tc.get("id"):
                                declared.add(str(tc["id"]))
    return 0  # 没有检测到孤儿消息


@dataclass
class Session:
    """
    会话：存储对话历史。
    
    关键设计：
    - messages 是追加写的（有利于 LLM cache 效率）
    - last_consolidated 标记已压缩到文件的消息进度
    - get_history() 只返回未 consolidated 的消息
    """
    key: str                                     # 会话标识（channel:chat_id）
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0                   # 已 persistent 的消息计数

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """添加一条消息到会话"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """
        获取 LLM 输入用的消息列表。
        
        只返回未 consolidated 的消息，对齐到合法的 tool-call 边界。
        """
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # 避免以非 user 消息开头
        for i, msg in enumerate(sliced):
            if msg.get("role") == "user":
                sliced = sliced[i:]
                break

        # 移除开头的孤儿 tool 消息
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        # 确保 content 字段不会是 None（旧数据兼容 + 防止 LLM 将 null 渲染为 "None"）
        for msg in sliced:
            if msg.get("content") is None:
                msg["content"] = ""

        return sliced

    def clear(self) -> None:
        """清空会话"""
        self.messages.clear()
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    """
    会话管理器：CRUD + JSONL 持久化。
    
    每个会话存为一个 JSONL 文件：sessions/{safe_key}.jsonl
    格式：
    - 第一行：元数据（_type: "metadata"）
    - 后续行：消息 JSON
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self._cache: dict[str, Session] = {}  # 内存缓存，避免反复读盘

    def get_or_create(self, key: str) -> Session:
        """获取或创建会话"""
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        """将会话持久化到磁盘（JSONL）"""
        path = self._get_session_path(session.key)
        tmp_path = path.with_suffix(".jsonl.tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # 写元数据行
                metadata_line = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "last_consolidated": session.last_consolidated,
                }
                f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
                # 写消息行
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            # 原子替换（先写 .tmp 再 rename，防止写一半崩溃）
            os.replace(tmp_path, path)
        except BaseException:
            # 清理临时文件
            tmp_path.unlink(missing_ok=True)
            raise

        self._cache[session.key] = session

    def delete_session(self, key: str) -> bool:
        """删除会话"""
        self._cache.pop(key, None)
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def _get_session_path(self, key: str) -> Path:
        """获取会话文件路径"""
        return self.sessions_dir / f"{safe_filename(key.replace(':', '_'))}.jsonl"

    def _load(self, key: str) -> Session | None:
        """从磁盘加载会话"""
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            updated_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        if data.get("created_at"):
                            created_at = datetime.fromisoformat(data["created_at"])
                        if data.get("updated_at"):
                            updated_at = datetime.fromisoformat(data["updated_at"])
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception as e:
            logger.warning(f"加载会话 {key} 失败: {e}")
            return self._repair(key)

    def _repair(self, key: str) -> Session | None:
        """尝试从损坏的文件中恢复会话"""
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            updated_at = None
            last_consolidated = 0
            skipped = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        if data.get("created_at"):
                            with suppress(ValueError):
                                created_at = datetime.fromisoformat(data["created_at"])
                        if data.get("updated_at"):
                            with suppress(ValueError):
                                updated_at = datetime.fromisoformat(data["updated_at"])
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            if skipped:
                logger.warning(f"跳过 {skipped} 行损坏数据")

            if not messages and not metadata:
                return None

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception:
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话"""
        sessions = []
        for path in self.sessions_dir.glob("*.jsonl"):
            key = path.stem.replace("_", ":", 1)
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append({
                                "key": data.get("key") or key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                            })
            except Exception:
                continue
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
