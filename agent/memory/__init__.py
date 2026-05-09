"""三层记忆系统

参考 nanobot 的 agent/memory.py 设计：

层级           | 实现                          | 功能
---------------|-------------------------------|---------------------------
System Memory  | memory_store.py (MEMORY.md)   | 长期事实记忆（跨会话持久化）
History        | memory_store.py (HISTORY.md)  | 可 grep 搜索的行为日志
Session        | session/manager.py (JSONL)    | 当前会话上下文（短期记忆）

新增组件：
- ContextBuilder: 构建 System Prompt（读取 AGENTS.md, SOUL.md, USER.md, TOOLS.md）
- MemoryConsolidator: 定时/定量自动 consolidation

保留原有：
- LongTermMemory: ChromaDB 向量检索长期记忆
- WorkingMemory: 任务执行状态追踪
- ShortTermMemory: 简化后作为 Session 的兼容封装
"""
