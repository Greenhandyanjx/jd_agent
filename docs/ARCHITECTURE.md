# JD-Agent 系统架构文档

> 一个参考 **nanobot**（OpenClaw 的轻量级 Python 替代品）架构的大模型应用智能体框架
>
> nanobot 是 OpenClaw 的"平替"——用 99% 更少的代码实现了相同的核心 Agent 功能

---

## 一、设计理念

### 为什么从 OpenClaw 转向 nanobot？

| 对比 | OpenClaw | nanobot → JD-Agent |
|------|----------|-------------------|
| **语言** | TypeScript | Python |
| **代码量** | 60+ 模块，大量重抽象 | ~10 个文件，清晰直接 |
| **架构复杂度** | 多层抽象（gateway/plugin/harness/routing） | 总线 + 循环 + Provider 三位一体 |
| **学习曲线** | 需要深入理解整个 gateway 插件体系 | 几小时内可读完全部代码 |

### 核心理念：MessageBus 解耦一切

```
  Channel (CLI / Streamlit / API)
        │
        ▼  publish_inbound()
  ┌─────────────────┐
  │   MessageBus    │  ← 异步 asyncio.Queue
  └─────────────────┘
        │  consume_inbound()
        ▼
  ┌─────────────────┐
  │   AgentLoop     │  ← ReAct 循环（Think→Act→Observe）
  └─────────────────┘
        │  publish_outbound()
        ▼
  ┌─────────────────┐
  │   MessageBus    │
  └─────────────────┘
        │  consume_outbound()
        ▼
  OutboundMessage → Channel
```

关键：通道只管 publish / consume 消息，Agent 核心不关心消息来自于 Streamlit 还是 API。

---

## 二、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JD-Agent v2 架构                              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      AgentOrchestrator                       │   │
│  │  ┌────────────┐  ┌───────────┐  ┌────────────┐             │   │
│  │  │ MessageBus │  │ AgentLoop │  │ Provider   │             │   │
│  │  │ (异步队列)  │  │ (ReAct)    │  │ (LLM抽象)  │             │   │
│  │  └────────────┘  └───────────┘  └────────────┘             │   │
│  │                     │          │                            │   │
│  │  ┌──────────────────┴──────────┴────────────────────────┐   │   │
│  │  │                 ToolRegistry                         │   │   │
│  │  │   register() → get_definitions() → execute()        │   │   │
│  │  └──────────────────────┬───────────────────────────────┘   │   │
│  │                         │                                    │   │
│  │  ┌──────────────────────┴───────────────────────────────┐   │   │
│  │  │              SessionManager + MemoryStore             │   │   │
│  │  │   JSONL 持久化 · MEMORY.md · HISTORY.md · 向量检索    │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、Agent Core：ReAct 循环（仿 nanobot）

### 核心流程

```
                     User Input
                         │
                         ▼
             ┌─────────────────────┐
             │  ContextBuilder     │
             │  构建 System Prompt │
             │  + 历史 + 当前消息   │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  ReAct Loop         │
             │  (最大 N 轮)         │
             │                     │
    ┌────────┴────────┐            │
    │   Call LLM      │            │
    │  (带重试退避)    │            │
    └────────┬────────┘            │
             │                     │
      ┌──────┴──────┐              │
      ▼              ▼              │
  ┌─────────┐  ┌──────────┐        │
  │ Tool Call│  │Text Reply│        │
  │ (Act)    │  │ (Final)  │──→Done│
  └────┬────┘  └──────────┘        │
       │                           │
       ▼                           │
  ┌─────────┐                      │
  │ Execute │   ┌──────────┐       │
  │ Tool(s) │──▶│ Observe  │──→Loop│
  │(并发执行)│   │ (注入结果)│       │
  └─────────┘   └──────────┘       │
             └─────────────────────┘
```

### Key Design（来自 nanobot）

| 特性 | 说明 | 对应代码 |
|------|------|---------|
| **异步** | 全程 asyncio，跨 session 可并发 | `agent/loop.py` |
| **重试** | 指数退避 + Retry-After 解析 | `agent/core/llm_provider.py` |
| **Streaming** | 流式输出 + tool call 累加 | `providers/*.py` |
| **并行工具** | `asyncio.gather()` 并发执行同轮工具调用 | `agent/loop.py` |
| **会话隔离** | per-session Lock，同 session 串行 | `agent/loop.py` |
| **并发门控** | 全局 Semaphore 控制并发数 | `agent/loop.py` |

---

## 四、消息总线（MessageBus）— 参考 nanobot

```
Agent ←─────────────────────────────────────── 通道
       │   subscribe                            │
       ▼                                        │
  ┌─────────────────────────┐                   │
  │    MessageBus           │                   │
  │                         │                   │
  │  Inbound Queue ──────── AgentLoop           │
  │  Outbound Queue ─────── 通道                │
  │                         │                   │
  └─────────────────────────┘                   │
                                                │
  入站: InboundMessage(channel, sender, content)
  出站: OutboundMessage(channel, chat_id, content)
```

参考 nanobot 的 `bus/queue.py` 和 `bus/events.py`，用两个 `asyncio.Queue` 实现通道与 Agent 的解耦。

---

## 五、LLM Provider 抽象 — 参考 nanobot

```
                 LLMProvider (抽象基类)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    OpenAIProvider          TongyiProvider
    (DeepSeek/GPT/...)      (通义千问)
          │                       │
    统一的 chat() / chat_stream() 接口
          │
          ▼
    LLMResponse(content, tool_calls, finish_reason, usage)
```

核心设计（来自 nanobot `providers/base.py`）：
- `chat_with_retry()` / `chat_stream_with_retry()` 自动处理重试
- 瞬态错误自动识别（429/500/502/503/504/timeout）
- 指数退避 + Retry-After 头部解析
- 统一的 Tool Call 解析层

---

## 六、Tool 系统 — 参考 nanobot

```
                 Tool (抽象基类)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     ReadFileTool            WebFetchTool
     WriteFileTool           WeatherTool
                              RAGQueryTool
                              ...
                      │
                      ▼
                ToolRegistry
          register() → get_definitions() → execute()
```

参考 nanobot 的 `agent/tools/base.py` 和 `agent/tools/registry.py`：
- 每个工具继承 `Tool` 基类，实现 `name` / `description` / `parameters` / `execute()`
- `to_schema()` 转为 OpenAI Function Calling 格式
- `register_all_tools()` 统一注册入口

### 内置工具

| 工具名 | 来源 | 功能 |
|--------|------|------|
| `read_file` | nanobot | 读取文件内容 |
| `write_file` | nanobot | 写入/创建文件 |
| `web_fetch` | nanobot | 抓取网页内容 |
| `rag_query` | 原有 | RAG 检索增强查询 |
| `get_weather` | 原有迁移 | 天气查询（Mock） |
| `get_user_location` | 原有迁移 | 用户位置获取 |
| `get_user_id` | 原有迁移 | 用户ID获取 |
| `get_current_month` | 原有迁移 | 当前月份 |
| `query_order` | 原有迁移 | 外卖订单查询 |
| `recommend_dish` | 原有迁移 | 菜品推荐 |
| `check_delivery_status` | 原有迁移 | 配送状态查询 |

---

## 七、记忆系统（三层 + LLM Consolidation）— 参考 nanobot

```
┌─────────────────────────────────────────────────────────────────┐
│                         Memory Layer                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Layer 1: Session Messages (JSONL) — 短期记忆            │   │
│  │   - sessions/{key}.jsonl（每会话一个文件）               │   │
│  │   - 追加写，首次写入时创建，原子替换                     │   │
│  │   - 断裂恢复：自动检测 orphan tool 消息并跳过           │   │
│  │   - SessionManager.get_or_create() 懒加载                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Layer 2: HISTORY.md — 可 grep 搜索的行为日志            │   │
│  │   - 由 MemoryConsolidator 自动写入                      │   │
│  │   - 格式：[YYYY-MM-DD HH:MM] 关键事件摘要               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Layer 3: MEMORY.md — 长期事实记忆（nanobot 模式）       │   │
│  │   - 跨会话持久：用户偏好、关键事实、重要结论             │   │
│  │   - 通过 LLM tool call 自动写入（不是向量数据库）        │   │
│  │   - 参考 nanobot 的 memory.py 设计                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  保留：ChromaDB 向量检索长期记忆（long_term.py）         │   │
│  │   - 用于 RAG 场景的语义检索                             │   │
│  │   - 与 nanobot 的 MEMORY.md 模式互补                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### MemoryConsolidator（来自 nanobot 的核心创新）

```
每 N 条消息 / 每 T 分钟
        │
        ▼
  收集未 consolidated 的对话
        │
        ▼
  调用 LLM 分析关键信息
        │
        ▼
  同时写入 HISTORY.md + MEMORY.md
        │
        ▼
  更新 session.last_consolidated
```

参考 nanobot 的 `agent/memory.py`。让 LLM 自己决定应该记住什么——而不是硬编码记忆规则。

---

## 八、Context Builder — 参考 nanobot

```
ContextBuilder
  │
  ├─ _get_identity()        → 核心身份 + 运行时信息
  ├─ _load_bootstrap_content() → AGENTS.md / SOUL.md / USER.md / TOOLS.md
  ├─ memory.get_memory_context() → MEMORY.md 长期记忆
  │
  └─ build_messages()       → 完整消息列表：system + 历史 + 当前
```

参考 nanobot 的 `agent/context.py`，System Prompt 从 workspace 文件中读取。

---

## 九、Session 管理 — 参考 nanobot

```
SessionManager
  │
  ├─ get_or_create(key) → 内存缓存 / 懒加载 JSONL
  ├─ save(session)      → 原子写（先 .tmp 再 rename）
  └─ delete_session()   → 删除文件 + 清除缓存

Session
  ├─ key: "channel:chat_id"
  ├─ messages: [{role, content, tool_calls, ...}]
  ├─ last_consolidated: N  ← 记忆 consolidation 进度
  └─ get_history() → 返回对齐到合法 tool-call 边界的消息
```

参考 nanobot 的 `session/manager.py`。JSONL 格式的好处：
- 追加写（高性能）
- 每行一条独立 JSON（原子性好）
- 不需要数据库依赖

---

## 十、目录结构（完整版）

```
jd_agent/
├── agent/
│   ├── __init__.py              # 统一导出
│   ├── loop.py                  # [新] ReAct 循环核心（仿 nanobot agent/loop.py）
│   ├── orchestrator.py          # [改] 编排器（集成 MessageBus + AgentLoop + Provider）
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py             # [新] 核心数据类型（LLMResponse, ToolCallRequest, InboundMessage...）
│   │   ├── llm_provider.py      # [新] LLM Provider 抽象基类（仿 nanobot providers/base.py）
│   │   ├── react_loop.py        # [保留] 原有 ReAct 循环（兼容旧代码）
│   │   ├── function_calling.py  # [保留] Function Call 解析器
│   │   ├── tool_registry.py     # [保留] 装饰器式工具注册（旧接口）
│   │   ├── schema_validator.py  # [保留] 参数校验器
│   │   └── fallback.py          # [保留] 错误回退策略
│   │
│   ├── bus/
│   │   ├── __init__.py          # [新]
│   │   └── queue.py             # [新] 消息总线（仿 nanobot bus/queue.py）
│   │
│   ├── providers/
│   │   ├── __init__.py          # [新]
│   │   ├── openai_provider.py   # [新] OpenAI/DeepSeek Provider
│   │   └── tongyi_provider.py   # [新] 通义千问 Provider
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py              # [新] 工具基类（仿 nanobot agent/tools/base.py）
│   │   ├── registry.py          # [新] 工具注册中心（仿 nanobot agent/tools/registry.py）
│   │   ├── tool_definitions.py  # [改] 工具定义（改用新 Tool 基类）
│   │   ├── filesystem.py        # [新] 文件系统工具（read/write_file）
│   │   ├── web.py               # [新] Web 工具（web_fetch）
│   │   └── middleware.py        # [保留] 工具调用中间件
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── memory_store.py      # [新] MEMORY.md + HISTORY.md + MemoryConsolidator
│   │   ├── context_builder.py   # [新] System Prompt 构建器
│   │   ├── short_term.py        # [保留] 短期记忆（兼容层）
│   │   ├── working_memory.py    # [保留] 工作记忆
│   │   └── long_term.py         # [保留] 向量检索长期记忆
│   │
│   ├── session/
│   │   ├── __init__.py          # [新]
│   │   └── manager.py           # [新] JSONL 会话管理（仿 nanobot session/manager.py）
│   │
│   ├── planner/                 # [保留] 任务规划器
│   │   └── ...
│   │
│   └── middleware/              # [保留] 中间件
│       └── ...
│
├── model/                       # [保留] 模型工厂
├── api/
├── config/
├── prompts/
├── data/
├── db/
├── utils/
├── tests/
├── docs/
│   ├── ARCHITECTURE.md          # [改] 本文档
│   └── NANOBOT_REFERENCE.md     # [新] nanobot 参考指南
├── main.py                      # [改] Streamlit UI（适配异步 orchestrator）
└── requirements.txt
```

### 文件说明（[新] vs [改] vs [保留]）

| 文件 | 状态 | 说明 |
|------|------|------|
| `agent/loop.py` | **新增** | ReAct 循环核心，仿 nanobot `agent/loop.py` |
| `agent/core/types.py` | **新增** | LLMResponse, ToolCallRequest, Inbound/OutboundMessage 等 DTO |
| `agent/core/llm_provider.py` | **新增** | LLM Provider 抽象基类 + 重试退避机制 |
| `agent/providers/openai_provider.py` | **新增** | OpenAI/DeepSeek 兼容 Provider |
| `agent/providers/tongyi_provider.py` | **新增** | 通义千问 Provider |
| `agent/bus/queue.py` | **新增** | asyncio.Queue 消息总线 |
| `agent/session/manager.py` | **新增** | JSONL 会话持久化 |
| `agent/tools/base.py` | **新增** | Tool 抽象基类 |
| `agent/tools/registry.py` | **新增** | ToolRegistry 注册中心 |
| `agent/tools/filesystem.py` | **新增** | read_file/write_file 工具 |
| `agent/tools/web.py` | **新增** | web_fetch 工具 |
| `agent/memory/memory_store.py` | **新增** | MEMORY.md + HISTORY.md 内存管理 |
| `agent/memory/context_builder.py` | **新增** | 从 workspace 文件构建 System Prompt |
| `agent/orchestrator.py` | **重写** | 集成新的 AgentLoop + MessageBus |
| `agent/tools/tool_definitions.py` | **重写** | 改用新的 Tool 基类 |
| `agent/__init__.py` | **重写** | 统一导出 |
| `main.py` | **重写** | 适配异步 orchestrator + LLMProvider |
| `agent/core/react_loop.py` | **保留** | 原有 ReAct 循环（兼容旧代码） |
| `agent/core/function_calling.py` | **保留** | Function Call 解析器 |
| `agent/core/tool_registry.py` | **保留** | 装饰器式工具注册 |
| `agent/core/schema_validator.py` | **保留** | 参数校验器 |
| `agent/core/fallback.py` | **保留** | 错误回退策略 |
| `agent/memory/short_term.py` | **保留** | 兼容层 |
| `agent/memory/working_memory.py` | **保留** | 工作记忆 |
| `agent/memory/long_term.py` | **保留** | ChromaDB 向量检索 |

---

## 十一、与外卖系统集成（保留不变）

```
JD-Agent ──HTTP──▶ SYSU Campus Food Delivery (Go/Gin Backend)
  │                      │
  tools:                  └── MySQL DB
  query_order
  recommend_dish
  check_delivery_status
```

工具内部封装外卖系统的 API 调用逻辑。

---

## 十二、部署架构

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Streamlit    │────▶│ AgentOrchestrat│────▶│   LLM API     │
│   UI / API     │     │   (FastAPI)   │     │  (DeepSeek/通义)│
└───────────────┘     └───────┬───────┘     └───────────────┘
                              │
                ┌─────────────┴─────────────┐
                │           磁盘文件           │
                │  sessions/{key}.jsonl      │
                │  memory/MEMORY.md          │
                │  memory/HISTORY.md         │
                │  db/chroma_db/ (向量检索)   │
                └───────────────────────────┘
```

---

## 十三、参考：nanobot ↔ JD-Agent 映射表

| nanobot 模块 | JD-Agent 对应 | 功能 |
|-------------|--------------|------|
| `agent/loop.py` | `agent/loop.py` | ReAct 循环 |
| `providers/base.py` | `agent/core/llm_provider.py` | Provider 抽象 |
| `bus/queue.py` | `agent/bus/queue.py` | 消息队列 |
| `bus/events.py` | `agent/core/types.py` | 消息类型 |
| `agent/context.py` | `agent/memory/context_builder.py` | System Prompt |
| `agent/memory.py` | `agent/memory/memory_store.py` | 记忆管理 |
| `agent/tools/base.py` | `agent/tools/base.py` | 工具基类 |
| `agent/tools/registry.py` | `agent/tools/registry.py` | 工具注册 |
| `session/manager.py` | `agent/session/manager.py` | 会话持久化 |
| `config/schema.py` | `agent/core/types.py` + config/ | 配置 |

---

> **设计理念**: 参考 nanobot（OpenClaw 的 Python 轻量替代）的 MessageBus + Provider + ReAct Loop 三位一体架构，结合原有 jd_agent 的工具系统和 RAG 能力，构建一个"自己能说清楚原理、面试能展示亮点"的工程化 Agent 框架。
