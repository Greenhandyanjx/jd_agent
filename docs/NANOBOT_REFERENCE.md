# Nanobot 参考指南

> nanobot — OpenClaw 的"平替"：Python 实现的超轻量 Agent 框架
> 
> 源码位置: `D:\桌面\深度学习\nanobot-main\`
> README 说："inspired by OpenClaw — delivers core agent functionality with 99% fewer lines of code"

---

## 为什么参考 nanobot 而不是 OpenClaw

| 对比 | OpenClaw (TypeScript) | nanobot (Python) |
|------|----------------------|-------------------|
| 代码量 | 60+ 模块，大量重抽象 | ~10 个 Python 文件 |
| 架构 | Gateway + Plugin + Harness + Routing | MessageBus + Provider + Loop |
| 学习成本 | 需要理解整个插件体系 | 几小时读完所有代码 |
| 与 jd_agent 契合度 | 语言不同，难以复用 | Python，可直接参考 |

---

## nanobot 核心文件导读

| 文件 | 行数 | 核心作用 | JD-Agent 对应 |
|------|------|---------|--------------|
| `agent/loop.py` | 555 | ReAct 循环（boot → call_llm → exec → inject → loop） | `agent/loop.py` |
| `agent/context.py` | ~200 | 构建 System Prompt | `agent/memory/context_builder.py` |
| `agent/memory.py` | ~370 | MEMORY.md + HISTORY.md + Consolidation | `agent/memory/memory_store.py` |
| `agent/skills.py` | — | 技能加载 | 预留 |
| `agent/subagent.py` | ~240 | 后台子 Agent | 预留 |
| `bus/queue.py` | ~50 | asyncio.Queue 消息队列 | `agent/bus/queue.py` |
| `bus/events.py` | ~50 | InboundMessage / OutboundMessage | `agent/core/types.py` |
| `providers/base.py` | ~370 | LLMProvider 抽象基类 | `agent/core/llm_provider.py` |
| `session/manager.py` | ~270 | JSONL 会话管理 | `agent/session/manager.py` |
| `config/schema.py` | ~80 | Pydantic 配置 | 预留 |
| `agent/tools/base.py` | ~190 | Tool 基类 | `agent/tools/base.py` |
| `agent/tools/registry.py` | ~80 | ToolRegistry | `agent/tools/registry.py` |

---

## 核心模式：MessageBus

nanobot 最核心的解耦模式：

```python
# 通道侧
msg = InboundMessage(channel="streamlit", sender_id="user", chat_id="default", content="你好")
await bus.publish_inbound(msg)

# Agent 侧
msg = await bus.consume_inbound()
response = await process(msg)
await bus.publish_outbound(response)

# 通道侧
response = await bus.consume_outbound()
display(response.content)
```

## 核心模式：ReAct Loop

nanobot 的循环结构（约 150 行核心逻辑）：

```python
async def run(self):
    while self._running:                        # 1. 主循环
        msg = await bus.consume_inbound()       # 2. 消费入站消息
        task = create_task(_dispatch(msg))      # 3. 每个 session 一个任务

async def _dispatch(msg):
    lock = _session_locks[msg.session_key]
    gate = concurrency_gate
    async with lock, gate:                      # 4. 同 session 串行 + 全局并发控制
        return await _process_message(msg)       # 5. 处理

async def _process_message(msg):
    session = sessions.get_or_create(msg.session_key)   # 6. 获取/创建会话
    messages = context.build_messages(history, msg)     # 7. 构建消息列表
    content, tools, all_messages = await _react_loop(initial_messages)  # 8. ReAct
    save_turn(session, all_messages)                     # 9. 持久化
    maybe_consolidate(session)                           # 10. 记忆 consolidation
    return OutboundMessage(content=content)
```

## 核心模式：ReAct 内部循环

```python
async def _react_loop(initial_messages):
    messages = initial_messages
    iteration = 0
    while iteration < max_iterations:
        response = await llm.chat(messages, tools)       # Think → Act
        if response.tool_calls:
            results = await gather(*(
                registry.execute(tc.name, tc.args)
                for tc in response.tool_calls
            ))                                             # Act（并行执行）
            for tc, result in zip(tool_calls, results):
                messages.append(tool_result_msg(tc, result))  # Observe
        else:
            return response.content                        # Final Answer
```

## 核心模式：记忆 Consolidation

```python
class MemoryConsolidator:
    async def maybe_consolidate(self, session):
        if new_messages >= consolidate_every_n or time_since_last >= 15min:
            # 用 LLM 分析对话
            response = await llm.chat([
                {"role": "system", "content": "请分析对话并返回JSON..."},
                {"role": "user", "content": "当前MEMORY.md: ...\n近期对话: ..."}
            ])
            # 解析 JSON：{history_entry, memory_update}
            memory_store.append_history(result["history_entry"])
            memory_store.write_long_term(result["memory_update"])
            session.last_consolidated = len(session.messages)
```

## 核心模式：LLM Provider 抽象

```python
class LLMProvider(ABC):
    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_ERROR_MARKERS = ("429", "rate limit", "500", "502", "timeout")

    @abstractmethod
    async def chat(self, messages, tools=None, ...) -> LLMResponse:
        pass                                    # 每个 Provider 实现

    async def chat_with_retry(self, ...):
        for attempt in range(3):
            response = await self._safe_chat(...)
            if response.finish_reason != "error":
                return response                 # 成功
            if not self._is_transient(response):
                return response                 # 非瞬态错误不重试
            await asyncio.sleep(delay)          # 指数退避
```

---

## 从 nanobot 迁移的关键决策

1. **MessageBus 替代原有的直接调用** — 解耦通道与 Agent 核心
2. **异步 asyncio 替代同步** — 跨 session 并发，同 session 串行
3. **JSONL 会话文件替代内存字典** — 进程重启后恢复对话
4. **Provider 抽象替代 LangChain BaseChatModel 直接调用** — 切换模型更灵活
5. **Tool 基类替代装饰器风格** — 更清晰的结构，OOP 方式
6. **MEMORY.md 替代向量数据库作为主要长期记忆** — 更简单有效
7. **LLM Consolidation 替代硬编码记忆规则** — 让 AI 自己决定记什么
