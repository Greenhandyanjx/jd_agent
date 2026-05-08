# JD-Agent 🤖

> **自实现工程化 AI Agent 框架** | 参考 OpenClaw 架构 + Claude Code 任务规划机制

一个功能完整的 AI Agent 框架，包含自实现的 ReAct 循环、三层记忆系统、RAG 检索增强、任务规划器、工具调用验证等核心模块。

---

## ✨ 核心特性

| 模块 | 功能 | 亮点 |
|------|------|------|
| **Agent Core** | 自实现 ReAct 循环 | 不依赖 LangChain Agent，Think→Act→Observe 全自控 |
| **Memory** | 三层记忆架构 | 短期记忆（会话）/ 工作记忆（任务）/ 长期记忆（跨会话） |
| **RAG** | 检索增强生成 | 查询改写 + 多路召回 + 重排序 |
| **Task Planner** | 复杂任务分解 | 借鉴 Claude Code 的 Plan-Solve 模式 |
| **Tool System** | 工具注册与调度 | Schema 校验 + 自动重试 + 指数退避 |
| **API** | FastAPI + Streamlit | 双模式：REST API 或 Web 聊天界面 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         JD-Agent                                │
│                                                                  │
│  用户输入 → [API/Streamlit] → [Orchestrator]                    │
│                                  │                              │
│        ┌─────────────────────────┼──────────────────┐          │
│        ▼                         ▼                  ▼           │
│  ┌──────────┐            ┌──────────────┐  ┌─────────────┐     │
│  │ ReAct    │            │ Task Planner │  │ RAG System  │     │
│  │ Loop     │            │ (任务分解)    │  │ (检索增强)   │     │
│  └────┬─────┘            └──────┬───────┘  └──────┬──────┘     │
│       │                        │                  │            │
│  ┌────▼─────┐            ┌──────▼───────┐  ┌──────▼──────┐    │
│  │ Tool     │            │  Dependency  │  │ Query       │    │
│  │ System   │            │  Graph       │  │ Rewrite     │    │
│  └────┬─────┘            └──────────────┘  └──────┬──────┘    │
│       │                                           │            │
│  ┌────▼─────┐                              ┌──────▼──────┐    │
│  │ Memory   │                              │ Reranker    │    │
│  │ 3 Layers │                              │ + Retrieval │    │
│  └──────────┘                              └─────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

详细架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🚀 快速开始

### 前置条件

- Python 3.10+
- 通义千问 API Key（DashScope）

### 安装

```bash
# 1. 克隆项目
cd D:\桌面\深度学习\jd_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
set DASHSCOPE_API_KEY=你的API_KEY

# 4. 初始化知识库
python scripts/init_db.py
```

### 运行

**方式一：Streamlit UI（推荐）**

```bash
streamlit run main.py
```

**方式二：FastAPI 服务**

```bash
python -m api.app
```

API 文档自动生成在: http://localhost:8000/docs

### 测试

```bash
python -m pytest tests/
```

---

## 📁 项目结构

```
jd_agent/
├── agent/                     # Agent 核心
│   ├── core/                  # ReAct 循环 / Function Calling / 工具注册 / Schema校验
│   ├── memory/                # 短期记忆 / 工作记忆 / 长期记忆
│   ├── tools/                 # 工具定义（RAG查询/天气/外卖等10+工具）
│   ├── planner/               # 任务规划器 + 依赖图执行引擎
│   ├── middleware/             # 鉴权 / 限流 / 日志中间件
│   └── orchestrator.py        # Agent 编排器（统一入口）
├── rag/                       # RAG 系统
│   ├── retrieval/             # 向量检索 / BM25 / 多路召回
│   ├── optimizer/             # 查询改写
│   ├── reranker/              # 重排序
│   └── rag_service.py         # RAG 服务入口
├── api/                       # REST API 层
│   ├── app.py                 # FastAPI 主应用
│   ├── routes.py              # 路由定义
│   └── schemas.py             # 请求/响应 Schema
├── model/                     # 模型工厂（通义千问封装）
├── utils/                     # 工具函数（配置/日志/路径）
├── config/                    # YAML 配置文件
├── prompts/                   # Prompt 模板
├── data/                      # 知识库 + 外部数据
├── db/                        # ChromaDB 持久化
├── docs/                      # 文档
├── tests/                     # 测试
├── scripts/                   # 初始化脚本
├── main.py                    # Streamlit 入口
├── Dockerfile                 # Docker 部署
└── docker-compose.yml
```

---

## 🔌 已注册工具（10个）

| 工具名 | 描述 | 状态 |
|--------|------|------|
| `rag_query` | RAG 知识库查询 | ✅ 生产就绪 |
| `get_weather` | 天气查询 | ✅ Mock |
| `get_user_location` | 用户定位 | ✅ Mock |
| `get_user_id` | 用户身份 | ✅ Mock |
| `get_current_month` | 当前月份 | ✅ Mock |
| `fetch_external_data` | 外部数据获取 | ✅ 生产就绪 |
| `fill_context_for_report` | 报告模式 | ✅ 生产就绪 |
| `query_order` | 外卖订单查询 | ✅ Mock（待接入外卖API） |
| `recommend_dish` | 菜品推荐 | ✅ Mock（待接入外卖API） |
| `check_delivery_status` | 配送状态 | ✅ Mock（待接入外卖API） |

---

## 📅 开发路线

按照 [原分析报告](https://chat.openai.com) 的规划：

- **5/9** ✅ 自实现 ReAct 循环（已完成）
- **5/10** 🔄 三层记忆系统 + 任务规划器
- **5/11** 🔄 RAG 增强（查询改写 + 重排序 + 多路召回）
- **5/12** 🔄 API 化 + Docker 部署
- **5/13** 🔄 迁移到新 workspace + 外卖系统集成
- **5/14** 🔄 单元测试 + 文档完善
- **5/15** 🔄 简历 + 面试准备

---

## 📚 面试准备

## 面试核心 Q&A

### 1. Agent 的 ReAct 循环是怎么实现的？

**答：** 我自实现了 Think → Act → Observe 闭环。Think 阶段让模型根据历史推理，Act 阶段解析模型的 Function Call 输出并调度注册的工具，Observe 阶段将工具返回结果追加到消息历史。循环直到模型给出不含工具调用的最终回复，或达到最大迭代轮数（默认 10 轮）。核心代码在 `agent/core/react_loop.py`。

### 2. 为什么不自接用 LangChain Agent？

**答：** 为了展示我对底层原理的理解。LangChain 封装了太多细节，面试官一问"Agent 具体怎么工作的"就答不上了。自己实现意味着我对 ReAct 的每一步都有掌控——包括什么时候调用模型、怎么解析输出、怎么处理错误、什么时候终止循环。

### 3. 记忆系统怎么设计的？

**答：** 三层架构。短期记忆（ShortTermMemory）保存当前会话的消息历史，用滑动窗口控制最大消息数。工作记忆（WorkingMemory）保存当前任务的执行状态和中间结果，Task Planner 写入、ReAct Loop 读取。长期记忆（LongTermMemory）跨会话持久化，用 ChromaDB 做向量存储，按语义相似度检索相关记忆。

### 4. RAG 做了哪些优化？

**答：** 三点优化。一是 Query Rewrite：用户问题可能模糊，先让 LLM 改写为更适合检索的查询。二是多路召回：向量检索 + BM25 关键词检索做 RRF 融合，提高召回率。三是 Reranker：对检索结果重排序，保留最相关的 top_k 个文档，减少无关信息对模型回答的干扰。

---

## 📌 许可证

MIT License
