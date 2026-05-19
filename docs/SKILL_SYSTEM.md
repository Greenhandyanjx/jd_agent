# JD-Agent Skill 系统文档

> **参考 OpenClaw 的 Skill 架构设计**，为 JD-Agent 增加可插拔的"技能"能力包机制

---

## 概述

Skill 系统是 JD-Agent 的"能力包"管理机制，对标 OpenClaw 的 `skills/` 目录结构。

### 核心概念

```
SkillSystem
  │
  ├─ skills/ 目录        ── 所有技能包都放在这里
  │    ├─ weather/        ── 🌤️ 天气技能
  │    │   ├─ SKILL.md    ── 技能声明文档（YAML + Markdown）
  │    │   └─ __init__.py ── 代码实现
  │    ├─ rag/            ── 📚 知识库检索技能
  │    │   ├─ SKILL.md
  │    │   └─ __init__.py
  │    └─ template/       ── 📋 技能模板（创建新技能时参考）
  │
  ├─ agent/skills/ 模块   ── 技能系统核心代码
  │    ├─ base.py         ── Skill / SkillAbility / SkillAsset
  │    ├─ manager.py      ── SkillManager
  │    ├─ loader.py       ── SkillLoader
  │    └─ SKILL_PATTERN.md─ 技能规范文档
  │
  └─ AgentLoop 集成
       └─ _match_skills() ── 每轮对话自动匹配技能
```

### Skill vs Tool 对比

| 维度 | Tool | Skill |
|------|------|-------|
| 粒度 | **原子操作** | **能力包** |
| 内容 | name + desc + execute() | name + desc + **多个 Tool** + SKILL.md |
| 发现 | 代码显式注册 | **文件系统自动发现** |
| 匹配 | Function Calling 自动 | 关键词 + 示例 + **意图匹配** |
| 附带 | 无 | 文档 + 数据 + prompt 模板 |
| 依赖 | 无 | SKILL.md 中声明 `requires` |

---

## 架构流程

### 启动时

```
main.py / api/routes.py
  │
  ├─ AgentOrchestrator()
  │    └─ AgentLoop(skills_dir="skills")
  │         └─ SkillManager("skills")
  │
  ├─ await orchestrator.initialize()
  │    ├─ AgentLoop.__init__()  → 注册基础 Tool
  │    └─ await loop.initialize_skills()
  │         ├─ SkillLoader.discover_all()
  │         │    ├─ 扫描 skills/ 下所有子目录
  │         │    ├─ 解析 SKILL.md 的 YAML 头部
  │         │    └─ 导入 __init__.py → 获取 skill_instance
  │         └─ SkillManager.register_all_tools(ToolRegistry)
  │              └─ 每个 Skill.abilities[].tool → ToolRegistry
  │
  └─ register_all_tools()  ← 旧的业务工具注册
```

### 对话中

```
用户输入 "今天北京天气怎么样？"
  │
  ├─ AgentLoop._process_message()
  │    ├─ _match_skills("今天北京天气怎么样？")
  │    │    ├─ SkillManager.match(query, top_k=3)
  │    │    │    ├─ weather   → score=0.9  ← 命中"天气""北京"关键词
  │    │    │    ├─ rag       → score=0.1
  │    │    │    └─ ...
  │    │    └─ compute_context(matches)
  │    │         → "[可用技能包]\n- [weather] 天气查询..."
  │    │
  │    ├─ context.build_messages(skill_context=...)
  │    │    → System Prompt 中注入了技能上下文
  │    │
  │    └─ _run_agent_loop(messages)
  │         ├─ LLM 看到 "可用技能包: weather"
  │         ├─ LLM 决定调用 get_weather(city="北京")
  │         └─ ToolRegistry.execute("get_weather", {city:"北京"})
  │              ↓
  │              WeatherSkill.get_weather 中的实现
```

---

## 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| **`agent/skills/__init__.py`** | 技能包入口，导出所有公开类 |
| **`agent/skills/base.py`** | Skill 抽象基类 + SkillAbility / SkillAsset |
| **`agent/skills/manager.py`** | SkillManager：注册、查询、意图匹配 |
| **`agent/skills/loader.py`** | SkillLoader：目录扫描、SKILL.md 解析 |
| **`agent/skills/SKILL_PATTERN.md`** | 技能规范文档（对标 OpenClaw SKILL.md） |
| **`skills/weather/SKILL.md`** | 天气技能声明文档 |
| **`skills/weather/__init__.py`** | 天气技能实现（3个 Tool） |
| **`skills/rag/SKILL.md`** | RAG 技能声明文档 |
| **`skills/rag/__init__.py`** | RAG 技能实现（1个 Tool） |
| **`skills/template/SKILL.md`** | 技能模板文档 |
| **`skills/template/__init__.py`** | 技能模板代码 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| **`agent/__init__.py`** | 导出 Skill / SkillManager / SkillLoader / SkillMatch |
| **`agent/core/types.py`** | 新增 `SkillMatch` 数据类型 |
| **`agent/loop.py`** | AgentLoop 集成 SkillManager，新增 `initialize_skills()`、`_match_skills()`，`_run_agent_loop()` 支持 `skill_context` |
| **`agent/orchestrator.py`** | 支持 `skills_dir` 参数，`initialize()` 自动调用 `initialize_skills()`，新增 `get_all_skills()` / `get_skill_count()` / `get_skill_names()` |
| **`api/routes.py`** | HealthResponse 新增 `skills_count`/`skills_names`，传入 `skills_dir` 初始化 |
| **`main.py`** | 侧边栏展示技能系统状态 |
| **`config/agent.yml`** | 新增技能系统配置项 |

---

## 阅读顺序

建议按以下顺序阅读新增和修改的文件：

### 第一步：理解 Skill 概念（5分钟）

1. **`agent/skills/SKILL_PATTERN.md`** — 技能规范文档，快速理解 Skill 是什么
2. **`docs/SKILL_SYSTEM.md`**（本文）— 系统整体设计

### 第二步：核心代码（10分钟）

3. **`agent/skills/base.py`** — Skill 抽象基类（最核心的抽象）
4. **`agent/skills/manager.py`** — SkillManager（调度中心）
5. **`agent/skills/loader.py`** — SkillLoader（自动发现引擎）

### 第三步：集成点（5分钟）

6. **`agent/loop.py`** 中的 `initialize_skills()` 和 `_match_skills()` 方法
7. **`agent/orchestrator.py`** → 构造器和 `initialize()`

### 第四步：示例技能（5分钟）

8. **`skills/weather/__init__.py`** — 完整的多 Tool 技能实现
9. **`skills/weather/SKILL.md`** — 配套文档
10. **`skills/rag/__init__.py`** — 单 Tool 技能实现
11. **`skills/template/__init__.py`** — 创建新技能的模板

### 第五步：Schema 补全（2分钟）

12. **`agent/core/types.py`** → `SkillMatch`
13. **`agent/__init__.py`** → 统一导出

### 第六步：UI 展示（2分钟）

14. **`main.py`** → 侧边栏技能状态
15. **`api/routes.py`** → Health 接口增加技能信息
16. **`config/agent.yml`** → 技能配置项

---

## 如何添加新技能

### 快速开始

```bash
# 1. 复制模板
cp -r skills/template skills/my_skill

# 2. 修改 SKILL.md（YAML 头部 + 描述）
# 3. 修改 __init__.py（替换 YourSkill / YourTool）
# 4. 重启 API
```

### 完整步骤

参见 `agent/skills/SKILL_PATTERN.md` → 六、编写指南。

### 关键原则

1. **导出 `skill_instance`** — 这是 SkillLoader 自动发现的唯一标识
2. **填写 keywords** — 意图匹配的好坏取决于关键词
3. **SKILL.md 必须带 YAML 头部** — `--- name: xxx ---`
4. **一个能力对应一个 Tool** — `SkillAbility(name="...", tool=YourTool())`
5. **Tool 和普通 Tool 完全一样** — 继承 `Tool`，实现 `execute()`

---

## FAQ

### Q：旧的 Tool 注册方式还能用吗？

A：可以。旧的 `register_all_tools(registry)` 仍然保留。Skill 系统是 **叠加** 在现有 Tool 系统之上的组织层。

### Q：Skill 自动发现的性能开销？

A：只在启动时扫描一次文件系统。运行时没有额外开销。

### Q：匹配不到技能怎么办？

A：LLM 仍然可以使用所有已注册的 Tool（通过 Function Calling）。技能匹配只是**附加信息**。

### Q：一个 Tool 可以属于多个 Skill 吗？

A：不建议。Tool 的名称在 ToolRegistry 中是唯一的。如果两个 Skill 注册同名的 Tool，后注册的会覆盖前一个。

### Q：SKILL.md 没有 YAML 头部会怎样？

A：不影响代码加载，但意图匹配和分类展示会缺少信息。建议总是带上 YAML 头部。
