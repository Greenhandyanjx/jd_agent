---
# 模板 SKILL.md：创建新技能时请参考此文件
# 将 "template" / "模板" 替换为你的技能信息
# 更详细的说明见：agent/skills/SKILL_PATTERN.md
---

name: template
description: "技能模板。复制此目录开始创建新技能。"
version: "0.1.0"
categories: ["模板"]
metadata:
  openclaw:
    emoji: "📋"
---

# Template Skill

技能模板，创建新技能时请参考。

## When to Use

这个技能是让你参考的，不是运行的。

## How to Create a New Skill

1. 复制 `skills/template/` → `skills/your_skill/`
2. 修改 SKILL.md 中的 YAML 头部
3. 修改 `__init__.py` 中的代码
4. 确保导出了 `skill_instance`
