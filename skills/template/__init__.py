"""技能模板：创建新技能时请参考此文件"""

# ─────────────────────────────────────────────────────────
# 技能模板
#
# 创建新技能的步骤：
#   1. 复制此目录 `skills/template/` → `skills/your_skill/`
#   2. 修改 SKILL.md 中的 YAML 头部和正文
#   3. 修改 __init__.py，替换 YourSkill / YourTool
#   4. 确保导出 skill_instance
#
# 更详细的说明见：agent/skills/SKILL_PATTERN.md
# ─────────────────────────────────────────────────────────

from agent.skills.base import Skill, SkillAbility
from agent.tools.base import Tool


# ── 1. 定义工具类 ──────────────────────────────

class YourTool(Tool):
    """你的工具：在这里实现具体功能"""

    @property
    def name(self) -> str:
        return "your_tool"  # Tool 名称（Function Calling 中的函数名）

    @property
    def description(self) -> str:
        return "你的工具描述（LLM 用来决定何时调用此工具）"

    @property
    def parameters(self) -> dict:
        return {
            "param1": {
                "type": "string",  # 支持: string / integer / number / boolean / array / object
                "description": "参数描述",
                "required": True,  # 是否必填
            },
            "param2": {
                "type": "integer",
                "description": "可选参数",
                "required": False,
            },
        }

    async def execute(self, param1: str, param2: int = 0, **kwargs) -> str:
        """工具执行逻辑"""
        # 在这里实现你的功能
        return f"执行结果: param1={param1}, param2={param2}"


# ── 2. 定义技能类 ──────────────────────────────

class YourSkill(Skill):
    """你的技能：将多个相关工具组织为一个能力包"""

    @property
    def name(self) -> str:
        return "your_skill"  # 技能名称（也是目录名）

    @property
    def description(self) -> str:
        return "你的技能描述"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def categories(self) -> list[str]:
        """技能分类（用于管理界面）"""
        return ["自定义"]

    @property
    def abilities(self) -> list[SkillAbility]:
        """能力列表：一个 Skill 可以包含多个 Tool"""
        return [
            SkillAbility(
                name="your_tool",
                description="工具描述",
                keywords=["关键词A", "关键词B"],  # 意图匹配用
                examples=["示例查询1", "示例查询2"],  # 意图匹配+文档用
                tool=YourTool(),
            ),
            # 可以添加更多能力...
        ]

    async def initialize(self) -> None:
        """技能初始化钩子（可选）"""
        # 在这里做初始化，如连接数据库、加载模型等
        pass


# ── 3. 导出实例（必须！） ──────────────────────

skill_instance = YourSkill()
