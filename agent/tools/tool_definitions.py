"""
Agent Tools: 所有内置工具定义
参考 nanobot 的 agent/tools/ 下各文件设计，同时兼容原有工具逻辑

所有工具都继承 agent.tools.base.Tool 基类，
通过 ToolRegistry.register() 注册。

包含：
1. 文件系统工具（在 filesystem.py 中定义）
2. Web 工具（在 web.py 中定义）
3. 以下在本文定义：RAG 查询、天气、Mock 数据等业务工具
"""

import random
from typing import Any

from loguru import logger

from agent.tools.base import Tool
try:
    from tools.rag_service import RagSummarizeService
except ModuleNotFoundError:
    from rag.rag_service import RagSummarizeService

# RAG 服务单例
_rag = RagSummarizeService()

# Mock 数据
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉"]
MONTH_ARR = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
             "2025-06", "2025-07", "2025-08", "2025-09", "2025-10",
             "2025-11", "2025-12"]
USER_IDS = ["1001", "1002", "1003", "1004", "1005"]


class RAGQueryTool(Tool):
    """RAG 检索增强查询"""

    @property
    def name(self) -> str:
        return "rag_query"

    @property
    def description(self) -> str:
        return "RAG 检索增强查询：基于知识库文档回答用户关于产品、使用方法的专业问题"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "查询内容",
                "required": True,
            },
        }

    async def execute(self, query: str, **kwargs: Any) -> str:
        return _rag.rag_summarize(query)


class WeatherTool(Tool):
    """天气查询"""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "获取指定城市的天气信息，返回天气、温度、湿度等"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "city": {
                "type": "string",
                "description": "城市名称",
                "required": True,
            },
        }

    async def execute(self, city: str, **kwargs: Any) -> str:
        weathers = ["晴朗", "多云", "小雨", "阴天", "晴转多云"]
        temps = [22, 26, 28, 30, 18, 20, 24]
        return (f"{city}今日天气：{random.choice(weathers)}，"
                f"气温{random.choice(temps)}°C，湿度{random.randint(40, 80)}%")


class UserLocationTool(Tool):
    """获取用户位置"""

    @property
    def name(self) -> str:
        return "get_user_location"

    @property
    def description(self) -> str:
        return "获取用户当前所在城市的名称"

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, **kwargs: Any) -> str:
        return random.choice(CITIES)


class UserIDTool(Tool):
    """获取用户 ID"""

    @property
    def name(self) -> str:
        return "get_user_id"

    @property
    def description(self) -> str:
        return "获取当前登录用户的ID，用于查询用户数据"

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, **kwargs: Any) -> str:
        return random.choice(USER_IDS)


class CurrentMonthTool(Tool):
    """获取当前月份"""

    @property
    def name(self) -> str:
        return "get_current_month"

    @property
    def description(self) -> str:
        return "获取当前的月份，格式为YYYY-MM"

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, **kwargs: Any) -> str:
        return random.choice(MONTH_ARR)


class OrderQueryTool(Tool):
    """查询外卖订单"""

    @property
    def name(self) -> str:
        return "query_order"

    @property
    def description(self) -> str:
        return "查询外卖订单信息，根据订单ID获取订单状态、金额等详情"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "order_id": {
                "type": "string",
                "description": "订单ID",
                "required": False,
            },
        }

    async def execute(self, order_id: str = "", **kwargs: Any) -> str:
        statuses = ["已支付", "配送中", "已完成", "已取消"]
        return (f"订单 {order_id or '未知'}："
                f"状态 {random.choice(statuses)}，"
                f"金额 ¥{random.randint(10, 50)}.{random.randint(0, 99):02d}，"
                f"下单时间 2026-05-{random.randint(1, 8):02d} 12:30")


class DishRecommendTool(Tool):
    """推荐菜品"""

    @property
    def name(self) -> str:
        return "recommend_dish"

    @property
    def description(self) -> str:
        return "根据用户偏好推荐热门菜品，返回菜品名称和价格列表"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "user_id": {
                "type": "string",
                "description": "用户ID",
                "required": False,
            },
        }

    async def execute(self, user_id: str = "", **kwargs: Any) -> str:
        dishes = [
            {"name": "黄焖鸡米饭", "price": 18.0, "sales": 256},
            {"name": "麻辣香锅", "price": 28.0, "sales": 189},
            {"name": "番茄牛腩面", "price": 22.0, "sales": 145},
            {"name": "螺蛳粉", "price": 15.0, "sales": 320},
            {"name": "肠粉", "price": 12.0, "sales": 278},
        ]
        selected = random.sample(dishes, 3)
        return "\n".join([
            f"- {d['name']} ¥{d['price']} (月售{d['sales']})" for d in selected
        ])


class DeliveryStatusTool(Tool):
    """查询配送状态"""

    @property
    def name(self) -> str:
        return "check_delivery_status"

    @property
    def description(self) -> str:
        return "查询外卖配送状态，返回骑手位置和预计送达时间"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "order_id": {
                "type": "string",
                "description": "订单ID",
                "required": False,
            },
        }

    async def execute(self, order_id: str = "", **kwargs: Any) -> str:
        return (f"订单 {order_id or '当前订单'}："
                f"骑手距您 {random.randint(200, 1500)} 米，"
                f"预计 {random.randint(5, 30)} 分钟后送达")


# ─── 工具注册函数 ────────────────────────────

def register_all_tools(registry) -> None:
    """
    将所有业务工具注册到指定的 ToolRegistry。
    
    对应 nanobot 的 _register_default_tools() 模式，
    使得工具注册集中到一处管理。
    """
    tools = [
        RAGQueryTool(),
        WeatherTool(),
        UserLocationTool(),
        UserIDTool(),
        CurrentMonthTool(),
        OrderQueryTool(),
        DishRecommendTool(),
        DeliveryStatusTool(),
    ]
    for tool in tools:
        registry.register(tool)
    logger.info(f"[Tools] 注册完成: {len(tools)} 个业务工具")
