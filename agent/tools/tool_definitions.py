"""
Agent Tools: 所有工具定义
将原有项目的工具迁移并注册到新的 Tool Registry
"""
import random
import os
from utils.logger import logger
from utils.config import agent_conf
from utils.path_tool import get_abs_path
from agent.core.tool_registry import tool, clear_registry
from rag.rag_service import RagSummarizeService

# RAG 服务单例
_rag = RagSummarizeService()

# Mock 数据
USER_IDS = ["1001", "1002", "1003", "1004", "1005",
            "1006", "1007", "1008", "1009", "1010"]
MONTH_ARR = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
             "2025-06", "2025-07", "2025-08", "2025-09", "2025-10",
             "2025-11", "2025-12"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉"]


# ==================== 原有工具迁移 ====================

@tool(description="RAG检索增强查询：基于知识库文档回答用户关于产品、使用方法的专业问题")
def rag_query(query: str) -> str:
    """RAG 检索增强查询"""
    return _rag.rag_summarize(query)


@tool(description="获取指定城市的天气信息，返回天气、温度、湿度等")
def get_weather(city: str) -> str:
    """获取天气信息（Mock）"""
    weathers = ["晴朗", "多云", "小雨", "阴天", "晴转多云"]
    temps = [22, 26, 28, 30, 18, 20, 24]
    w = random.choice(weathers)
    t = random.choice(temps)
    return f"{city}今日天气：{w}，气温{t}°C，湿度{random.randint(40, 80)}%"


@tool(description="获取用户当前所在城市的名称")
def get_user_location() -> str:
    """获取用户位置"""
    return random.choice(CITIES)


@tool(description="获取当前登录用户的ID，用于查询用户数据")
def get_user_id() -> str:
    """获取用户ID"""
    return random.choice(USER_IDS)


@tool(description="获取当前的月份，格式为YYYY-MM")
def get_current_month() -> str:
    """获取当前月份"""
    return random.choice(MONTH_ARR)


# 外部数据缓存
_external_data = {}

def _load_external_data():
    """加载外部数据"""
    if _external_data:
        return

    try:
        data_path = get_abs_path(agent_conf.get("external_data_path", "data/external/records.csv"))
        if not os.path.exists(data_path):
            logger.warning(f"[Tools] 外部数据文件不存在: {data_path}")
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")
                if len(arr) < 6:
                    continue
                uid = arr[0].strip('"')
                feature = arr[1].strip('"')
                efficiency = arr[2].strip('"')
                consumables = arr[3].strip('"')
                comparison = arr[4].strip('"')
                time = arr[5].strip('"')

                if uid not in _external_data:
                    _external_data[uid] = {}
                _external_data[uid][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }
    except Exception as e:
        logger.error(f"[Tools] 加载外部数据失败: {e}")


@tool(description="从外部系统获取指定用户指定月份的使用记录，以字符串格式返回")
def fetch_external_data(user_id: str, month: str) -> str:
    """获取外部系统数据"""
    _load_external_data()
    try:
        return str(_external_data.get(user_id, {}).get(month, ""))
    except KeyError:
        return ""


@tool(description="触发报告生成模式，调用后系统会切换到报告提示词模板")
def fill_context_for_report() -> str:
    """填充报告上下文（触发动态Prompt切换）"""
    return "报告模式已启动"


# ==================== 新增工具（为外卖系统集成预留） ====================

@tool(description="查询外卖订单信息，根据订单ID获取订单状态、金额等详情")
def query_order(order_id: str = "") -> str:
    """查询订单信息（Mock，后续接入外卖系统API）"""
    statuses = ["已支付", "配送中", "已完成", "已取消"]
    return (f"订单 {order_id or '未知'}："
            f"状态 {random.choice(statuses)}，"
            f"金额 ¥{random.randint(10, 50)}.{random.randint(0, 99):02d}，"
            f"下单时间 2026-05-{random.randint(1, 8):02d} 12:30")


@tool(description="根据用户偏好推荐热门菜品，返回菜品名称和价格列表")
def recommend_dish(user_id: str = "") -> str:
    """推荐菜品（Mock，后续接入外卖系统API）"""
    dishes = [
        {"name": "黄焖鸡米饭", "price": 18.0, "sales": 256},
        {"name": "麻辣香锅", "price": 28.0, "sales": 189},
        {"name": "番茄牛腩面", "price": 22.0, "sales": 145},
        {"name": "螺蛳粉", "price": 15.0, "sales": 320},
        {"name": "肠粉", "price": 12.0, "sales": 278},
    ]
    selected = random.sample(dishes, 3)
    return "\n".join([f"- {d['name']} ¥{d['price']} (月售{d['sales']})" for d in selected])


@tool(description="查询外卖配送状态，返回骑手位置和预计送达时间")
def check_delivery_status(order_id: str = "") -> str:
    """查询配送状态（Mock，后续接入外卖系统API）"""
    return (f"订单 {order_id or '当前订单'}："
            f"骑手距您 {random.randint(200, 1500)} 米，"
            f"预计 {random.randint(5, 30)} 分钟后送达，"
            f"骑手电话: 138****{random.randint(1000, 9999)}")


# 导入时自动注册（@tool 装饰器在函数定义时即注册到全局 registry）
def init_all_tools():
    """确认所有工具已注册"""
    from agent.core.tool_registry import get_all_tools
    tools = get_all_tools()
    logger.info(f"[Tools] 注册完成: {len(tools)} 个工具: {[t.name for t in tools]}")
