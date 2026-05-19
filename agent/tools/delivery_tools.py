"""
Agent Tools: 外卖配送业务工具
=========================================================
为 Campus Food Delivery 系统提供四个核心工具：
  1. query_order        — 查询订单状态和详情
  2. recommend_dish     — 根据用户偏好推荐菜品
  3. check_delivery_status — 查询配送进度和骑手位置
  4. customer_service   — 生成智能客服回复（模板化）

所有工具继承基础 Tool 类，调用外部 Delivery Backend API。

依赖：
  - httpx（已在 requirements.txt 中添加）
  - delivery_backend_url 从 delivery_config 获取
"""

import re
from typing import Any

import httpx
from loguru import logger

from agent.tools.base import Tool
from delivery_config import DELIVERY_BACKEND_URL, AVAILABLE_MERCHANT_IDS


class QueryOrderTool(Tool):
    """
    查询订单工具
    ============
    通过调用后端 GET /api/merchant/order/detail 接口获取订单详情。
    需要认证信息（会在请求中携带 Cookie/token）。
    """

    @property
    def name(self) -> str:
        return "query_order"

    @property
    def description(self) -> str:
        return (
            "查询订单状态和详情。根据订单 ID 获取订单的完整信息，"
            "包括订单状态、菜品列表、金额、收货地址、下单时间等。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "order_id": {
                "type": "string",
                "description": "订单 ID，例如 'ORD20260513001'",
                "required": True,
            },
            "user_id": {
                "type": "string",
                "description": "用户 ID（可选），用于确认订单归属",
                "required": False,
            },
        }

    async def execute(self, order_id: str, user_id: str = "", **kwargs: Any) -> str:
        """
        查询订单详情：
          1. 调用后端 GET /api/merchant/order/detail?id={order_id}
          2. 解析返回的 JSON 数据
          3. 格式化为易读文本
        """
        url = f"{DELIVERY_BACKEND_URL}/api/merchant/order/detail?id={order_id}"
        logger.info(f"[QueryOrderTool] 查询订单: order_id={order_id}, url={url}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # 注意：/api/merchant/order/detail 需要 auth 中间件
                # 但工具本身没有登录上下文，这里使用不携带 Cookie 的请求
                # 前端在使用时会在代理层处理 Cookie
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return "⚠️ 订单查询超时，请稍后重试。"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "⚠️ 查询订单需要登录认证，请先登录后再试。"
            elif e.response.status_code == 404:
                return f"⚠️ 未找到订单 '{order_id}'，请检查订单号是否正确。"
            return f"⚠️ 订单查询失败 (HTTP {e.response.status_code})"
        except Exception as e:
            logger.error(f"[QueryOrderTool] 请求异常: {e}")
            return f"⚠️ 订单查询出错: {type(e).__name__}"

        # 解析返回数据（适配后端不同的返回格式）
        # 后端可能的格式：{code: 1, data: {...}} 或直接 {...}
        order_data = None
        if isinstance(data, dict):
            if data.get("code") == 1 or data.get("code") == "1":
                order_data = data.get("data")
            elif "id" in data or "orderId" in data or "order_id" in data:
                order_data = data
            elif "order" in data:
                order_data = data["order"]
            # 如果 code!==1，将提示消息显示给用户
            elif data.get("code") is not None and data.get("code") != 1:
                msg = data.get("msg") or data.get("message") or "未知错误"
                return f"⚠️ 查询失败: {msg}"

        if not order_data:
            return f"⚠️ 订单 '{order_id}' 未找到或返回数据格式异常。"

        # 格式化输出订单详情
        lines = []
        lines.append(f"📋 **订单详情** (ID: {order_data.get('id') or order_data.get('orderId') or order_id})")
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")

        # 订单状态
        status_map = {
            "pending": "⏳ 待处理", "accepted": "✅ 已接单", "preparing": "👨‍🍳 准备中",
            "delivering": "🚴 配送中", "delivered": "📦 已送达", "completed": "✅ 已完成",
            "cancelled": "❌ 已取消", "rejected": "❌ 已拒绝",
            0: "⏳ 待支付", 1: "✅ 已支付", 2: "🚴 配送中", 3: "✅ 已完成", 4: "❌ 已取消",
        }
        status = order_data.get("status")
        status_str = status_map.get(status) if status is not None else "未知"
        if isinstance(status, str):
            status_str = status_map.get(status, status)
        lines.append(f"**状态**: {status_str}")

        # 基本信息
        created_at = order_data.get("createdAt") or order_data.get("created_at") or order_data.get("createTime") or "-"
        total_amount = order_data.get("totalAmount") or order_data.get("total_amount") or order_data.get("totalPrice") or "-"
        lines.append(f"**下单时间**: {created_at}")
        lines.append(f"**订单金额**: ¥{total_amount}")

        # 商家信息
        merchant_name = order_data.get("merchantName") or order_data.get("merchant_name") or order_data.get("shopName") or ""
        if merchant_name:
            lines.append(f"**商家**: {merchant_name}")

        # 收货信息
        address = order_data.get("address") or order_data.get("consigneeAddress") or ""
        consignee = order_data.get("consignee") or order_data.get("consigneeName") or ""
        if consignee:
            lines.append(f"**收货人**: {consignee}")
        if address:
            lines.append(f"**地址**: {address}")

        # 菜品列表
        dishes = order_data.get("dishes") or order_data.get("items") or order_data.get("orderItems") or []
        if dishes and isinstance(dishes, list):
            lines.append(f"\n**菜品明细**:")
            for i, dish in enumerate(dishes, 1):
                dname = dish.get("name") or dish.get("dishName") or dish.get("dish_name") or f"菜品{i}"
                dprice = dish.get("price") or dish.get("dishPrice") or "-"
                dqty = dish.get("quantity") or dish.get("qty") or 1
                lines.append(f"  {i}. {dname} × {dqty}  ¥{dprice}")

        # 骑手信息
        rider = order_data.get("rider") or order_data.get("deliveryMan") or {}
        if rider and isinstance(rider, dict):
            rider_name = rider.get("name") or rider.get("riderName") or ""
            rider_phone = rider.get("phone") or rider.get("riderPhone") or ""
            if rider_name:
                lines.append(f"\n**骑手信息**: {rider_name} {rider_phone}")

        return "\n".join(lines)


class RecommendDishTool(Tool):
    """
    菜品推荐工具
    ============
    根据用户的口味偏好，从所有商家中匹配最合适的菜品。
    关键词匹配基于菜品名称、描述和标签。
    """

    @property
    def name(self) -> str:
        return "recommend_dish"

    @property
    def description(self) -> str:
        return (
            "根据用户偏好推荐菜品。支持口味偏好如：辣、清淡、甜、粤菜、川菜等，"
            "系统会从所有商家中匹配菜品并返回推荐列表。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "preference": {
                "type": "string",
                "description": "用户的口味偏好，例如：'辣', '清淡', '甜', '粤菜', '川菜', '日料', '汉堡'",
                "required": True,
            },
            "max_items": {
                "type": "integer",
                "description": "最大推荐数量（默认5个）",
                "required": False,
            },
        }

    async def execute(self, preference: str, max_items: int = 5, **kwargs: Any) -> str:
        """
        推荐菜品流程：
          1. 遍历所有商家 ID
          2. 调用 GET /api/store/dishes?merchant_id={mid}&page=1&pageSize=50 获取菜品
          3. 根据偏好关键词匹配菜品名称、描述、标签
          4. 按相关度排序，返回 Top N
        """
        logger.info(f"[RecommendDishTool] 推荐菜品: preference={preference}, max_items={max_items}")

        # 偏好关键词拆解
        keywords = self._extract_keywords(preference)

        all_matched = []  # [(score, dish_info), ...]

        # 并发请求所有商家
        async with httpx.AsyncClient(timeout=15.0) as client:
            for mid in AVAILABLE_MERCHANT_IDS:
                try:
                    url = f"{DELIVERY_BACKEND_URL}/api/store/dishes?merchant_id={mid}&page=1&pageSize=50"
                    response = await client.get(url, follow_redirects=True)
                    if response.status_code != 200:
                        continue
                    data = response.json()
                except Exception:
                    continue

                # 解析菜品列表
                dishes = self._extract_dishes(data)
                if not dishes:
                    continue

                # 获取商家名称
                merchant_name = self._get_merchant_name(data) or f"商家{mid}"

                for dish in dishes:
                    score = self._match_dish(dish, keywords)
                    if score > 0:
                        all_matched.append({
                            "score": score,
                            "name": dish.get("name", "未知菜品"),
                            "price": dish.get("price", 0),
                            "description": dish.get("description", ""),
                            "merchant": merchant_name,
                            "sales": dish.get("sales", 0),
                        })

        if not all_matched:
            return (
                f"😅 抱歉，没有找到与「{preference}」相关的菜品推荐。\n"
                f"建议试试其他关键词：辣、清淡、甜、粤菜、川菜、日料、汉堡、面条等。"
            )

        # 按相关度排序，取 Top N
        all_matched.sort(key=lambda x: x["score"], reverse=True)
        top_n = all_matched[:min(max_items, len(all_matched))]

        lines = []
        lines.append(f"🍽️ **「{preference}」推荐菜品** (共找到 {len(all_matched)} 个匹配)")
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
        for i, item in enumerate(top_n, 1):
            lines.append(f"**{i}. {item['name']}**")
            lines.append(f"   💰 ¥{item['price']}  |  🏪 {item['merchant']}")
            if item.get("description"):
                desc = item["description"][:60]
                lines.append(f"   📝 {desc}")
            lines.append("")

        return "\n".join(lines)

    def _extract_keywords(self, preference: str) -> list[str]:
        """将用户偏好拆解为搜索关键词"""
        # 常见口味组合
        taste_groups = {
            "辣": ["辣", "麻辣", "香辣", "酸辣", "辣椒", "spicy"],
            "清淡": ["清淡", "清炒", "白灼", "蒸", "原味", "light"],
            "甜": ["甜", "甜品", "糖", "蜜", "sweet"],
            "粤菜": ["粤菜", "广东", "广式", "煲仔", "叉烧", "烧腊", "白切"],
            "川菜": ["川菜", "四川", "麻辣", "水煮", "酸菜", "腊肉"],
            "日料": ["日料", "寿司", "刺身", "拉面", "鳗鱼", "日式", "便当"],
            "汉堡": ["汉堡", "burger", "炸鸡", "薯条", "西式"],
            "面": ["面", "面条", "拉面", "拌面", "汤面", "米粉", "米线"],
            "饭": ["饭", "米饭", "盖饭", "炒饭", "煲仔饭", "丼"],
            "饮品": ["茶", "奶茶", "咖啡", "果汁", "奶昔", "可乐", "雪碧"],
        }

        keywords = []
        pref_lower = preference.lower()

        # 先从口味分组找
        for group, group_kw in taste_groups.items():
            if group in pref_lower or any(kw in group for kw in [pref_lower]):
                keywords.extend(group_kw)

        # 直接把用户输入的每个字作为关键词
        keywords.append(pref_lower)
        keywords = list(set(keywords))  # 去重
        return keywords

    def _extract_dishes(self, data: Any) -> list[dict]:
        """从后端返回的数据中提取菜品列表"""
        if isinstance(data, dict):
            if data.get("code") == 1 or data.get("code") == "1":
                dishes = data.get("data", [])
                if isinstance(dishes, list):
                    return dishes
                if isinstance(dishes, dict):
                    return dishes.get("records", dishes.get("list", []))
            elif "dishes" in data:
                return data["dishes"]
            elif "records" in data:
                return data["records"]
            elif "list" in data:
                return data["list"]
            return []
        return []

    def _get_merchant_name(self, data: Any) -> str:
        """从响应中提取商家名称"""
        if isinstance(data, dict):
            # 有些接口在 data 中包含 merchant_name
            d = data.get("data") if isinstance(data.get("data"), dict) else data
            return d.get("merchantName") or d.get("merchant_name") or d.get("shopName") or ""
        return ""

    def _match_dish(self, dish: dict, keywords: list[str]) -> int:
        """
        计算菜品与关键词的匹配得分。
        得分规则：
          - 名称完全匹配关键词 → +10
          - 名称包含关键词 → +5
          - 描述包含关键词 → +3
          - 标签包含关键词 → +4
        """
        score = 0
        name = (dish.get("name") or "").lower()
        desc = (dish.get("description") or dish.get("desc") or "").lower()
        tags = [t.lower() for t in (dish.get("tags") or []) if isinstance(t, str)]
        category = (dish.get("categoryName") or dish.get("category") or "").lower()

        for kw in keywords:
            kw_lower = kw.lower().strip()
            if not kw_lower:
                continue

            # 名称匹配
            if name == kw_lower:
                score += 10
            elif kw_lower in name.split():
                score += 7
            elif kw_lower in name:
                score += 5

            # 描述匹配
            if kw_lower in desc:
                score += 3

            # 标签匹配
            if any(kw_lower in tag for tag in tags):
                score += 4

            # 分类匹配
            if kw_lower in category:
                score += 4

        return score


class CheckDeliveryStatusTool(Tool):
    """
    配送状态查询工具
    ================
    通过调用后端 GET /api/order/status?id={order_id} 获取配送进度。
    该接口无需认证，可以直接查询。
    """

    @property
    def name(self) -> str:
        return "check_delivery_status"

    @property
    def description(self) -> str:
        return (
            "查询配送进度和骑手位置。根据订单 ID 获取当前配送状态、"
            "预计送达时间、骑手信息等。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "order_id": {
                "type": "string",
                "description": "订单 ID",
                "required": True,
            },
        }

    async def execute(self, order_id: str, **kwargs: Any) -> str:
        """
        查询配送状态：
          1. 调用后端 GET /api/order/status?id={order_id}
          2. 解析返回的状态数据
          3. 格式化为易读的配送跟踪文本
        """
        url = f"{DELIVERY_BACKEND_URL}/api/order/status?id={order_id}"
        logger.info(f"[CheckDeliveryStatusTool] 查询配送状态: order_id={order_id}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return "⚠️ 配送状态查询超时，请稍后重试。"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"⚠️ 未找到订单 '{order_id}' 的配送信息。"
            return f"⚠️ 配送状态查询失败 (HTTP {e.response.status_code})"
        except Exception as e:
            logger.error(f"[CheckDeliveryStatusTool] 请求异常: {e}")
            return f"⚠️ 配送状态查询出错: {type(e).__name__}"

        # 解析状态数据
        status_data = None
        if isinstance(data, dict):
            if data.get("code") == 1 or data.get("code") == "1":
                status_data = data.get("data")
            elif "status" in data or "deliveryStatus" in data:
                status_data = data
            elif "msg" in data:
                return f"⚠️ {data.get('msg')}"

        if not status_data:
            return "⚠️ 无法获取配送状态信息。"

        lines = []
        lines.append(f"🚚 **配送跟踪** (订单: {order_id})")
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")

        # 配送状态
        status = status_data.get("status") or status_data.get("deliveryStatus") or ""
        status_map = {
            "pending": "⏳ 等待接单",
            "accepted": "✅ 商家已接单",
            "preparing": "👨‍🍳 商家准备中",
            "looking_rider": "🔍 正在分配骑手",
            "rider_accepted": "🚴 骑手已接单",
            "delivering": "🚴 配送中",
            "delivered": "📦 已送达",
            "completed": "✅ 已完成",
            "cancelled": "❌ 已取消",
        }
        status_str = status_map.get(status, status if isinstance(status, str) else f"状态码: {status}")
        lines.append(f"**当前状态**: {status_str}")

        # 预计送达时间
        estimated = (
            status_data.get("estimatedDeliveryTime")
            or status_data.get("estimatedTime")
            or status_data.get("estimatedArrival")
            or ""
        )
        if estimated:
            lines.append(f"**预计送达**: {estimated}")
        else:
            lines.append(f"**预计送达**: 计算中...")

        # 骑手信息
        rider = status_data.get("rider") or status_data.get("deliveryMan") or {}
        if rider and isinstance(rider, dict):
            rider_name = rider.get("name") or rider.get("riderName") or ""
            rider_phone = rider.get("phone") or rider.get("riderPhone") or ""
            rider_loc = rider.get("location") or rider.get("position") or {}

            if rider_name:
                lines.append(f"\n**👤 骑手信息**:")
                lines.append(f"  名字: {rider_name}")
                if rider_phone:
                    lines.append(f"  电话: {rider_phone}")

            # 骑手位置
            if isinstance(rider_loc, dict):
                lat = rider_loc.get("lat") or rider_loc.get("latitude")
                lng = rider_loc.get("lng") or rider_loc.get("longitude")
                if lat and lng:
                    lines.append(f"  当前位置: ({lat}, {lng})")
            elif isinstance(rider_loc, str) and rider_loc:
                lines.append(f"  当前位置: {rider_loc}")

            # 距离信息
            distance = rider.get("distance") or status_data.get("distance")
            if distance:
                lines.append(f"  距您: {distance}米")

        # 配送进度条
        progress = status_data.get("progress") or status_data.get("deliveryProgress")
        if progress is not None:
            bar_len = 20
            filled = int(progress * bar_len / 100) if isinstance(progress, (int, float)) else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(f"\n**配送进度**: {bar} {progress}%")

        return "\n".join(lines)


class CustomerServiceTool(Tool):
    """
    智能客服工具
    ============
    不调用外部 API，而是通过模板化策略生成针对性的客服回复。
    支持以下场景：
      - 订单丢失 → 建议追踪、联系骑手
      - 配送延迟 → 道歉、建议联系商家
      - 送错商品 → 道歉、建议退款/换货
      - 一般咨询 → 提供有用信息
    """

    @property
    def name(self) -> str:
        return "customer_service"

    @property
    def description(self) -> str:
        return (
            "生成智能客服回复。处理用户的投诉、咨询、售后问题等，"
            "例如订单丢失、配送延迟、送错商品等情况。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "user_query": {
                "type": "string",
                "description": "用户的问题或投诉内容",
                "required": True,
            },
            "order_id": {
                "type": "string",
                "description": "相关订单 ID（可选）",
                "required": False,
            },
        }

    async def execute(self, user_query: str, order_id: str = "", **kwargs: Any) -> str:
        """
        智能客服回复生成：
          1. 分析用户问题，识别场景类型
          2. 根据场景选择对应的回复模板
          3. 生成共情 + 解决方案的回复
        """
        logger.info(f"[CustomerServiceTool] 客服请求: query={user_query[:50]}..., order_id={order_id}")

        # 场景识别
        query_lower = user_query.lower()
        scenarios = self._identify_scenario(query_lower)

        # 构建回复
        lines = []
        lines.append("🤖 **智能客服助手**")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")

        # 1. 共情开头
        lines.append(self._get_empathy_line(scenarios))

        # 2. 针对具体场景生成解决方案
        solution = self._generate_solution(scenarios, order_id, user_query)
        lines.append(solution)

        # 3. 如果包含订单 ID，附上操作建议
        if order_id:
            lines.append(f"\n📌 **订单号**: {order_id}")
            lines.append("您可以在「订单详情」中查看最新进度。")

        # 4. 常用联系方式
        lines.append("\n💡 **如需进一步帮助**:")
        lines.append("- 联系商家: 在订单页面点击「联系商家」按钮")
        lines.append("- 联系骑手: 配送中可直接电话联系骑手")
        lines.append("- 平台客服: 工作时间 9:00-22:00")

        return "\n".join(lines)

    def _identify_scenario(self, query: str) -> list[str]:
        """
        识别用户问题的场景类型。
        返回匹配的场景名称列表。
        """
        scenarios = []

        # 场景关键词匹配
        scenario_rules = {
            "lost_order": [
                "丢", "不见了", "没收到", "找不到", "missing", "lost",
                "还没到", "一直没", "没到", "消失",
            ],
            "late_delivery": [
                "慢", "迟", "晚", "太久", "还没来", "超时", "延时",
                "late", "delay", "wait", "等很久", "还没到",
            ],
            "wrong_item": [
                "错", "不对", "不是我点的", "送错了", "少了", "多",
                "wrong", "not what", "incorrect",
            ],
            "refund": [
                "退款", "退钱", "退", "refund", "退货", "取消订单",
                "cancel", "不想吃了",
            ],
            "quality": [
                "不好吃", "难吃", "不新鲜", "坏了", "变质", "异味",
                "bad", "terrible", "质量", "差评",
            ],
            "price": [
                "贵", "价格", "price", "收费", "多收", "乱收费",
                "coupon", "优惠", "折扣", "discount",
            ],
        }

        for scenario, keywords in scenario_rules.items():
            if any(kw in query for kw in keywords):
                scenarios.append(scenario)

        if not scenarios:
            scenarios.append("general")

        return scenarios

    def _get_empathy_line(self, scenarios: list[str]) -> str:
        """根据场景生成共情语句"""
        empathy_map = {
            "lost_order": "😔 很抱歉您的订单似乎遇到了一些问题，我理解等待的心情。",
            "late_delivery": "⏰ 非常抱歉配送延误给您带来了不便。",
            "wrong_item": "😅 很抱歉给您送错了商品，我们会尽快处理。",
            "refund": "💰 了解您的退款需求，我来帮您说明流程。",
            "quality": "😞 很抱歉菜品质量没有达到预期。",
            "price": "🔍 我帮您核实一下收费情况。",
            "general": "👋 您好！很高兴为您服务。",
        }

        for s in scenarios:
            if s in empathy_map:
                return empathy_map[s]

        return empathy_map["general"]

    def _generate_solution(self, scenarios: list[str], order_id: str, query: str) -> str:
        """根据场景生成解决方案"""
        parts = []

        for scenario in scenarios:
            if scenario == "lost_order":
                parts.append(
                    "**🔍 关于订单丢失**:\n"
                    "1. 请先确认订单状态是否为「配送中」或「已完成」\n"
                    "2. 如果状态是配送中，请尝试直接联系骑手确认位置\n"
                    "3. 如果是已送达但未收到，可能是地址有误或误放\n"
                    "4. 您可以申请平台介入，由客服核实后进行退款处理"
                )

            elif scenario == "late_delivery":
                parts.append(
                    "**⏱️ 关于配送延迟**:\n"
                    "1. 很抱歉，高峰期配送可能稍有延迟\n"
                    "2. 您可以在订单页面查看骑手实时位置\n"
                    "3. 如果超时较长，建议直接联系商家了解情况\n"
                    "4. 超时严重可申请订单补偿或部分退款"
                )

            elif scenario == "wrong_item":
                parts.append(
                    "**🔄 关于商品问题**:\n"
                    "1. 请拍照留存证据（送错的商品照片）\n"
                    "2. 联系商家沟通换货或部分退款\n"
                    "3. 如果是漏送，商家通常会安排补送或退款\n"
                    "4. 在订单页面点击「联系商家」按钮即可沟通"
                )

            elif scenario == "refund":
                parts.append(
                    "**💳 关于退款**:\n"
                    "1. 如果订单尚未制作，可以直接取消订单（全额退款）\n"
                    "2. 如果商家已接单，需要联系商家协商取消\n"
                    "3. 退款金额一般在 1-7 个工作日原路返回\n"
                    "4. 在订单页面点击「取消订单」或联系客服"
                )

            elif scenario == "quality":
                parts.append(
                    "**⭐ 关于菜品质量**:\n"
                    "1. 请拍照记录问题，便于商家了解情况\n"
                    "2. 建议直接联系商家反馈，多数商家会积极处理\n"
                    "3. 您可以在订单评价中如实反映\n"
                    "4. 如商家不处理，可申请平台客服介入"
                )

            elif scenario == "price":
                parts.append(
                    "**💵 关于价格问题**:\n"
                    "1. 请确认是否使用了优惠券或满减活动\n"
                    "2. 配送费根据距离和时段可能有所调整\n"
                    "3. 如有疑问可查看订单明细，或联系商家确认\n"
                    "4. 收费异常可联系平台客服核实"
                )

            elif scenario == "general":
                parts.append(
                    "**💬 常见问题自助查询**:\n"
                    "• 查订单状态 → 试试说「查一下我的订单」\n"
                    "• 推荐菜品 → 试试说「推荐几个菜」\n"
                    "• 配送进度 → 试试说「配送情况」\n"
                    "• 联系商家 → 在订单页面点击联系商家按钮"
                )

        return "\n\n".join(parts)
