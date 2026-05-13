"""
Delivery Agent: 配送系统配置
=================================
定义配送后端的连接信息、可用商家列表等配置常量。

环境变量覆盖优先级高于默认值。
"""

import os

# 配送后端服务的 URL 地址
# 默认为本地开发环境 http://localhost:8080
# 可通过环境变量 DELIVERY_BACKEND_URL 覆盖
DELIVERY_BACKEND_URL = os.environ.get(
    "DELIVERY_BACKEND_URL",
    "http://localhost:8080",
)

# 系统中可用的商家 ID 列表
# 用于菜品推荐工具遍历所有商家获取菜品
AVAILABLE_MERCHANT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# 菜品推荐最大请求数量（每个商家）
DISHES_PAGE_SIZE = 50

# HTTP 请求超时时间（秒）
HTTP_TIMEOUT = 15.0
