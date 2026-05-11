"""
RAG Retry: 重试与降级机制
==========================
功能：
1. retry_with_fallback — 带重试和降级的通用装饰器/函数
2. 支持指数退避 + 随机抖动（避免惊群）
3. fallback 链：主逻辑失败 → 依次尝试备选方案 → 返回最简结果
4. 适用于 LLM 调用、向量检索、BM25 检索等可能失败的场景

参考 nanobot 的 retry 设计，适配 RAG 场景
"""

import asyncio
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from utils.logger import logger

T = TypeVar("T")


def retry_with_fallback(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    fallback_func: Optional[Callable[..., T]] = None,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    name: str = "RAG",
) -> Callable:
    """
    带指数退避 + 随机抖动 + 降级回退的重试装饰器。

    适合同步函数。

    Args:
        max_retries: 最大重试次数
        base_delay: 基础等待时间（秒）
        max_delay: 最大等待时间（秒），防止退避过大
        backoff_factor: 退避因子，每次重试 delay *= backoff_factor
        fallback_func: 所有重试失败后的降级函数
        jitter: 是否加入随机抖动（默认 True），防止惊群
        retryable_exceptions: 可重试的异常元组
        name: 日志中的操作名称

    Usage:
        @retry_with_fallback(max_retries=2, fallback_func=some_safe_default)
        def risky_llm_call(query: str) -> str:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc = None
            for attempt in range(1 + max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        if jitter:
                            delay = delay * (0.5 + random.random() * 0.5)
                        logger.warning(
                            f"[{name}] 第 {attempt + 1}/{max_retries + 1} 次重试 "
                            f"({type(e).__name__}: {e}), 等待 {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{name}] 重试 {max_retries + 1} 次均失败: {e}"
                        )

            # 所有重试失败 → 尝试降级函数
            if fallback_func is not None:
                logger.info(f"[{name}] 降级到 fallback 函数")
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fb_e:
                    logger.error(f"[{name}] fallback 也失败了: {fb_e}")

            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


async def retry_async_with_fallback(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    fallback_func: Optional[Callable[..., Any]] = None,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    name: str = "RAG",
    **kwargs: Any,
) -> Any:
    """
    异步版本的重试 + 降级。

    适合直接调用（非装饰器模式），更好地支持异步上下文。

    Args:
        func: 异步函数
        *args: 传给 func 的位置参数
        max_retries: 最大重试次数
        base_delay: 基础等待时间（秒）
        max_delay: 最大等待时间（秒）
        backoff_factor: 退避因子
        fallback_func: 降级函数
        jitter: 是否加入随机抖动
        retryable_exceptions: 可重试的异常元组
        name: 日志中的操作名称
        **kwargs: 传给 func 的关键字参数

    Returns:
        函数执行结果或降级结果

    Usage:
        result = await retry_async_with_fallback(
            risky_llm_call, query=user_query,
            max_retries=2,
            fallback_func=simple_keyword_search,
        )
    """
    last_exc = None
    for attempt in range(1 + max_retries):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                if jitter:
                    delay = delay * (0.5 + random.random() * 0.5)
                logger.warning(
                    f"[{name}] 第 {attempt + 1}/{max_retries + 1} 次异步重试 "
                    f"({type(e).__name__}: {e}), 等待 {delay:.2f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{name}] 异步重试 {max_retries + 1} 次均失败: {e}"
                )

    # 所有重试失败 → 尝试降级函数
    if fallback_func is not None:
        logger.info(f"[{name}] 降级到 async fallback 函数")
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)
        except Exception as fb_e:
            logger.error(f"[{name}] async fallback 也失败了: {fb_e}")

    raise last_exc  # type: ignore[misc]


def build_fallback_chain(*funcs: Callable) -> Callable:
    """
    构建降级链：依次尝试多个策略，直到有一个成功。

    每个函数签名需一致。常用于：BM25 → 关键词 → 空结果 的降级路径。

    Args:
        funcs: 从精确到宽松的函数列表

    Returns:
        包装函数

    Usage:
        safe_search = build_fallback_chain(
            vector_search,    # 1. 精确向量检索
            bm25_search,      # 2. 宽松关键词检索
            lambda q: [],     # 3. 兜底：返回空列表
        )
        results = safe_search(query)
    """
    if not funcs:
        raise ValueError("降级链至少需要 1 个函数")

    def chain(*args: Any, **kwargs: Any) -> Any:
        errors = []
        for i, func in enumerate(funcs):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                errors.append(f"  [{i}]: {type(e).__name__}: {e}")
                logger.warning(f"降级链第 {i} 层失败: {e}")

        logger.error(f"降级链全部失败:\n" + "\n".join(errors))
        return funcs[-1](*args, **kwargs)  # 最后一层无论如何执行

    return chain
