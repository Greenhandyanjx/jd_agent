"""
RAG Schema Validator: 工具参数 Pydantic 校验模块
==================================================
功能：
1. 为 RAG 检索相关函数定义 Pydantic 模型
2. 提供统一的 validate_params() 入口
3. 校验失败时抛出清晰的错误信息
4. 支持嵌套校验和自定义校验规则

参考 nanobot 的 payload-schema-validator skill 设计
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError


class RetrieverParams(BaseModel):
    """检索器参数 Schema"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询文本")
    k: int = Field(default=5, ge=1, le=50, description="返回文档数量")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="向量检索权重")
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="BM25 检索权重")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("查询文本不能为空或全空白字符")
        return stripped


class RerankerParams(BaseModel):
    """重排序器参数 Schema"""
    top_k: int = Field(default=3, ge=1, le=20, description="重排序后保留的文档数")
    use_cross_encoder: bool = Field(default=False, description="是否使用交叉编码器（需手动实现）")


class QueryRewriteParams(BaseModel):
    """查询改写参数 Schema"""
    query: str = Field(..., min_length=1, max_length=2000, description="原始用户查询")
    style: str = Field(default="auto", pattern=r"^(auto|search|expand|decompose|question|keyword)$", description="改写风格")


class HybridRetrieverParams(BaseModel):
    """混合检索参数 Schema"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询文本")
    k: int = Field(default=5, ge=1, le=50, description="最终返回文档数量")
    vector_k: int = Field(default=10, ge=1, le=100, description="向量检索初选数量")
    bm25_k: int = Field(default=10, ge=1, le=100, description="BM25 检索初选数量")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF 融合常数")


class RetryConfigParams(BaseModel):
    """重试配置参数 Schema"""
    max_retries: int = Field(default=3, ge=0, le=10, description="最大重试次数")
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="重试基础延迟（秒）")
    max_delay: float = Field(default=30.0, ge=1.0, le=300.0, description="重试最大延迟（秒）")
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0, description="退避因子")


# ─── 统一校验入口 ─────────────────────────────────────

def validate_params(params: dict, model_class: type[BaseModel]) -> dict:
    """
    校验参数字典是否符合指定的 Pydantic 模型。

    Args:
        params: 待校验的参数字典
        model_class: Pydantic 模型类

    Returns:
        校验后的参数字典（经过类型转换和默认值填充）

    Raises:
        ValueError: 校验失败，包含详细错误信息
    """
    try:
        validated = model_class(**params)
        return validated.model_dump()
    except ValidationError as e:
        # 收集所有错误信息
        error_details = []
        for err in e.errors():
            loc = " -> ".join(str(l) for l in err["loc"])
            msg = err["msg"]
            error_details.append(f"  [{loc}]: {msg}")
        error_msg = (
            f"参数校验失败 (model={model_class.__name__}):\n"
            + "\n".join(error_details)
        )
        raise ValueError(error_msg) from e
