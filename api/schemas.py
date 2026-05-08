"""API 请求/响应 Schema 定义"""
from pydantic import BaseModel, Field
from typing import Optional, Any


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID（用于多会话）")
    stream: bool = Field(False, description="是否流式输出")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 回复")
    session_id: str = Field("", description="会话ID")


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    tools_count: int = 0
