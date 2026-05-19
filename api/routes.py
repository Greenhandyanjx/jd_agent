"""
API Routes: FastAPI 路由定义
=========================================================
JD-Agent 的 RESTful API 层，提供三个核心端点：
  - POST /api/v1/chat         → 聊天（支持 SSE 流式/非流式）
  - GET  /api/v1/health       → 健康检查
  - GET  /api/v1/history      → 获取当前会话的历史消息

架构说明
--------
- AgentOrchestrator 以全局单例方式持有，API 首次启动时自动初始化
- 流式聊天通过 Server-Sent Events (SSE) 推送 token-by-token
- 非流式返回完整 JSON
- 所有会话状态由 orchestrator 内部的 SessionManager 持久化到 JSONL
"""
import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from agent.orchestrator import AgentOrchestrator
from agent.providers.openai_provider import OpenAIProvider
from agent.tools.tool_definitions import register_all_tools


# ─────────────────────────────────────────────────────────
# Pydantic Schemas（请求/响应模型）
# ─────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str = Field(..., description="用户输入的消息", min_length=1)
    session_id: Optional[str] = Field(None, description="会话 ID（不传则自动生成）")
    stream: bool = Field(False, description="是否使用 SSE 流式返回")


class ChatResponse(BaseModel):
    """聊天响应体（非流式）"""
    response: str = Field(..., description="AI 回复文本")
    session_id: str = Field(..., description="会话 ID")
    tools_used: list[str] = Field(default_factory=list, description="本轮用到的工具列表")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.0.0"
    tools_count: int = 0
    skills_count: int = 0
    skills_names: list[str] = []
    sessions_count: int = 0


class HistoryMessage(BaseModel):
    """单条历史消息"""
    role: str = Field(..., description="角色: user / assistant / tool")
    content: Optional[str] = Field(None, description="消息内容")
    tool_calls: Optional[list[dict]] = Field(None, description="工具调用信息")


class HistoryResponse(BaseModel):
    """历史消息响应"""
    history: list[HistoryMessage] = Field(default_factory=list, description="消息列表")
    total: int = 0


# ─────────────────────────────────────────────────────────
# AgentOrchestrator 全局单例
# ─────────────────────────────────────────────────────────

_orchestrator: AgentOrchestrator | None = None


async def get_orchestrator() -> AgentOrchestrator:
    """
    获取（并在首次时初始化）全局 AgentOrchestrator 单例。

    初始化流程：
      1. 确定 Provider（优先 DEEPSEEK_API_KEY / OPENAI_API_KEY，
         否则回退到通义千问）
      2. 创建 Orchestrator
      3. 初始化内部组件（AgentLoop、SessionManager、Memory 等）
      4. 注册内置工具
    """
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    # ── 1. 确定 LLM Provider ──
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEFAULT_API_KEY")
    )
    api_base = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://api.deepseek.com"
    )
    model = os.environ.get("DEFAULT_MODEL", "deepseek-chat")

    workspace = os.environ.get(
        "JD_AGENT_WORKSPACE",
        str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    skills_dir = os.environ.get(
        "JD_AGENT_SKILLS_DIR",
        os.path.join(workspace, "skills")
    )

    if api_key:
        logger.info(f"[API] 使用 OpenAIProvider: model={model}, base={api_base}")
        provider = OpenAIProvider(api_key=api_key, api_base=api_base, model=model)
    else:
        logger.info("[API] 未配置 API Key，尝试使用通义千问")
        from agent.providers.tongyi_provider import TongyiProvider
        provider = TongyiProvider(model="qwen-plus")

    # ── 2. 创建 Orchestrator（自动初始化技能系统）──
    orchestrator = AgentOrchestrator(
        provider=provider,
        workspace=workspace,
        skills_dir=skills_dir,  # 技能自动发现目录
    )
    await orchestrator.initialize()

    # ── 3. 注册内置业务工具 ──
    registry = orchestrator.get_tool_registry()
    if registry is not None:
        register_all_tools(registry)
        logger.info(f"[API] 已注册 {len(registry)} 个业务工具")

    _orchestrator = orchestrator
    logger.info("[API] AgentOrchestrator 初始化完成")
    return orchestrator


# ─────────────────────────────────────────────────────────
# Router 定义
# ─────────────────────────────────────────────────────────

router = APIRouter()


# ─── Health ──────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    GET /api/v1/health — 健康检查端点

    返回服务状态、版本号、已注册工具数、活跃会话数。
    """
    orch = await get_orchestrator()
    tools = orch.get_all_tools()
    # 统计 sessions 目录中的会话文件数
    sessions_count = 0
    try:
        sessions_dir = orch.workspace / "sessions"
        if sessions_dir.exists():
            sessions_count = len(list(sessions_dir.glob("*.jsonl")))
    except Exception:
        pass
    # 技能系统信息
    skills_count = 0
    skills_names = []
    if orch.loop:
        skills_count = orch.loop.skill_manager.count
        skills_names = [s.name for s in orch.loop.skill_manager.get_all()]

    return HealthResponse(
        status="ok",
        version="1.0.0",
        tools_count=len(tools),
        skills_count=skills_count,
        skills_names=skills_names,
        sessions_count=sessions_count,
    )


# ─── Chat ────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    POST /api/v1/chat — 聊天接口

    支持两种模式：
      1. 非流式（stream=false）→ 返回 JSON { response, session_id, tools_used }
      2. 流式   （stream=true） → 返回 SSE 流，每行一个 data: chunk
         - 最终会收到 data: [DONE] 标记结束
         - 使用 &#x60;Accept: text/event-stream&#x60; 消费

    参数:
      - message:    用户消息（必填）
      - session_id: 会话 ID（选填，不传则自动生成）
      - stream:     是否流式（默认 false）
    """
    orch = await get_orchestrator()
    session_id = request.session_id or str(uuid.uuid4())[:8]

    # ── 非流式模式 ──
    if not request.stream:
        try:
            response_text = await orch.chat_async(
                request.message,
                session_key=f"api:{session_id}",
            )
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                tools_used=[],  # 非流式下暂时不追踪工具列表
            )
        except Exception as e:
            logger.error(f"[API] 非流式聊天失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── 流式模式 (SSE) ──
    async def event_stream():
        """
        SSE 生成器:
          每一块文本以 data: <text>\n\n 格式推送
          最终以 data: [DONE]\n\n 标记结束
        """
        try:
            async for chunk in orch.chat_stream_async(
                request.message,
                session_key=f"api:{session_id}",
            ):
                # JSON 编码确保换行符等特殊字符不出问题
                encoded = json.dumps({"text": chunk}, ensure_ascii=False)
                yield f"data: {encoded}\n\n"
        except Exception as e:
            logger.error(f"[API] 流式聊天失败: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ─── History ─────────────────────────────────────────────

@router.get("/history", response_model=HistoryResponse)
async def get_history(session_key: Optional[str] = Query(None, description="会话标识")):
    """
    GET /api/v1/history — 获取会话历史

    参数:
      - session_key: 会话标识（选填，默认返回 "cli:direct" 会话）
    """
    orch = await get_orchestrator()
    if orch.loop is None:
        return HistoryResponse(history=[], total=0)

    key = session_key or "cli:direct"
    try:
        session = orch.loop.sessions.get_or_create(key)
        history = session.get_history(max_messages=200)
        # 过滤敏感字段，只保留 role / content / tool_calls
        cleaned = []
        for msg in history:
            entry = {"role": msg.get("role", "")}
            if msg.get("content"):
                entry["content"] = msg["content"]
            if msg.get("tool_calls"):
                entry["tool_calls"] = msg["tool_calls"]
            cleaned.append(entry)
        return HistoryResponse(history=cleaned, total=len(cleaned))
    except Exception as e:
        logger.error(f"[API] 获取历史失败: {e}")
        return HistoryResponse(history=[], total=0)
