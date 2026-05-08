"""
API Routes: FastAPI 路由定义
"""
import uuid
from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse, ToolsResponse, ToolInfo, HealthResponse
from agent.orchestrator import AgentOrchestrator
from agent.core.tool_registry import get_all_tools

router = APIRouter()
orchestrator = AgentOrchestrator()

# 会话存储（简单内存实现，生产环境应使用 Redis）
sessions: dict[str, AgentOrchestrator] = {}


def _get_or_create_session(session_id: str = None) -> tuple[AgentOrchestrator, str]:
    """获取或创建会话"""
    if session_id and session_id in sessions:
        return sessions[session_id], session_id
    new_id = session_id or str(uuid.uuid4())[:8]
    sessions[new_id] = AgentOrchestrator()
    return sessions[new_id], new_id


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    tools = get_all_tools()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        tools_count=len(tools),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    agent, session_id = _get_or_create_session(request.session_id)
    response = agent.chat(request.message)
    return ChatResponse(response=response, session_id=session_id)


@router.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """获取可用工具列表"""
    tools = get_all_tools()
    tool_infos = [
        ToolInfo(
            name=t.name,
            description=t.description,
            parameters=t.schema.parameters,
        )
        for t in tools
    ]
    return ToolsResponse(tools=tool_infos)


@router.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """清空会话"""
    if session_id in sessions:
        sessions[session_id].clear_session()
        return {"status": "ok", "message": f"会话 {session_id} 已清空"}
    raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
