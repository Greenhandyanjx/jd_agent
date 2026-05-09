"""
Streamlit UI 入口：JD-Agent 智能助手（v2 — nanobot 兼容架构）
参考 nanobot 的设计，使用异步 AgentOrchestrator 驱动
"""
import asyncio
import time
import streamlit as st
from loguru import logger

from agent.orchestrator import AgentOrchestrator
from agent.providers.openai_provider import OpenAIProvider
from agent.tools.tool_definitions import register_all_tools


def get_provider():
    """
    获取 LLM Provider。
    
    优先读取环境变量，否则使用默认配置。
    支持：
    - DEEPSEEK_API_KEY / DEFAULT_API_KEY + api_base → OpenAIProvider
    - 通义千问 → TongyiProvider
    """
    import os

    # 尝试多种 API Key 来源
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

    if api_key:
        logger.info(f"[Provider] 使用 OpenAIProvider: model={model}, base={api_base}")
        return OpenAIProvider(api_key=api_key, api_base=api_base, model=model)

    # 回退到通义千问
    logger.info("[Provider] 未配置 API Key，尝试使用通义千问")
    from agent.providers.tongyi_provider import TongyiProvider
    return TongyiProvider(model="qwen-plus")


# ─── Streamlit 页面配置 ──────────────────────

st.set_page_config(
    page_title="JD-Agent 智能助手",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 JD-Agent 智能助手")
st.caption(
    "基于 ReAct 架构 · 参考 OpenClaw → nanobot 设计 | "
    "支持工具调用 · 三层记忆 · 会话持久化"
)
st.divider()

# ─── 初始化 ──────────────────────────────────

if "initialized" not in st.session_state:
    st.session_state.provider = get_provider()
    st.session_state.orchestrator = AgentOrchestrator(
        provider=st.session_state.provider,
        workspace="D:\\桌面\\深度学习\\jd_agent",
    )
    # 注册业务工具
    register_all_tools(st.session_state.orchestrator.get_tool_registry())
    st.session_state.initialized = True

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── 侧边栏 ──────────────────────────────────

with st.sidebar:
    st.header("🔧 系统状态")

    # 工具列表
    tools = st.session_state.orchestrator.get_tool_names()
    st.metric("可用工具数", len(tools))

    with st.expander("📋 工具列表"):
        for t_name in tools:
            st.markdown(f"- **{t_name}**")

    # 会话管理
    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state.orchestrator.clear_session()
        st.session_state.messages = []
        st.rerun()

    # 架构信息
    with st.expander("📖 架构说明"):
        st.markdown("""
        ### JD-Agent v2 架构（参考 nanobot）
        
        | 层 | 功能 |
        |---|---|
        | **AgentLoop** | ReAct 循环引擎（Think→Act→Observe） |
        | **MessageBus** | 异步消息总线 |
        | **LLMProvider** | 模型抽象（切换模型） |
        | **ToolRegistry** | 工具注册与调度 |
        | **SessionManager** | 会话 JSONL 持久化 |
        | **MemoryStore** | MEMORY.md + HISTORY.md |
        | **MemoryConsolidator** | 自动记忆总结 |
        """)

# ─── 消息显示 ────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── 用户输入 ────────────────────────────────

prompt = st.chat_input("请输入您的问题...")
if prompt:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 处理并显示回复
    with st.chat_message("assistant"):
        import asyncio as _asyncio

        response_container = st.empty()
        full_response = ""

        with st.spinner("🤔 思考中..."):
            try:
                # 流式输出
                async def stream_chat():
                    nonlocal full_response
                    async for chunk in st.session_state.orchestrator.chat_stream_async(prompt):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                    return full_response

                result = _asyncio.run(stream_chat())
                response_container.markdown(result)

            except Exception as e:
                error_msg = f"抱歉，处理时出错: {str(e)}"
                response_container.error(error_msg)
                logger.error(f"[Streamlit] 处理失败: {e}")
                result = error_msg

    st.session_state.messages.append({"role": "assistant", "content": result})
    st.rerun()
