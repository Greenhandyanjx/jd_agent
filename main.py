"""
Streamlit UI 入口：JD-Agent 智能助手
"""
import time
import streamlit as st
from agent.orchestrator import AgentOrchestrator
from agent.core.tool_registry import get_all_tools

# 页面配置
st.set_page_config(
    page_title="JD-Agent 智能助手",
    page_icon="🤖",
    layout="wide",
)

# 标题
st.title("🤖 JD-Agent 智能助手")
st.caption("基于 ReAct + RAG 的自实现 AI Agent 框架 | 参考 OpenClaw + Claude Code 架构")
st.divider()

# 初始化 Session State
if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = AgentOrchestrator()
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 侧边栏
with st.sidebar:
    st.header("🔧 系统状态")
    tools = get_all_tools()
    st.metric("可用工具数", len(tools))

    with st.expander("📋 工具列表"):
        for t in tools:
            st.markdown(f"- **{t.name}**: {t.description}")

    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state["orchestrator"].clear_session()
        st.session_state["messages"] = []
        st.rerun()

    if st.button("📖 查看系统架构", use_container_width=True):
        st.markdown("""
        ### JD-Agent 架构
        1. **Agent Core**: 自实现ReAct循环
        2. **Memory**: 三层记忆（短期/工作/长期）
        3. **RAG**: 查询改写 + 多路召回 + 重排序
        4. **Task Planner**: 复杂任务分解与规划
        5. **Tool System**: Schema校验 + 重试 + 安全沙箱
        """)

# 显示历史消息
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
prompt = st.chat_input("请输入您的问题...")
if prompt:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 处理并显示回复
    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            # 流式输出
            response_container = st.empty()
            full_response = ""

            for chunk in st.session_state["orchestrator"].chat_stream(prompt):
                full_response += chunk
                response_container.markdown(full_response + "▌")

            # 最终显示
            response_container.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
    st.rerun()
