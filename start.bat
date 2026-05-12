@echo off
REM ─────────────────────────────────────────────────────────
REM JD-Agent 一键启动脚本 (Windows)
REM ─────────────────────────────────────────────────────────
REM 使用方式:
REM   1. 配置 .env（复制 .env.example 并填入 API Key）
REM   2. 双击 start.bat 或命令行运行
REM ─────────────────────────────────────────────────────────

echo ============================================
echo   JD-Agent 启动脚本
echo   前后端分离模式: Streamlit + FastAPI
echo ============================================

REM ── 检查虚拟环境 ──
if exist venv\Scripts\activate (
    echo [✓] 使用虚拟环境
    call venv\Scripts\activate
) else (
    echo [!] 未找到 venv，使用全局 Python
)

REM ── 配置参数 ──
set API_PORT=8000
set STREAMLIT_PORT=8501

echo.
echo [1/2] 启动 FastAPI 后端 (端口 %API_PORT%)...
echo       查看文档: http://localhost:%API_PORT%/docs
echo.

REM 后台启动 API
start "JD-Agent API" cmd /c "uvicorn api.app:app --host 127.0.0.1 --port %API_PORT% --reload"
REM 等待 API 启动
timeout /t 5 /nobreak >nul

echo.
echo [2/2] 启动 Streamlit 前端 (端口 %STREAMLIT_PORT%)...
echo       打开浏览器: http://localhost:%STREAMLIT_PORT%
echo.
echo 按 Ctrl+C 停止服务
echo.

streamlit run main.py --server.port %STREAMLIT_PORT%

REM ── 如果 Streamlit 窗口关闭，也清理 API ──
echo 正在清理 API 进程...
taskkill /f /fi "WINDOWTITLE eq JD-Agent API" >nul 2>&1
echo [✓] 已停止所有服务
